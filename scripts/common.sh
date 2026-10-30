#!/usr/bin/env bash
# scripts/common.sh — funções compartilhadas pelos sbatch do educability-annotator na DGX.
#
# Sourcear depois do config.env:
#   source "${PROJECT_DIR}/config.env"
#   source "${PROJECT_DIR}/scripts/common.sh"
#
# cleanup_orphan_scratches e o padrão de trap vêm do data-scraping, com as correções
# dos jobs 7712 (guarda do squeue) e 7785/7786 (container em background + wait).

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

# Aborta se alguma das variáveis estiver vazia ou indefinida.
require_vars() {
    local _missing=0 _v
    for _v in "$@"; do
        if [[ -z "${!_v:-}" ]]; then
            echo "[ERRO] variável obrigatória não definida: ${_v}" >&2
            _missing=1
        fi
    done
    [[ ${_missing} -eq 0 ]] || exit 2
}

# Nome no formato {project}-{role}-{JOBID}[-t{TASK}][-{passo}], o padrão dos projetos do pipeline.
container_name() {
    local _role="$1" _step="${2:-}"
    local _job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
    local _name="${PROJECT_NAME:-educability-annotator}-${_role}-${_job}"
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && _name="${_name}-t${SLURM_ARRAY_TASK_ID}"
    [[ -n "${_step}" ]] && _name="${_name}-${_step}"
    printf '%s' "${_name}" | tr -c 'A-Za-z0-9_.-' '-'
}

# Scratch deste job: ${SCRATCH_BASE}/educability-annotator_<jobid>[_t<task>].
job_scratch() {
    local _job="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
    local _dir="${SCRATCH_BASE}/${PROJECT_NAME:-educability-annotator}_${_job}"
    [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] && _dir="${_dir}_t${SLURM_ARRAY_TASK_ID}"
    printf '%s' "${_dir}"
}

# Constrói a imagem só se ela ainda não existir (padrão lazy_build).
# O docker build lê o contexto pelo cliente, então o contexto pode ficar no NFS.
lazy_build() {
    local _image="$1" _dockerfile="$2" _context="$3"
    shift 3  # o resto vai para o docker build (ex.: --build-arg BASE_IMAGE=...)
    if docker image inspect "${_image}" &>/dev/null; then
        log "Imagem ${_image} já existe"
        return 0
    fi
    log "Construindo ${_image} (${_dockerfile})"
    docker build -t "${_image}" -f "${_dockerfile}" "$@" "${_context}"
}

# Alvos do docker/Dockerfile com os argumentos do config.env: toolchain (Python + uv, gera o lock)
# e runtime (dependências do lock + flash-attn). Uso: lazy_build_edu toolchain|runtime
lazy_build_edu() {
    local _target="$1" _image
    require_vars EDU_BASE_IMAGE UV_VERSION PROJECT_DIR
    case "${_target}" in
        toolchain) require_vars EDU_TOOLCHAIN_IMAGE; _image="${EDU_TOOLCHAIN_IMAGE}" ;;
        runtime)
            require_vars EDU_IMAGE FLASH_ATTN_VERSION FLASH_ATTN_SHA256_ABI_TRUE FLASH_ATTN_SHA256_ABI_FALSE
            _image="${EDU_IMAGE}"
            if [[ ! -s "${PROJECT_DIR}/docker/requirements.gpu.lock" ]]; then
                echo "[ERRO] falta docker/requirements.gpu.lock: rode slurm/00_lock_deps.sbatch" >&2
                return 2
            fi
            ;;
        *) echo "[ERRO] alvo desconhecido: ${_target}" >&2; return 2 ;;
    esac
    lazy_build "${_image}" "${PROJECT_DIR}/docker/Dockerfile" "${PROJECT_DIR}" --target "${_target}" \
        --build-arg "BASE_IMAGE=${EDU_BASE_IMAGE}" --build-arg "UV_VERSION=${UV_VERSION}" \
        --build-arg "FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-}" \
        --build-arg "FLASH_ATTN_SHA256_ABI_TRUE=${FLASH_ATTN_SHA256_ABI_TRUE:-}" \
        --build-arg "FLASH_ATTN_SHA256_ABI_FALSE=${FLASH_ATTN_SHA256_ABI_FALSE:-}"
}

# GPUs físicas do job (sbatch: SLURM_JOB_GPUS; numa alocação srun: SLURM_STEP_GPUS). Vazio sem GPU.
job_gpus() {
    printf '%s' "${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"
}

# Copia código e arquivos pequenos para o scratch. --checksum: logo depois de editar no
# master, o cache de atributos do NFS no nó pode fazer o rsync pular o arquivo.
stage_in_code() {
    local _dst="$1"
    shift
    mkdir -p "${_dst}"
    rsync -a --checksum "$@" "${_dst}/"
}

# Remove scratches órfãos (sem job ativo, mais velhos que CLEANUP_MIN_AGE_MIN).
# Uso: cleanup_orphan_scratches <imagem para rm root-owned> <padrão> [padrão...]
# Só olha pastas de job (educability-annotator_<jobid>...); CPT_DATA_ROOT nunca casa com o padrão.
cleanup_orphan_scratches() {
    local _image="$1"
    shift
    local _min_age="${CLEANUP_MIN_AGE_MIN:-60}"
    [[ $# -eq 0 ]] && return 0
    if [[ -z "${SCRATCH_BASE:-}" ]] || [[ ! -d "${SCRATCH_BASE}" ]]; then
        return 0
    fi
    log "[CLEANUP] SCRATCH_BASE=${SCRATCH_BASE} | idade>${_min_age}min | padrões: $*"
    local _name_expr=( '(' )
    local _i=0 _pat
    for _pat in "$@"; do
        [[ $_i -gt 0 ]] && _name_expr+=( -o )
        _name_expr+=( -name "${_pat}" )
        _i=$((_i + 1))
    done
    _name_expr+=( ')' )
    # Job ids ainda na fila (%A = job da task, %F = job do array, que nomeia o scratch).
    # Sem isso, uma task longa tem o scratch apagado por uma irmã que começa depois
    # (job 7712 do data-scraping).
    local _active=" "
    if command -v squeue &>/dev/null; then
        _active=" $(squeue -h -u "${USER:-$(whoami)}" -o '%A %F' 2>/dev/null | tr '\n' ' ') "
    fi

    local _count=0 _failed=0 _kept=0 _orphan
    while IFS= read -r -d '' _orphan; do
        local _p _b _jid _skip=0
        _p=$(dirname "${_orphan}")
        _b=$(basename "${_orphan}")
        # Preserva se qualquer número do nome for de job ativo: preservar demais só adia
        # a limpeza; apagar de menos destrói trabalho em andamento.
        for _jid in $(grep -oE '[0-9]+' <<< "${_b}"); do
            if [[ "${_active}" == *" ${_jid} "* ]]; then
                log "[CLEANUP] Preservando ${_b} — job ${_jid} ainda ativo"
                _kept=$((_kept + 1))
                _skip=1
                break
            fi
        done
        [[ ${_skip} -eq 1 ]] && continue

        _count=$((_count + 1))
        log "[CLEANUP] Removendo ${_orphan}"
        if rm -rf "${_orphan}" 2>/dev/null; then
            :
        elif docker run --rm \
                --name "$(container_name cleanup "${_count}")" \
                --label "project=${PROJECT_NAME:-educability-annotator}" --label "slurm_job=${SLURM_JOB_ID:-local}" \
                --label "role=cleanup" \
                --entrypoint rm -v "${_p}:/to_clean" "${_image}" -rf "/to_clean/${_b}" 2>/dev/null; then
            :
        else
            log "[WARN] Limpeza falhou para ${_orphan} (dono root? imagem indisponível?)"
            _failed=$((_failed + 1))
        fi
    done < <(find "${SCRATCH_BASE}" -maxdepth 1 -mindepth 1 "${_name_expr[@]}" \
        -type d -mmin "+${_min_age}" -print0 2>/dev/null)
    log "[CLEANUP] ${_count} removidos, ${_kept} preservados (job ativo), ${_failed} falhas"
}

# TERM/INT (scancel ou limite de tempo): mata os containers deste job. O scancel sozinho
# mata só o cliente docker e o container segue vivo (lição do job 7688 do nemo-framework).
on_term() {
    log "Sinal recebido: matando os containers do job ${SLURM_JOB_ID:-local}"
    docker ps -q --filter "label=slurm_job=${SLURM_JOB_ID:-local}" 2>/dev/null \
        | xargs -r docker kill >/dev/null 2>&1 || true
}

# CPUs que o Slurm deu a este job (afinidade do shell do sbatch), ex.: "10-11,122-123". Vazio fora
# do Slurm.
slurm_cpuset() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || return 0
    taskset -cp $$ 2>/dev/null | sed 's/.*: //'
}

# Roda um container em background e espera por ele. O bash não executa trap enquanto
# espera um comando em primeiro plano; com `wait`, o sinal interrompe a espera e o trap
# roda na hora. Uso: run_container <role> <argumentos do docker run, imagem e comando>
# O Docker ignora o cgroup do Slurm: sem --cpuset-cpus, o container roda em qualquer núcleo,
# inclusive nos de outro job (ex.: o treino do outro grupo). Aqui ele fica nos núcleos do job.
run_container() {
    local _role="$1" _cpuset
    shift
    _cpuset="$(slurm_cpuset)"
    [[ -n "${_cpuset}" ]] && set -- --cpuset-cpus="${_cpuset}" "$@"
    docker run --rm --init \
        --name "$(container_name "${_role}")" \
        --label "project=${PROJECT_NAME:-educability-annotator}" \
        --label "slurm_job=${SLURM_JOB_ID:-local}" \
        --label "role=${_role}" \
        --user "$(id -u):$(id -g)" \
        -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
        "$@" &
    local _pid=$! _rc=0
    wait "${_pid}" || _rc=$?
    # wait volta cedo quando um trap dispara; espera o cliente docker terminar de fato.
    while kill -0 "${_pid}" 2>/dev/null; do
        wait "${_pid}" || _rc=$?
    done
    return "${_rc}"
}
