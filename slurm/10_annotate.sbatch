#!/usr/bin/env bash
#SBATCH --job-name=edu-annotate
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# TERM 10 min antes do limite: o trap mata os containers, e o que já tem commit fica.
#SBATCH --signal=B:TERM@600
#
# FASES 3.2 e 5.1: anota um corpus Parquet com o motor de rótulos laterais (python -m
# educability_annotator.sidecar). Os passos são:
#   1. prepare: plano imutável + nova tentativa;
#   2. torchrun com um rank por GPU do job e os workers do DataLoader;
#   3. verify.
# Para retomar, submeta de novo: tarefas com commit não são refeitas. Relatórios em reports/runs/<job>/.
#
# Entradas (ANNOTATE_INPUT):
#   corpus  unified/${CPT_VERSION} menos as decisões do dedup em EDU_EXCLUDE → ${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}
#   pilot   amostra da FASE 3.1 (${EDU_PILOT_ROOT}/sample) → ${EDU_PILOT_ROOT}/labels/${EDU_MODEL_TAG}
# ANNOTATE_OUTPUT troca o diretório de saída (ex.: outra precisão, que exige outra saída).
#
#   sbatch --export=ALL,ANNOTATE_INPUT=pilot slurm/10_annotate.sbatch
#   sbatch --exclusive --cpus-per-task=224 --mem=0 slurm/10_annotate.sbatch      # corpus inteiro (E4)
#
# Em CPU (EDU_DEVICE=cpu): vários processos de inferência FP32 com EDU_TORCH_THREADS threads cada, um
# worker de tokenização por processo; a saída padrão ganha o sufixo -cpu-fp32 (outro contrato).
#   sbatch --gres=gpu:0 --cpus-per-task=160 --export=ALL,EDU_DEVICE=cpu,ANNOTATE_INPUT=pilot slurm/10_annotate.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE EDU_HF_HOME EDU_MODEL EDU_MODEL_REVISION \
    EDU_MODEL_TAG CPT_DATA_ROOT CPT_VERSION EDU_LABELS_ROOT EDU_PILOT_ROOT EDU_GPU_POOL

DEVICE="${EDU_DEVICE:-cuda}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
GPUS="$(job_gpus)"
SUFFIX=""
if [[ "${DEVICE}" == cpu ]]; then
    TORCH_THREADS="${EDU_TORCH_THREADS:-4}"
    RANKS="${EDU_CPU_RANKS:-$(( CPUS / (TORCH_THREADS + 1) ))}"
    WORKERS="${EDU_LOADER_WORKERS:-1}"
    DTYPE="${EDU_DTYPE:-float32}"
    ATTENTION="${EDU_ATTENTION:-sdpa}"
    SUFFIX="-cpu-${DTYPE/float/fp}"
    GPU_ARGS=()
else
    if [[ -z "${GPUS}" ]]; then
        log "[ERRO] sem GPU no job: use --gres=gpu:N (ou EDU_DEVICE=cpu)"
        exit 1
    fi
    # EDU_RANKS_PER_GPU > 1 põe mais de um processo por GPU, se um só não a saturar (ver o benchmark 2.6).
    PER_GPU="${EDU_RANKS_PER_GPU:-1}"
    RANKS=$(( $(awk -F, '{print NF}' <<< "${GPUS}") * PER_GPU ))
    for gpu in ${GPUS//,/ }; do
        [[ ",${EDU_GPU_POOL}," == *",${gpu},"* ]] || log "[AVISO] GPU ${gpu} fora de EDU_GPU_POOL=${EDU_GPU_POOL}"
    done
    TORCH_THREADS=1
    # Um processo principal por rank; o resto das CPUs vira worker do DataLoader (tokenização).
    WORKERS="${EDU_LOADER_WORKERS:-$(( CPUS / RANKS - 1 ))}"
    DTYPE="${EDU_DTYPE:-bfloat16}"
    ATTENTION="${EDU_ATTENTION:-flash_attention_2}"
    GPU_ARGS=(--gpus "\"device=${GPUS}\"")
fi
(( WORKERS >= 1 )) || WORKERS=1
(( RANKS >= 1 )) || RANKS=1

INPUT_KIND="${ANNOTATE_INPUT:-corpus}"
EXCLUDE_ARGS=()
case "${INPUT_KIND}" in
    corpus)
        INPUT_DIR="${CPT_DATA_ROOT}/unified/${CPT_VERSION}"
        OUTPUT_DIR="${ANNOTATE_OUTPUT:-${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}${SUFFIX}}"
        for pattern in ${EDU_EXCLUDE:-}; do EXCLUDE_ARGS+=("/data/${pattern}"); done
        ;;
    kept)
        # Partição kept do cpt-corpus (2.7), já sem os removidos pelo dedup: nada a excluir.
        INPUT_DIR="${CPT_DATA_ROOT}/release/${CPT_VERSION}/kept"
        OUTPUT_DIR="${ANNOTATE_OUTPUT:-${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}${SUFFIX}}"
        ;;
    pilot)
        INPUT_DIR="${EDU_PILOT_ROOT}/sample"
        OUTPUT_DIR="${ANNOTATE_OUTPUT:-${EDU_PILOT_ROOT}/labels/${EDU_MODEL_TAG}${SUFFIX}}"
        ;;
    *) log "[ERRO] ANNOTATE_INPUT=${INPUT_KIND} (use corpus, kept ou pilot)"; exit 1 ;;
esac
# Subconjunto das partes (testes): glob relativo à entrada.
PATTERN="${EDU_PATTERN:-source=*/part-*.parquet}"
[[ -d "${INPUT_DIR}" ]] || { log "[ERRO] entrada ${INPUT_DIR} não existe"; exit 1; }

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/runs/${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${REPORT_DIR}" "${OUTPUT_DIR}"

stage_out() {
    local _rc=$?
    set +e
    for file in plan.json contract.json attempt.json; do
        [[ -f "${OUTPUT_DIR}/${file}" ]] && rsync -a "${OUTPUT_DIR}/${file}" "${REPORT_DIR}/"
    done
    [[ -d "${OUTPUT_DIR}/attempts" ]] && rsync -a "${OUTPUT_DIR}/attempts/" "${REPORT_DIR}/attempts/"
    [[ -d "${SCRATCH}/out" ]] && rsync -a "${SCRATCH}/out/" "${REPORT_DIR}/"
    log "Stage-out: ${REPORT_DIR}"
    rm -rf "${SCRATCH}" || log "[WARN] não consegui remover ${SCRATCH}"
    exit "${_rc}"
}
trap stage_out EXIT
trap on_term TERM INT

cleanup_orphan_scratches "${EDU_IMAGE}" "${PROJECT_NAME}_*"
lazy_build_edu runtime
stage_in_code "${SCRATCH}/code" "${PROJECT_DIR}/src"
mkdir -p "${SCRATCH}/out"

MODEL_DIR="/hf/hub/models--${EDU_MODEL//\//--}/snapshots/${EDU_MODEL_REVISION}"
COMMON=(--cpus="${CPUS}" -e HOME=/work -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1
        -e RAYON_NUM_THREADS=1 -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro"
        -v "${EDU_HF_HOME}:/hf:ro" -v "${CPT_DATA_ROOT}:/data:ro" -v "${INPUT_DIR}:/input:ro"
        -v "${OUTPUT_DIR}:/labels" -w /work)

log "Entrada ${INPUT_KIND} (${INPUT_DIR}) → ${OUTPUT_DIR} | ${DEVICE}: ${RANKS} processos × ${TORCH_THREADS} threads," \
    "${WORKERS} workers cada | GPUs: ${GPUS:-nenhuma} | ${CPUS} CPUs | ${DTYPE}, ${ATTENTION}"
run_container prepare "${COMMON[@]}" "${EDU_IMAGE}" \
    python -m educability_annotator.sidecar prepare --input-root /input --pattern "${PATTERN}" \
        ${EXCLUDE_ARGS[@]+--exclude "${EXCLUDE_ARGS[@]}"} --model "${MODEL_DIR}" --offline --output /labels \
        --dtype "${DTYPE}" --attention "${ATTENTION}" \
        --max-document-tokens "${EDU_MAX_DOC_TOKENS:-5110}" --threads "${CPUS}" \
        ${EDU_SAMPLE_TASKS:+--sample-tasks "${EDU_SAMPLE_TASKS}" --seed "${EDU_SAMPLE_SEED:-1}"}

run_container run "${COMMON[@]}" ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} --shm-size="${EDU_SHM_SIZE:-32g}" "${EDU_IMAGE}" \
    torchrun --standalone --nproc-per-node="${RANKS}" -m educability_annotator.sidecar run --output /labels \
        --device "${DEVICE}" --loader-workers "${WORKERS}" --torch-threads "${TORCH_THREADS}" \
        --ranks-per-device "${PER_GPU:-1}" \
        --batch-tokens "${EDU_BATCH_TOKENS:-65536}" --read-rows "${EDU_READ_ROWS:-4096}" \
        --log-interval "${EDU_LOG_INTERVAL:-60}"

rc=0
run_container verify "${COMMON[@]}" "${EDU_IMAGE}" \
    sh -c 'python -m educability_annotator.sidecar verify --output /labels --threads "$1" > /work/out/verify.json' \
        verify "${CPUS}" || rc=$?
log "Verify (rc=${rc}): $(head -c 400 "${SCRATCH}/out/verify.json" | tr -d '\n')"
exit "${rc}"
