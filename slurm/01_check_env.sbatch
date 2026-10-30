#!/usr/bin/env bash
#SBATCH --job-name=edu-check-env
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
#
# FASE 1.1–1.3: verifica o ambiente da ferramenta na DGX. Faz, em ordem:
#   1. constrói a imagem runtime, se faltar;
#   2. baixa o modelo, na revisão fixada, para o cache do NVMe;
#   3. roda no container a suíte do repositório em CPU;
#   4. roda o teste com o modelo publicado (CPU FP32 contra a referência);
#   5. roda o envcheck (versões; com GPU, o teste CUDA e FlashAttention-2 × SDPA no modelo real).
# Os relatórios vão para reports/env/<job>/, mesmo em falha.
#
# Submeter da raiz do projeto:
#   sbatch slurm/01_check_env.sbatch                   # só CPU
#   sbatch --gres=gpu:1 slurm/01_check_env.sbatch      # + CUDA e FlashAttention-2 (turno de GPU)
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE EDU_HF_HOME EDU_MODEL EDU_MODEL_REVISION EDU_GPU_POOL

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/env/${TAG}"
mkdir -p "${PROJECT_DIR}/logs" "${REPORT_DIR}" "${EDU_HF_HOME}"

stage_out() {
    local _rc=$?
    set +e
    [[ -d "${SCRATCH}/out" ]] && rsync -a "${SCRATCH}/out/" "${REPORT_DIR}/" && log "Stage-out: ${REPORT_DIR}"
    rm -rf "${SCRATCH}" || log "[WARN] não consegui remover ${SCRATCH}"
    exit "${_rc}"
}
trap stage_out EXIT
trap on_term TERM INT

GPUS="$(job_gpus)"
log "Job ${TAG} em $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs, GPUs: ${GPUS:-nenhuma}"
for gpu in ${GPUS//,/ }; do
    [[ ",${EDU_GPU_POOL}," == *",${gpu},"* ]] || log "[AVISO] GPU ${gpu} fora de EDU_GPU_POOL=${EDU_GPU_POOL}"
done

cleanup_orphan_scratches "${EDU_IMAGE}" "${PROJECT_NAME}_*"
lazy_build_edu runtime
stage_in_code "${SCRATCH}/code" "${PROJECT_DIR}/src" "${PROJECT_DIR}/tests" "${PROJECT_DIR}/tools" \
    "${PROJECT_DIR}/pyproject.toml"
mkdir -p "${SCRATCH}/out"

# Índices físicos do job; com --gpus all o container veria as GPUs de outros jobs.
GPU_ARGS=()
[[ -n "${GPUS}" ]] && GPU_ARGS=(--gpus "\"device=${GPUS}\"")
COMMON=(--cpus="${SLURM_CPUS_PER_TASK:-2}" -e HOME=/work -e HF_HOME=/hf -e OMP_NUM_THREADS=1
        -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" -v "${EDU_HF_HOME}:/hf"
        -w /opt/educability-annotator)
MODEL_DIR="/hf/hub/models--${EDU_MODEL//\//--}/snapshots/${EDU_MODEL_REVISION}"

run_container model-download "${COMMON[@]}" "${EDU_IMAGE}" \
    hf download "${EDU_MODEL}" --revision "${EDU_MODEL_REVISION}" --include '*.json' '*.safetensors'
log "Modelo em ${EDU_HF_HOME} (${MODEL_DIR#/hf/})"

# Sem cache do pytest: o código está montado só para leitura.
run_container tests "${COMMON[@]}" "${EDU_IMAGE}" \
    python -m pytest -q -p no:cacheprovider -m 'not hub and not cuda' --junitxml=/work/out/tests-cpu.xml
log "Suíte em CPU: ok"

run_container tests-hub "${COMMON[@]}" -e HF_HUB_OFFLINE=1 -e EDUCABILITY_TEST_HUB=1 \
    -e EDUCABILITY_TEST_MODEL="${MODEL_DIR}" "${EDU_IMAGE}" \
    python -m pytest -q -s -p no:cacheprovider -m hub tests/test_hardware.py --junitxml=/work/out/tests-hub.xml
log "Modelo publicado contra a referência: ok"

if [[ -n "${GPUS}" ]]; then
    run_container tests-cuda "${COMMON[@]}" "${GPU_ARGS[@]}" "${EDU_IMAGE}" \
        python -m pytest -q -p no:cacheprovider -m cuda --junitxml=/work/out/tests-cuda.xml
    log "Teste CUDA: ok"
fi

run_container envcheck "${COMMON[@]}" "${GPU_ARGS[@]}" -e HF_HUB_OFFLINE=1 "${EDU_IMAGE}" \
    python tools/envcheck.py --model "${MODEL_DIR}" --out "/work/out/envcheck_${TAG}.json"
log "Envcheck: ok"
