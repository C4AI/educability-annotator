#!/usr/bin/env bash
#SBATCH --job-name=edu-bench-gpu
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@120
#
# FASES 2.5 e 2.6 (1 GPU): paridade e vazão das configurações do motor contra a referência FP32 e a
# ferramenta original, sobre a amostra do piloto (FASE 3.1). Relatório em reports/bench/.
#
#   sbatch slurm/11_bench_gpu.sbatch
#   sbatch --export=ALL,BENCH_PER_SOURCE=500 slurm/11_bench_gpu.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE EDU_HF_HOME EDU_MODEL EDU_MODEL_REVISION EDU_PILOT_ROOT

GPUS="$(job_gpus)"
[[ -n "${GPUS}" ]] || { log "[ERRO] sem GPU no job: use --gres=gpu:1"; exit 1; }
SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/bench"
mkdir -p "${PROJECT_DIR}/logs" "${REPORT_DIR}"

stage_out() {
    local _rc=$?
    set +e
    [[ -d "${SCRATCH}/out" ]] && rsync -a "${SCRATCH}/out/" "${REPORT_DIR}/" && log "Stage-out: ${REPORT_DIR}"
    rm -rf "${SCRATCH}" || log "[WARN] não consegui remover ${SCRATCH}"
    exit "${_rc}"
}
trap stage_out EXIT
trap on_term TERM INT

cleanup_orphan_scratches "${EDU_IMAGE}" "${PROJECT_NAME}_*"
lazy_build_edu runtime
stage_in_code "${SCRATCH}/code" "${PROJECT_DIR}/src" "${PROJECT_DIR}/tools"
mkdir -p "${SCRATCH}/out"

log "GPU ${GPUS} | amostra ${EDU_PILOT_ROOT}/sample, ${BENCH_PER_SOURCE:-2000} docs por fonte"
run_container bench-gpu --cpus="${SLURM_CPUS_PER_TASK:-2}" --gpus "\"device=${GPUS}\"" \
    -e HOME=/work -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" \
    -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" -v "${EDU_HF_HOME}:/hf:ro" \
    -v "${EDU_PILOT_ROOT}/sample:/sample:ro" -w /opt/educability-annotator "${EDU_IMAGE}" \
    python tools/bench_gpu.py --sample /sample \
        --model "/hf/hub/models--${EDU_MODEL//\//--}/snapshots/${EDU_MODEL_REVISION}" \
        --per-source "${BENCH_PER_SOURCE:-2000}" --out "/work/out/bench_${TAG}.json"
