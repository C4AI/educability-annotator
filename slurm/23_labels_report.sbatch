#!/usr/bin/env bash
#SBATCH --job-name=edu-labels-report
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=512G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@120
#
# FASES 5.1/5.2 (CPU): cobertura (todo doc mantido tem rótulo) e distribuição exata por fonte dos rótulos
# da rodada completa, sobre a partição kept do cpt-corpus. Saída em reports/labels/.
#
#   sbatch slurm/23_labels_report.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE CPT_DATA_ROOT CPT_VERSION EDU_LABELS_ROOT EDU_MODEL_TAG

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/labels"
INPUT_DIR="${LABELS_REPORT_INPUT:-${CPT_DATA_ROOT}/release/${CPT_VERSION}/kept}"
LABELS_DIR="${LABELS_REPORT_LABELS:-${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}}"
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

log "Entrada ${INPUT_DIR} | rótulos ${LABELS_DIR}"
run_container labels-report --cpus="${SLURM_CPUS_PER_TASK:-2}" -e HOME=/work \
    -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" \
    -v "${INPUT_DIR}:/input:ro" -v "${LABELS_DIR}:/labels:ro" -w /opt/educability-annotator "${EDU_IMAGE}" \
    python tools/labels_report.py --unified /input --labels /labels --threads "${SLURM_CPUS_PER_TASK:-2}" \
        --out-json "/work/out/labels_${TAG}.json" --out-md "/work/out/labels_${TAG}.md"
