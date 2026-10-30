#!/usr/bin/env bash
#SBATCH --job-name=edu-compare
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@60
#
# FASE 2.5 (CPU): paridade entre duas rodadas de rótulos sobre os mesmos documentos (ex.: piloto em CPU
# FP32 × piloto em GPU FP16). Saída em reports/pilot/compare_<job>.{json,md}.
#
#   sbatch --export=ALL,COMPARE_REFERENCE=<dir>,COMPARE_CANDIDATE=<dir> slurm/22_compare_labels.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE COMPARE_REFERENCE COMPARE_CANDIDATE

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/pilot"
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

log "Referência ${COMPARE_REFERENCE} × candidato ${COMPARE_CANDIDATE}"
run_container compare --cpus="${SLURM_CPUS_PER_TASK:-2}" -e HOME=/work \
    -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" \
    -v "${COMPARE_REFERENCE}:/reference:ro" -v "${COMPARE_CANDIDATE}:/candidate:ro" \
    -w /opt/educability-annotator "${EDU_IMAGE}" \
    python tools/compare_labels.py --reference /reference --candidate /candidate \
        --out-json "/work/out/compare_${TAG}.json" --out-md "/work/out/compare_${TAG}.md"
