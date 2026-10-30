#!/usr/bin/env bash
#SBATCH --job-name=edu-pilot-report
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
# FASE 3.2 (CPU): relatório por fonte dos rótulos do piloto (10_annotate com ANNOTATE_INPUT=pilot)
# contra a amostra da FASE 3.1. Saída em reports/pilot/.
#
#   sbatch slurm/21_pilot_report.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE EDU_PILOT_ROOT EDU_MODEL_TAG

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/pilot"
LABELS_DIR="${PILOT_LABELS:-${EDU_PILOT_ROOT}/labels/${EDU_MODEL_TAG}}"
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

# Volumes do corpus: arquivos do cpt-corpus no NFS (o Docker não monta o NFS) e o resumo da amostra.
VOLUME_ARGS=()
SUMMARY="${PILOT_SAMPLE_SUMMARY:-$(ls -t "${REPORT_DIR}"/sample_summary_*.json 2>/dev/null | head -1)}"
if [[ -n "${CPT_PROJECT_DIR:-}" && -n "${SUMMARY}" ]]; then
    mkdir -p "${SCRATCH}/cpt"
    checks=()
    for file in ${CPT_INGEST_CHECKS}; do
        rsync -a "${CPT_PROJECT_DIR}/${file}" "${SCRATCH}/cpt/" && checks+=("/work/cpt/$(basename "${file}")")
    done
    rsync -a "${CPT_PROJECT_DIR}/catalog/sources.yaml" "${CPT_PROJECT_DIR}/${CPT_BEFORE_AFTER}" "${SUMMARY}" "${SCRATCH}/cpt/"
    VOLUME_ARGS=(--catalog /work/cpt/sources.yaml --checks "${checks[@]}"
                 --before-after "/work/cpt/$(basename "${CPT_BEFORE_AFTER}")"
                 --sample-summary "/work/cpt/$(basename "${SUMMARY}")")
    log "Volumes do corpus: ${checks[*]} | ${CPT_BEFORE_AFTER} | $(basename "${SUMMARY}")"
fi

run_container pilot-report --cpus="${SLURM_CPUS_PER_TASK:-2}" -e HOME=/work \
    -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" \
    -v "${EDU_PILOT_ROOT}/sample:/sample:ro" -v "${LABELS_DIR}:/labels:ro" -w /opt/educability-annotator "${EDU_IMAGE}" \
    python tools/pilot_report.py --sample /sample --labels /labels \
        --out-json "/work/out/pilot_${TAG}.json" --out-md "/work/out/pilot_${TAG}.md" ${VOLUME_ARGS[@]+"${VOLUME_ARGS[@]}"}
