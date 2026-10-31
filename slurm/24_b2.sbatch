#!/usr/bin/env bash
#SBATCH --job-name=edu-b2
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --signal=B:TERM@60
#
# Amostra B2 da Deduplicação (CPU). B2_MODE=prepare monta as entradas do motor em ${EDU_LABELS_ROOT}/b2/
# (removidos e ~500 mantidos de controle); B2_MODE=compare confere os rótulos do controle contra a rodada
# completa. Relatórios em reports/b2/.
#
#   sbatch --export=ALL,B2_MODE=prepare slurm/24_b2.sbatch
#   sbatch --export=ALL,B2_MODE=compare slurm/24_b2.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE CPT_DATA_ROOT CPT_VERSION EDU_LABELS_ROOT EDU_MODEL_TAG

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/b2"
B2_DIR="${B2_DIR:-${CPT_DATA_ROOT}/fase3/b2}"
WORK_DIR="${EDU_LABELS_ROOT}/b2"
mkdir -p "${PROJECT_DIR}/logs" "${REPORT_DIR}" "${WORK_DIR}"

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
MOUNTS=(-e HOME=/work -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro"
        -v "${B2_DIR}:/b2:ro" -v "${CPT_DATA_ROOT}/release/${CPT_VERSION}/kept:/kept:ro"
        -v "${EDU_LABELS_ROOT}/${EDU_MODEL_TAG}:/full:ro" -v "${WORK_DIR}:/out" -w /opt/educability-annotator)
case "${B2_MODE:-prepare}" in
    prepare)
        run_container b2-prepare --cpus="${SLURM_CPUS_PER_TASK:-2}" "${MOUNTS[@]}" "${EDU_IMAGE}" \
            python tools/b2_inputs.py prepare --b2 /b2 --kept /kept --out /out --control "${B2_CONTROL:-500}" ;;
    compare)
        run_container b2-compare --cpus="${SLURM_CPUS_PER_TASK:-2}" "${MOUNTS[@]}" "${EDU_IMAGE}" \
            python tools/b2_inputs.py compare --control-labels /out/control_labels --full-labels /full \
                --out "/work/out/control_${TAG}.json" ;;
    *) log "[ERRO] B2_MODE=${B2_MODE}"; exit 1 ;;
esac
