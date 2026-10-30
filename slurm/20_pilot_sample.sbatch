#!/usr/bin/env bash
#SBATCH --job-name=edu-pilot-sample
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@300
#
# FASE 3.1 (CPU): amostra uniforme por fonte dos documentos mantidos pelo dedup, com o texto, em
# ${EDU_PILOT_ROOT}/sample/source=<id>/. Lê o cpt-corpus só para leitura. Resumo em reports/pilot/.
#
#   sbatch slurm/20_pilot_sample.sbatch
#   sbatch --export=ALL,PILOT_PER_SOURCE=5000 slurm/20_pilot_sample.sbatch
#   sbatch --cpus-per-task=8 --mem=64G --export=ALL,PILOT_SPLIT_ONLY=1 slurm/20_pilot_sample.sbatch   # só reparte
#   ... --export=ALL,PILOT_SPLIT_ONLY=1,PILOT_SUBSAMPLE=10000 ...   # reparte e reduz a N por fonte (uniforme)
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE CPT_DATA_ROOT CPT_VERSION EDU_PILOT_ROOT

SCRATCH="$(job_scratch)"
TAG="${SLURM_JOB_ID:-local}"
REPORT_DIR="${PROJECT_DIR}/reports/pilot"
SAMPLE_DIR="${EDU_PILOT_ROOT}/sample"
mkdir -p "${PROJECT_DIR}/logs" "${REPORT_DIR}"
if [[ "${PILOT_SPLIT_ONLY:-0}" == 1 ]]; then
    [[ -d "${SAMPLE_DIR}" ]] || { log "[ERRO] ${SAMPLE_DIR} não existe"; exit 1; }
elif [[ -e "${SAMPLE_DIR}" ]]; then
    log "[ERRO] ${SAMPLE_DIR} já existe: a amostra é fixa; apague-a à mão para sortear de novo"
    exit 1
fi
mkdir -p "${SAMPLE_DIR}"

stage_out() {
    local _rc=$?
    set +e
    [[ -f "${SAMPLE_DIR}/sample_summary.json" ]] && rsync -a "${SAMPLE_DIR}/sample_summary.json" \
        "${REPORT_DIR}/sample_summary_${TAG}.json" && log "Stage-out: ${REPORT_DIR}"
    rm -rf "${SCRATCH}" || log "[WARN] não consegui remover ${SCRATCH}"
    exit "${_rc}"
}
trap stage_out EXIT
trap on_term TERM INT

cleanup_orphan_scratches "${EDU_IMAGE}" "${PROJECT_NAME}_*"
lazy_build_edu runtime
stage_in_code "${SCRATCH}/code" "${PROJECT_DIR}/src" "${PROJECT_DIR}/tools"

MOUNTS=(-e HOME=/work -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro"
        -v "${CPT_DATA_ROOT}:/data:ro" -v "${SAMPLE_DIR}:/out" -w /opt/educability-annotator)
if [[ "${PILOT_SPLIT_ONLY:-0}" == 1 ]]; then
    log "Repartindo ${SAMPLE_DIR} em partes de ${PILOT_ROWS_PER_PART:-1000}"
    run_container pilot-split --cpus="${SLURM_CPUS_PER_TASK:-2}" "${MOUNTS[@]}" "${EDU_IMAGE}" \
        python tools/pilot_sample.py --split-existing --rows-per-part "${PILOT_ROWS_PER_PART:-1000}" \
            ${PILOT_SUBSAMPLE:+--subsample "${PILOT_SUBSAMPLE}"} --out /out
    exit 0
fi
log "Amostra: ${PILOT_PER_SOURCE:-20000} por fonte | ${SLURM_CPUS_PER_TASK} CPUs | saída ${SAMPLE_DIR}"
run_container pilot-sample --cpus="${SLURM_CPUS_PER_TASK:-2}" "${MOUNTS[@]}" "${EDU_IMAGE}" \
    python tools/pilot_sample.py --data-root /data --version "${CPT_VERSION}" \
        --per-source "${PILOT_PER_SOURCE:-20000}" --seed "${PILOT_SEED:-2026}" \
        --threads "${SLURM_CPUS_PER_TASK:-2}" --rows-per-part "${PILOT_ROWS_PER_PART:-1000}" --out /out
log "Amostra pronta: $(du -sh "${SAMPLE_DIR}" | cut -f1)"
