#!/usr/bin/env bash
#SBATCH --job-name=edu-lock-deps
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Manda TERM ao shell do batch 120 s antes do limite, para o trap matar o container.
#SBATCH --signal=B:TERM@120
#
# FASE 1.1 (CPU): resolve docker/requirements.gpu.in em docker/requirements.gpu.lock, com todas as
# transitivas e hashes, no alvo toolchain da imagem e com o Python 3.12 dela. Só precisa rodar de novo
# quando o .in mudar. O lock fica no git, e o build da imagem runtime instala só dele.
#
# Submeter da raiz do projeto:  sbatch slurm/00_lock_deps.sbatch
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_TOOLCHAIN_IMAGE

SCRATCH="$(job_scratch)"
mkdir -p "${PROJECT_DIR}/logs"

stage_out() {
    local _rc=$?
    set +e
    if [[ ${_rc} -eq 0 && -s "${SCRATCH}/requirements.gpu.lock" ]]; then
        rsync -a "${SCRATCH}/requirements.gpu.lock" "${PROJECT_DIR}/docker/" && log "Stage-out: docker/requirements.gpu.lock"
    fi
    rm -rf "${SCRATCH}" || log "[WARN] não consegui remover ${SCRATCH}"
    exit "${_rc}"
}
trap stage_out EXIT
trap on_term TERM INT

log "Job ${SLURM_JOB_ID:-local} em $(hostname), ${SLURM_CPUS_PER_TASK:-?} CPUs"
cleanup_orphan_scratches "${EDU_TOOLCHAIN_IMAGE}" "${PROJECT_NAME}_*"
lazy_build_edu toolchain
stage_in_code "${SCRATCH}" "${PROJECT_DIR}/docker/requirements.gpu.in"

run_container lock-deps \
    --cpus="${SLURM_CPUS_PER_TASK:-2}" \
    -e HOME=/work -e UV_CACHE_DIR=/work/uv-cache -v "${SCRATCH}:/work" -w /work \
    "${EDU_TOOLCHAIN_IMAGE}" \
    uv pip compile requirements.gpu.in --python /usr/bin/python3.12 --generate-hashes \
        --output-file requirements.gpu.lock
log "Lock: $(command grep -cE '^[A-Za-z0-9_.-]+( @ |==)' "${SCRATCH}/requirements.gpu.lock") pacotes"
