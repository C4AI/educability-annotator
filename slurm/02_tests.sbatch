#!/usr/bin/env bash
#SBATCH --job-name=edu-tests
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:TERM@60
#
# Testes avulsos na imagem runtime (iteração do desenvolvimento): stage-in do código e pytest com os
# argumentos dados. O modelo publicado fica disponível em EDUCABILITY_TEST_MODEL (cache do NVMe).
#
#   sbatch slurm/02_tests.sbatch tests/test_sidecar.py
#   sbatch --export=ALL,EDUCABILITY_TEST_HUB=1 slurm/02_tests.sbatch -m hub tests/test_sidecar.py
#   sbatch --gres=gpu:1 slurm/02_tests.sbatch -m cuda
set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${PROJECT_DIR}/config.env"
source "${PROJECT_DIR}/scripts/common.sh"
require_vars PROJECT_NAME PROJECT_DIR SCRATCH_BASE EDU_IMAGE EDU_HF_HOME EDU_MODEL EDU_MODEL_REVISION

SCRATCH="$(job_scratch)"
mkdir -p "${PROJECT_DIR}/logs"
trap 'rm -rf "${SCRATCH}"' EXIT
trap on_term TERM INT

stage_in_code "${SCRATCH}/code" "${PROJECT_DIR}/src" "${PROJECT_DIR}/tests" "${PROJECT_DIR}/tools" \
    "${PROJECT_DIR}/pyproject.toml"
GPUS="$(job_gpus)"
GPU_ARGS=()
[[ -n "${GPUS}" ]] && GPU_ARGS=(--gpus "\"device=${GPUS}\"")
log "pytest $* | GPUs: ${GPUS:-nenhuma}"
run_container tests --cpus="${SLURM_CPUS_PER_TASK:-2}" "${GPU_ARGS[@]}" --shm-size=8g \
    -e HOME=/work -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
    -e EDUCABILITY_TEST_HUB="${EDUCABILITY_TEST_HUB:-0}" \
    -e EDUCABILITY_TEST_MODEL="/hf/hub/models--${EDU_MODEL//\//--}/snapshots/${EDU_MODEL_REVISION}" \
    -v "${SCRATCH}:/work" -v "${SCRATCH}/code:/opt/educability-annotator:ro" -v "${EDU_HF_HOME}:/hf:ro" \
    -w /opt/educability-annotator "${EDU_IMAGE}" \
    python -m pytest -q -p no:cacheprovider "$@"
