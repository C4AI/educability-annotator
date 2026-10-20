#!/usr/bin/env bash
#SBATCH --job-name=educability
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=educability-%j.log

set -euo pipefail
: "${ANNOTATOR_ROOT:?Defina o caminho absoluto do projeto}"
: "${ANNOTATOR_OUTPUT:?Defina o caminho absoluto da saída já preparada}"
source "$ANNOTATOR_ROOT/.venv/bin/activate"
export ANNOTATOR_MASTER_ADDR
ANNOTATOR_MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export ANNOTATOR_MASTER_PORT="${ANNOTATOR_MASTER_PORT:-29500}"
export ANNOTATOR_GPUS_PER_NODE="${ANNOTATOR_GPUS_PER_NODE:-8}"
export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false

srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 bash -c '
  set -euo pipefail
  recovery=()
  if [[ "${RECOVER_ANNOTATOR_LOCK:-0}" == "1" ]]; then
    recovery=(--recover-lock)
  fi
  exec torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$ANNOTATOR_GPUS_PER_NODE" \
    --node-rank="$SLURM_PROCID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$ANNOTATOR_MASTER_ADDR:$ANNOTATOR_MASTER_PORT" \
    --rdzv-id="$SLURM_JOB_ID" \
    --local-addr="$(hostname -f)" \
    --max-restarts=0 \
    -m educability_annotator run \
    --output "$ANNOTATOR_OUTPUT" \
    --device cuda --dtype auto --offline \
    --batch-size 64 --max-batch-tokens 32768 \
    --torch-threads 2 --arrow-threads 2 \
    --resume "${recovery[@]}"
'
