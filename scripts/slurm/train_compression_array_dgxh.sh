#!/bin/bash
#SBATCH -J ptq-tin-cmp-train
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-17%3
#SBATCH --mail-type=ALL,TIMELIMIT_90
#SBATCH --mail-user=bellaak@oregonstate.edu
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Tiny ImageNet compression training — one job per (model, regime, seed).
# Submit from the repository root (directory containing configs/ and scripts/).

set -euo pipefail

# GCC runtime for CompressAI / PyTorch C++ extensions — edit module name for your cluster.
module purge
module load gcc/11.5

# --- edit these ---
DATA_ROOT="/nfs/hpc/share/bellaak/research/INT4-ptq-uniform/tiny-imagenet-200"
OUTPUT_ROOT="/nfs/hpc/share/bellaak/research/INT4-ptq-uniform/results"
NUM_WORKERS=8
# ------------------

MODELS=(factorized_prior scale_hyperprior cheng2020_attention)
REGIMES=(clean noisy_uniform_a0.02)
SEEDS=(0 1 2)

idx=${SLURM_ARRAY_TASK_ID}
seed=${SEEDS[$(( idx % 3 ))]}
tmp=$(( idx / 3 ))
regime=${REGIMES[$(( tmp % 2 ))]}
model=${MODELS[$(( tmp / 2 ))]}

CONFIG="configs/compression/${model}_${regime}.yaml"

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: ${PWD}/${CONFIG}" >&2
  exit 1
fi
if [[ ! -f venv/bin/activate ]]; then
  echo "ERROR: venv not found: ${PWD}/venv/bin/activate" >&2
  exit 1
fi
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: DATA_ROOT is not a directory: ${DATA_ROOT}" >&2
  exit 1
fi
OUT_PARENT=$(dirname -- "$OUTPUT_ROOT")
if [[ ! -d "$OUT_PARENT" ]]; then
  echo "ERROR: OUTPUT_ROOT parent directory does not exist: ${OUT_PARENT}" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

mkdir -p logs
echo "DEBUG: gcc=$(command -v gcc 2>/dev/null || echo missing) $(gcc --version 2>&1 | head -n1)"
source venv/bin/activate

echo "SLURM_SUBMIT_DIR=${PWD}"
echo "model=${model} regime=${regime} seed=${seed}"
echo "CONFIG=${CONFIG}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "Starting TRAIN compression: model=${model} regime=${regime} seed=${seed}"

python scripts/train_compression.py \
  --config "${CONFIG}" \
  --seed "${seed}" \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --num-workers "${NUM_WORKERS}" \
  --output-root "${OUTPUT_ROOT}"
