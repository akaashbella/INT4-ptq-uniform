#!/bin/bash
#SBATCH -J ptq-tin-eval-quant-cls
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --array=0-17%4
#SBATCH --mail-type=ALL,TIMELIMIT_90
#SBATCH --mail-user=bellaak@oregonstate.edu
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

# Weight-only PTQ eval (w8/w6/w4) for classification — needs eval_fp32.json in the run dir.
# Submit from the repository root.

set -euo pipefail

# GCC runtime for CompressAI / PyTorch C++ extensions — edit module name for your cluster.
module purge
module load gcc/11.5

# --- edit these ---
# Must be provided explicitly (edit here or export DATA_ROOT before sbatch).
DATA_ROOT="${DATA_ROOT:-}"
OUTPUT_ROOT="/nfs/hpc/share/bellaak/research/INT4-ptq-uniform/results"
NUM_WORKERS=8
# ------------------

MODELS=(resnet50 mobilenetv3_large convnext_tiny)
REGIMES=(clean noisy_uniform_a0.02)
SEEDS=(0 1 2)

idx=${SLURM_ARRAY_TASK_ID}
seed=${SEEDS[$(( idx % 3 ))]}
tmp=$(( idx / 3 ))
regime=${REGIMES[$(( tmp % 2 ))]}
model=${MODELS[$(( tmp / 2 ))]}

CONFIG="configs/classification/${model}_${regime}.yaml"

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config file not found: ${PWD}/${CONFIG}" >&2
  exit 1
fi
if [[ ! -f venv/bin/activate ]]; then
  echo "ERROR: venv not found: ${PWD}/venv/bin/activate" >&2
  exit 1
fi
if [[ -z "${DATA_ROOT}" || "${DATA_ROOT}" == *"/path/to/"* || ! -d "${DATA_ROOT}" ]]; then
  echo "ERROR: DATA_ROOT is not a directory: ${DATA_ROOT}" >&2
  exit 1
fi
OUT_PARENT=$(dirname -- "$OUTPUT_ROOT")
if [[ ! -d "$OUT_PARENT" ]]; then
  echo "ERROR: OUTPUT_ROOT parent directory does not exist: ${OUT_PARENT}" >&2
  exit 1
fi

mkdir -p logs
echo "DEBUG: gcc=$(command -v gcc 2>/dev/null || echo missing) $(gcc --version 2>&1 | head -n1)"
source venv/bin/activate

echo "SLURM_SUBMIT_DIR=${PWD}"
echo "model=${model} regime=${regime} seed=${seed}"
echo "CONFIG=${CONFIG}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "REMINDER: FP32 baseline required — run eval_fp32 for this config/seed/checkpoint (best) first; eval_quant reads eval_fp32.json in the run directory."
echo "Starting EVAL QUANT classification: model=${model} regime=${regime} seed=${seed}"

python scripts/eval_quant.py \
  --config "${CONFIG}" \
  --seed "${seed}" \
  --checkpoint best \
  --data-root "${DATA_ROOT}" \
  --device cuda \
  --num-workers "${NUM_WORKERS}" \
  --output-root "${OUTPUT_ROOT}"
