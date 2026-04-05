#!/bin/bash
#SBATCH -J ptq-tin-aggregate
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --mail-type=ALL,TIMELIMIT_90
#SBATCH --mail-user=bellaak@oregonstate.edu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Rebuild master_results.csv, classification_results.csv, compression_results.csv
# from eval_*.json under the results tree. CPU-only; no GPU.
# Submit from the repository root.

set -euo pipefail

# GCC runtime for CompressAI / PyTorch C++ extensions — edit module name for your cluster.
module load gcc

# --- edit this ---
RESULTS_ROOT="results"
# -----------------

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f venv/bin/activate ]]; then
  echo "ERROR: venv not found: ${PWD}/venv/bin/activate" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "ERROR: RESULTS_ROOT is not a directory: ${PWD}/${RESULTS_ROOT}" >&2
  exit 1
fi

mkdir -p logs
echo "DEBUG: gcc=$(command -v gcc 2>/dev/null || echo missing) $(gcc --version 2>&1 | head -n1)"
source venv/bin/activate

echo "SLURM_SUBMIT_DIR=${PWD}"
echo "RESULTS_ROOT=${RESULTS_ROOT}"
echo "Aggregating results under ${RESULTS_ROOT}"

python scripts/export_master_csv.py --results-root "${RESULTS_ROOT}"
