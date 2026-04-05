# Slurm jobs (Tiny ImageNet, locked grid)

Plain bash scripts aligned with this repo’s CLI (`scripts/train_*.py`, `scripts/eval_*.py`, `scripts/export_master_csv.py`). Style matches the examples under `example/`.

## Before you submit

1. **Working directory**: submit from the **repository root** (the folder that contains `configs/`, `scripts/`, and `venv/`).
2. **Virtualenv**: `python -m venv venv && source venv/bin/activate` and install deps (see root `README.md`).
3. **Logs directory**: `mkdir -p logs` (Slurm needs `logs/` for `%x_%A_%a` / `%x_%j` paths).
4. **Edit each script’s variables** at the top (paths, `--mail-user`, optional `NUM_WORKERS`).
5. **chmod**: `chmod +x scripts/slurm/*.sh` if your site requires it.

Each script checks **before Python runs**: YAML config file exists (train/eval), `venv/bin/activate` exists, dataset and path sanity (`DATA_ROOT`, `OUTPUT_ROOT` parent, or `RESULTS_ROOT` for aggregate).

## Scripts

| Script | Partition | Array | Jobs | Purpose |
|--------|-----------|-------|------|---------|
| `train_classification_array_dgxh.sh` | dgxh | `0-17%3` | 18 | Train classification (3 models × 2 regimes × 3 seeds). |
| `train_compression_array_dgxh.sh` | dgxh | `0-17%3` | 18 | Train compression (3 models × 2 regimes × 3 seeds). |
| `eval_fp32_classification_array_dgxh.sh` | dgxh | `0-17%4` | 18 | FP32 eval, classification configs. |
| `eval_fp32_compression_array_dgxh.sh` | dgxh | `0-17%4` | 18 | FP32 eval, compression configs. |
| `eval_quant_classification_array_dgxh.sh` | dgxh | `0-17%4` | 18 | PTQ eval (w8/w6/w4); needs `eval_fp32.json` in the run dir. |
| `eval_quant_compression_array_dgxh.sh` | dgxh | `0-17%4` | 18 | Same for compression. |
| `aggregate_results_share.sh` | share | *(none)* | 1 | Rebuild CSVs from `eval_*.json` via `export_master_csv.py`. |

**`OUTPUT_ROOT`** (train/eval scripts): the **parent results directory** that will contain `classification/` and `compression/` (same meaning as `--output-root` in Python). Use an absolute path on the cluster if results live outside the repo.

**Quant eval (`eval_quant_*`)**: run **`eval_fp32_*` first** for the same model/regime/seed and `--checkpoint best` so `eval_fp32.json` exists beside the checkpoint; `eval_quant.py` depends on that baseline.

## Array indexing (18 tasks)

All array scripts use the same mapping for `SLURM_ARRAY_TASK_ID` (`idx`):

- `seed = SEEDS[idx % 3]` → cycles **0, 1, 2** as `idx` increases.
- `tmp = idx / 3` (integer division).
- `regime = REGIMES[tmp % 2]` → **clean**, then **noisy_uniform_a0.02**.
- `model = MODELS[tmp / 2]` → three models in array order.

**Traversal order** (example, classification): for each model, both regimes are listed with seeds `0,1,2` before moving to the next model:

- `idx` 0–2: `resnet50`, `clean`, seeds 0–2  
- `idx` 3–5: `resnet50`, `noisy_uniform_a0.02`, seeds 0–2  
- `idx` 6–8: `mobilenetv3_large`, `clean`, seeds 0–2  
- … through `idx` 15–17: `convnext_tiny`, `noisy_uniform_a0.02`, seeds 0–2  

Config path: `configs/<task>/${model}_${regime}.yaml` with `task` = `classification` or `compression`.

## Variables to edit

- **`DATA_ROOT`**: absolute path to Tiny ImageNet root (must exist as a directory).
- **`OUTPUT_ROOT`** (train/eval): parent of per-task trees `classification/...` and `compression/...`; the parent path must exist (the script can create `OUTPUT_ROOT` itself when training writes outputs).
- **`NUM_WORKERS`**: DataLoader workers (default `8`; tune for I/O).
- **`RESULTS_ROOT`** (`aggregate_results_share.sh` only): directory that **already exists** and contains the results tree (typically `results` under the repo, or your HPC path after runs).
- **`#SBATCH --mail-user`**: your address.

## Example

```bash
cd /path/to/INT4-ptq-uniform
mkdir -p logs
# edit DATA_ROOT, OUTPUT_ROOT, mail-user in the script
sbatch scripts/slurm/train_classification_array_dgxh.sh
```

See the root [README](../../README.md) for dataset layout and experiment design.
