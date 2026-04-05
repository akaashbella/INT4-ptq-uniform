# Weight noise and post-training quantization (Tiny ImageNet)

Research codebase for studying **training-time uniform weight noise** versus **clean** training, and **post-training weight-only quantization** (symmetric per-tensor, `fp32` / `w8` / `w6` / `w4`), on **Tiny ImageNet** only.

## Locked experiment design

| Item | Specification |
|------|-----------------|
| **Dataset** | Tiny ImageNet only (200 classes). |
| **Classification** | TorchVision: `resnet50`, `mobilenetv3_large`, `convnext_tiny`; input **224×224**, ImageNet normalization. |
| **Compression** | CompressAI: `factorized_prior`, `scale_hyperprior`, `cheng2020_attention`; input **64×64**, MSE rate–distortion with **λ = 0.0130**. |
| **Regimes** | `clean` (α = 0); `noisy_uniform_a0.02` (α = 0.02), uniform noise on eligible **Conv/Linear weights** only, temporary each batch. |
| **Training** | **100** epochs; seeds **0, 1, 2**; best checkpoint = max val top-1 (cls) / min val RD loss (cmp); `best.pt` + `last.pt`. |
| **PTQ** | Weight-only; symmetric per-tensor; zp = 0; never overwrite checkpoints. |

Configs under `configs/` encode these names and hyperparameters explicitly.

## Install

```bash
cd /path/to/INT4-ptq-uniform
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Dependencies: PyTorch, TorchVision, CompressAI, PyYAML, pandas, tqdm (see `pyproject.toml`).

**HPC:** Load CUDA-compatible PyTorch wheels matching the cluster’s driver/CUDA stack; do not assume a GPU at install time.

## Tiny ImageNet layout

Point `--data-root` at the dataset root (folder that contains `train/` and `val/`):

```
tiny-imagenet-200/
  train/
    <wnid>/
      images/
        *.JPEG
  val/
    images/
      *.JPEG
    val_annotations.txt
```

- **200** class folders under `train/` (WordNet ids).
- **val_annotations.txt**: each line `filename<TAB or SPACE><wnid>…`; only filename and wnid are used.
- Classification and compression use the **same** wnid → index mapping (lexicographic wnid order).

## One training job (CLI)

Classification (example):

```bash
export REPO_ROOT=/path/to/INT4-ptq-uniform
export DATA_ROOT=/path/to/tiny-imagenet-200
python scripts/train_classification.py \
  --config "${REPO_ROOT}/configs/classification/resnet50_clean.yaml" \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --output-root "${REPO_ROOT}/results" \
  --device cuda:0 \
  --num-workers 8
```

Compression (example):

```bash
python scripts/train_compression.py \
  --config "${REPO_ROOT}/configs/compression/factorized_prior_clean.yaml" \
  --seed 0 \
  --data-root "${DATA_ROOT}" \
  --output-root "${REPO_ROOT}/results" \
  --device cuda:0 \
  --num-workers 8
```

**Outputs per run:** `config.json`, `environment.json`, `train_log.csv`, `val_log.csv`, `best.pt`, `last.pt` under  
`results/<classification|compression>/<model>/<regime>/seed_<k>/`.

**Resume:** Not automated; each epoch overwrites `last.pt`. To resume from a checkpoint would require a dedicated resume path (not implemented).

## FP32 evaluation

After training, evaluate `best` (default) or `last`:

```bash
python scripts/eval_fp32.py \
  --config "${REPO_ROOT}/configs/classification/resnet50_clean.yaml" \
  --seed 0 \
  --checkpoint best \
  --data-root "${DATA_ROOT}" \
  --output-root "${REPO_ROOT}/results"
```

Writes `eval_fp32.json` in the same run directory.

## Quantized evaluation (w8 / w6 / w4)

Requires `eval_fp32.json` in that run directory.

```bash
python scripts/eval_quant.py \
  --config "${REPO_ROOT}/configs/classification/resnet50_clean.yaml" \
  --seed 0 \
  --checkpoint best \
  --data-root "${DATA_ROOT}" \
  --output-root "${REPO_ROOT}/results"
```

Writes `eval_quant.json`.

## Export aggregate CSVs

Rebuilds `master_results.csv`, `classification_results.csv`, `compression_results.csv`, and `summary.json` under the results root (deterministic ordering; overwrites):

```bash
python scripts/export_master_csv.py --results-root "${REPO_ROOT}/results"
```

## Command matrix (no execution)

Generate bash lines for the full locked grid (**3×2×3** classification, **3×2×3** compression, plus eval/export). This script **only prints or writes a file**; it does **not** run training.

```bash
# Print all classification training commands (placeholders for data/results paths)
python scripts/launch_commands.py train-classification --header

# Write a script; use --dry-run to skip writing the file
python scripts/launch_commands.py train-all --header --output run_all_train.sh
python scripts/launch_commands.py eval-all --checkpoint best --output run_all_eval.sh --dry-run
```

Placeholders default to `${DATA_ROOT}`, `${RESULTS_ROOT}`, `${PYTHON:-python}` (override with `--data-root-placeholder`, etc.).

## Reproducibility metadata (no training)

Print JSON to stdout; optionally save:

```bash
python scripts/report_environment.py --output "${REPO_ROOT}/metadata/preflight_env.json"
```

Training runs also write `environment.json` per run via the same metadata collector (git, Python, torch, torchvision, compressai, numpy, CUDA, hostname, timestamp).

## Slurm (examples)

Templates live in `scripts/slurm/`. Copy/edit placeholders:

- `YOUR_ACCOUNT`, `YOUR_PARTITION`, `HH:MM:SS`, `YOUR_MEM`
- Export `REPO_ROOT`, `DATA_ROOT`, `RESULTS_ROOT`, `CONFIG_PATH`, `SEED` (and `CHECKPOINT` for eval)

Example:

```bash
export REPO_ROOT=/path/to/INT4-ptq-uniform
export DATA_ROOT=/path/to/tiny-imagenet-200
export RESULTS_ROOT=${REPO_ROOT}/results
export CONFIG_PATH=${REPO_ROOT}/configs/classification/resnet50_clean.yaml
export SEED=0
sbatch --export=ALL,REPO_ROOT,DATA_ROOT,RESULTS_ROOT,CONFIG_PATH,SEED \
  scripts/slurm/train_classification.slurm
```

Use **one job per (config, seed)** for the locked grid; combine with job arrays or your scheduler’s array syntax if desired.

## Logging (HPC)

- Training/eval scripts configure **stdout logging** with timestamps (`weight_noise_ptq.common.logging_setup`).
- **tqdm** progress bars write to stderr; in non-interactive jobs, Slurm captures both `.out` and `.err`.
- Checkpoints are saved **atomically** (temp file + replace) in `checkpointing.py`.

## Optional sanity scripts (manual)

Run **on the cluster or locally** after install; not invoked by this repo automatically:

| Script | Purpose |
|--------|---------|
| `scripts/sanity_check_models.py` | Build all six models; one forward pass each. |
| `scripts/sanity_check_data.py` | Load Tiny ImageNet train/val loaders (`--data-root` required). |
| `scripts/sanity_check_noise_and_quant.py` | Tiny net: noise eligibility + PTQ w8/w6/w4 + `results/` writable. |

## Results tree

```
results/
  classification/<model>/<regime>/seed_<k>/{config.json, environment.json, train_log.csv, val_log.csv, best.pt, last.pt, eval_fp32.json, eval_quant.json}
  compression/...
  master_results.csv
  classification_results.csv
  compression_results.csv
  summary.json
```

## License / attribution

Follow your institution’s policies; cite TorchVision, CompressAI, and Tiny ImageNet sources in publications.
