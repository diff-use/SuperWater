# SuperWater — Running Guide

End-to-end reference for **every** runnable workflow in this repo: environment setup,
data preparation, score-model training, score inference/evaluation, confidence-model
setup + training, and confidence (full-pipeline) inference. For a quick high-level
overview see the top-level [README](../README.md); this document is the detailed
operational reference.

All commands are run from the repository root with the project environment active
(`conda activate superwater`, or the `uv` venv — see [Setup](#1-setup)). A CUDA 11.8
NVIDIA GPU is required; CPU-only execution is not supported.

## Pipeline at a glance

```
raw structures ──► [organize / setup] ──► per-complex dataset ──► [ESM embeddings]
                                                                         │
                                                                         ▼
                                          ┌──────────── score (diffusion) training ───────────┐
                                          │                models/<score_run>/                │
                                          └────────────────────────┬───────────────────────--┘
                          ┌──────────────────────────────┬─────────┴───────────────────────────┐
                          ▼                               ▼                                     ▼
              score inference / eval         confidence training            full pipeline (score+confidence+cluster)
            scripts/score_inference.py      superwater-confidence-train      superwater-predict  (production)
            scripts/benchmark_score_pr.py   models/<conf_run>/               superwater-infer    (evaluation)
```

| Stage | Command | Section |
|-------|---------|---------|
| Install env | `bash scripts/install.sh` / `scripts/install_uv.sh` | [1](#1-setup) |
| Build dataset | `scripts/setup_custom_dataset.py`, `superwater-organize` | [2](#2-data-preparation) |
| ESM embeddings | `superwater-embed` | [2.3](#23-generate-esm-2-embeddings) |
| Train score model | `superwater-train` / `python -m superwater.train` | [3](#3-score-diffusion-model-training) |
| Score inference / eval | `scripts/score_inference.py`, `scripts/benchmark_score_pr.py` | [4](#4-score-model-inference--evaluation) |
| Train confidence model | `superwater-confidence-train` | [5](#5-confidence-model-setup--training) |
| Full-pipeline inference | `superwater-infer` (eval) / `superwater-predict` (production) | [6](#6-confidence--full-pipeline-inference) |

---

## 1. Setup

Requires an NVIDIA GPU with CUDA 11.8 and a working `conda`/`mamba` (or `uv`).

### 1.1 Conda (recommended)

```bash
bash scripts/install.sh
conda activate superwater
```

This creates the `superwater` conda env (PyTorch 2.5.1 + CUDA 11.8, e3nn 0.5.4, rdkit),
installs the PyTorch Geometric CUDA extension wheels from the PyG wheel index
(`torch-2.5.0+cu118`), and runs `pip install -e .` so the `superwater-*` console scripts
are registered. Equivalent manual steps:

```bash
conda env create -f environment.yml
conda activate superwater
pip install -r requirements-pyg-cu118.txt
pip install -e .
```

### 1.2 uv (recommended, reproducible)

```bash
uv sync --extra cu126               # or: bash scripts/install_uv.sh
uv run superwater-predict --config examples/configs/predict_5srf.yaml
```

`uv sync --extra cu126` resolves the full GPU stack from `uv.lock` (PyTorch 2.8 + CUDA
12.6 and the matching PyG extension wheels, pinned in `pyproject.toml`'s `[tool.uv]`),
including `torch-geometric==2.6.1`, and installs the package into `./.venv`. Prefix
subsequent commands with `uv run` (or activate `.venv`). Add `--extra dev` for pytest.
For a different CUDA build, edit the two `[[tool.uv.index]]` URLs in `pyproject.toml` and
re-run `uv lock && uv sync --extra cu126`, or use the conda installer above.

### 1.3 Verify the GPU

```bash
python scripts/check_gpu.py
```

Prints the visible CUDA devices and their memory. If it reports "CUDA is not available",
nothing downstream will run.

### 1.4 Console scripts

`pip install -e .` registers these entry points (defined in `pyproject.toml`); each has an
equivalent `python -m superwater.<module>` form:

| Console script | Module | Purpose |
|----------------|--------|---------|
| `superwater-predict` | `superwater.predict` | One-command production prediction from a YAML config |
| `superwater-embed` | `superwater.embed` | Generate ESM-2 embeddings for a dataset |
| `superwater-organize` | `superwater.organize_dataset` | Organize raw PDBs into the per-complex layout |
| `superwater-train` | `superwater.train:main_function` | Train the score (diffusion) model |
| `superwater-confidence-train` | `superwater.confidence.train` | Train the confidence model |
| `superwater-infer` | `superwater.inference` | Full-pipeline inference/evaluation over a split |

---

## 2. Data preparation

Training, score inference and confidence training all read a **per-complex dataset
directory** plus a directory of **ESM-2 embeddings**. The dataset layout is one folder per
complex:

```
data/<dataset>/<PDB_ID>/
├── <PDB_ID>_protein_processed.{cif,pdb}   # protein only
└── <PDB_ID>_water.{cif,pdb}               # crystallographic waters (oxygens)
```

The data layer reads CIF in preference to PDB and falls back to the other format if one is
missing or fails to parse. CIF is preferred because legacy fixed-column PDB cannot
represent 5-character ligand CCD codes (e.g. `A1ADA`) without corrupting the file.

The paper's splits are in `examples/data/splits/` (`train_res15.txt`, `val_res15.txt`,
`test_res15.txt`) — each a plain list of PDB IDs. Additional/working splits used for the
larger experiments live in `splits/`.

### 2.1 Download the published dataset

Download `waterbind` (17,092 complexes) from
[Zenodo](https://doi.org/10.5281/zenodo.17229778) and unpack it under `data/<dataset>/`.
Skip to [2.3](#23-generate-esm-2-embeddings) (embeddings) if it already ships the
per-complex layout above.

### 2.2 Build a dataset from raw files

**From raw PDB-REDO `*_final.cif`/`*_final.pdb` (full workflow, recommended).**
`scripts/setup_custom_dataset.py` splits each source into a protein-only
`_protein_processed` file and a `_water` file, generates ESM embeddings, and writes
normalized (lowercased, `_final`-stripped) split files to `--split_out_dir` — use those
for training.

```bash
python scripts/setup_custom_dataset.py \
    --raw_data_dir <raw_dir> \
    --split_train <train.txt> --split_val <val.txt> --split_test <test.txt> \
    --out_dir data/<dataset> --split_out_dir data/<dataset>_splits \
    --embeddings_dir data/<dataset>_embeddings \
    --skip_existing --download_missing
```

Useful flags:

- `--skip_existing` — incremental re-run: complexes that already have both output files are
  left untouched, and the embedding stage skips complexes that already have a `_chain_0.pt`.
- `--download_missing` — split ids absent from `--raw_data_dir` are fetched from PDB-REDO
  into a temp dir, processed, then deleted.
- `--out_format {cif,pdb}` — output structure format (default `cif`).
- `--skip_embeddings` — skip the ESM stage (run `superwater-embed` later instead).
- `--build_cache` / `--cache_path` / `--cache_scope {dataset,split}` — also prebuild the
  PyG graph cache (`--cache_scope dataset` shares per-complex graphs across split files).
- Featurization flags (`--all_atoms`/`--no_all_atoms`, `--remove_hs`/`--keep_hs`,
  `--receptor_radius`, `--c_alpha_max_neighbors`, …) must match the model you'll train.

Run with `--help` for the complete list.

**From a folder of plain `.pdb` files (lightweight).** `superwater-organize` truncates
each filename to a 4-character PDB id, drops duplicates, and writes the per-complex layout
plus split files:

```bash
superwater-organize --raw_data <folder_name> --data_root data \
    --output_dir <dataset> --splits_path data/splits
```

### 2.3 Generate ESM-2 embeddings

Skip if you already generated embeddings via `setup_custom_dataset.py`. Embeddings are
generated **in-process** (no cloned ESM repo needed); the first run downloads the ESM-2
model (~2.5 GB) to `~/.cache/torch`.

```bash
superwater-embed --data_dir data/<dataset> --out_dir data/<dataset>_embeddings
```

- `--skip_existing` — embed only complexes that lack a `_chain_0.pt` in `--out_dir` (a
  fully-cached re-run does no model load, since ESM is loaded lazily).
- `--device {cuda,cpu}` — defaults to `cuda`.

Output: one `<name>_chain_<i>.pt` per chain. Point training/inference `--esm_embeddings_path`
at this directory.

### 2.4 Auditing a dataset

`setup_custom_dataset.py` writes CIF by default, which round-trips 5-character ligand CCD
codes (e.g. `A1ADA`) that legacy PDB silently corrupts — so a CIF build needs no separate
repair step. To health-check an existing dataset, use the read-only auditor:

```bash
python scripts/audit_dataset.py \
    --data_dir data/<dataset> --embeddings_dir data/<dataset>_embeddings
```

It writes `<data_dir>/logs/audit_report.tsv` (override with `--report` for read-only dirs)
and exits non-zero if any complex is missing/unparseable, has zero waters, or lacks an
embedding. To repair flagged complexes, delete their processed files and re-run prep — the
auditor's `--delete_broken` does the deletion in place, then `setup_custom_dataset.py
--skip_existing` re-derives them as CIF and re-embeds (dataset-scope caching rebuilds only
the now-missing graphs).

---

## 3. Score (diffusion) model training

Trains the equivariant score model that reverse-diffuses random water particles onto
hydration sites. Entry point: `superwater-train` (≡ `python -m superwater.train`).

```bash
python -m superwater.train \
    --run_name water_score_res15_retrain \
    --data_dir data/<dataset> \
    --esm_embeddings_path data/<dataset>_embeddings \
    --split_train examples/data/splits/train_res15.txt \
    --split_val   examples/data/splits/val_res15.txt \
    --split_test  examples/data/splits/test_res15.txt \
    --log_dir models \
    --all_atoms --remove_hs --receptor_radius 15 --c_alpha_max_neighbors 24 \
    --ns 24 --nv 6 --num_conv_layers 3 \
    --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 \
    --tr_sigma_min 0.1 --tr_sigma_max 30 --scale_by_sigma --dynamic_max_cross \
    --lr 1e-3 --batch_size 8 --n_epochs 300 \
    --scheduler plateau --scheduler_patience 30 --dropout 0.1 \
    --use_ema --cudnn_benchmark --test_sigma_intervals \
    --num_workers 10 --num_dataloader_workers 10
```

The command above reproduces the shipped `models/water_score_res15` checkpoint. Add
`--wandb --wandb_entity <user>` to log to Weights & Biases.

**Outputs** (`models/<run_name>/`):

- `best_model.pt`, `best_ema_model.pt`, `last_model.pt` — checkpoints (EMA only with
  `--use_ema`). `last_model.pt` also holds optimizer/EMA state for `--restart_dir`.
- `model_parameters.yml` — full arg snapshot; downstream stages read graph/arch params from
  here, so the dataset only needs to be specified once.
- `losses_iter.csv` (per batch) and `losses_epoch.csv` (per-epoch train/val) — plot with
  `python plot_training.py`.

**Key flags** (see `superwater-train --help` / `src/superwater/utils/parsing.py` for all):

| Flag | Meaning |
|------|---------|
| `--cache_path` | PyG graph cache dir (default `data/cache`); built on first run, reused after. |
| `--cache_scope {split,dataset}` | `dataset` reuses per-complex graphs across split files. |
| `--receptor_radius`, `--c_alpha_max_neighbors`, `--atom_radius`, `--atom_max_neighbors`, `--all_atoms`, `--remove_hs` | Graph construction — must be reused consistently downstream. |
| `--ns`, `--nv`, `--num_conv_layers`, `--*_embed_dim`, `--dropout` | Network architecture. |
| `--tr_sigma_min`, `--tr_sigma_max`, `--scale_by_sigma` | Diffusion noise schedule. |
| `--n_epochs`, `--batch_size`, `--lr`, `--scheduler`, `--scheduler_patience`, `--use_ema` | Optimization. |
| `--restart_dir <dir>` | Resume from a previous run's `last_model.pt`. |
| `--config <file.yml>` | Load any of the above from a YAML file. |

Complexes that fail graph preprocessing are recorded in `failed_complexes.txt` inside the
cache directory; the only other trace is a missing `<name>.pt`.

---

## 4. Score-model inference / evaluation

These workflows run the score model **alone** (no confidence model, no clustering): every
sampled water particle is recorded so downstream analysis (precision/recall, clustering,
confidence scoring) can be done on the saved outputs. Use them to evaluate a score
checkpoint against crystallographic waters.

### 4.1 `scripts/score_inference.py` — sample + evaluate one split

Reverse-diffuses `n_residues × water_ratio` particles per complex, saves raw positions, and
(unless `--no_distances`) computes nearest-neighbour distances to true waters with
precision/recall at 0.5/1.0/1.5/2.0 Å.

```bash
python scripts/score_inference.py \
    --score_dir models/water_score_res15 --ckpt best_model.pt \
    --data_dir data/<dataset> \
    --split   data/<dataset>_splits/test.txt \
    --esm     data/<dataset>_embeddings \
    --cache_path data/score_inf_cache \
    --out outputs/score_inference/best_r10 \
    --water_ratio 10 --inference_steps 20 --gpu 0 --resume
```

- `--ckpt` selects the checkpoint file inside `--score_dir` (`best_model.pt`,
  `best_ema_model.pt`, `model_epoch25.pt`, …).
- `--resume` skips complexes that already have a `positions/<pdb>.npy` (safe to restart).
- `--limit_complexes N` processes only the first N (use `1` for a smoke test).
- `--no_distances` skips the true-water comparison (faster; no summary stats).
- `--sampling_batch_size` is the inner per-timestep forward-pass batch (default 32); lower
  it on OOM. OOM on a complex is caught, the complex is skipped, and the run exits non-zero.

**Outputs** (`--out`): `positions/<pdb>.npy` (un-centered particle coords),
`distances/<pdb>.npz` (`d_pred2true`, `d_true2pred`), `summary.csv` (per-complex stats),
`raw_distances.pkl` (nested dict for downstream tooling), `run_args.json`. A
micro-averaged precision/recall table is printed at the end.

### 4.2 `scripts/benchmark_score_pr.py` — precision/recall sweep across ratios

Same "raw cloud" method as above but sweeps multiple water-to-residue ratios and caches the
raw distance arrays, so any distance cutoff can be applied at plot time.

```bash
python scripts/benchmark_score_pr.py \
    --data_dir data/bench62 --split data/bench62/bench62_ids.txt \
    --esm data/bench62_embeddings --score_dir models/water_score_res15 \
    --out outputs/score_pr
```

`scripts/prepare_bench62.py` builds the 62-PDB benchmark dataset used here from a raw mount
(adjust the hard-coded paths inside before running).

### 4.3 `scripts/run_val_sweep.sh` — batch sweep over checkpoints × ratios

Convenience wrapper that runs `score_inference.py` sequentially over several checkpoints and
ratios on a validation split (`--resume` makes it restartable). Edit the paths/device at the
top of the script before use.

---

## 5. Confidence model setup + training

The confidence model is a classifier that scores each sampled water particle by how far it
is likely to be from a true hydration site. **Setup is implicit in training**: the first
training run uses the trained score model (step 3) to sample water positions for every
complex, computes each particle's mean-absolute-deviation (MAD) from the nearest true
water, and caches `(positions, MAD)` pairs; the classifier then trains on that cache.

Entry point: `superwater-confidence-train` (≡ `python -m superwater.confidence.train`). The
dataset/architecture flags must match the score model.

```bash
python -m superwater.confidence.train \
    --original_model_dir models/water_score_res15 \
    --run_name water_confidence_res15_retrain \
    --data_dir data/<dataset> \
    --esm_embeddings_path data/<dataset>_embeddings \
    --split_train examples/data/splits/train_res15.txt \
    --split_val   examples/data/splits/val_res15.txt \
    --split_test  examples/data/splits/test_res15.txt \
    --log_dir models \
    --all_atoms --remove_hs \
    --ns 24 --nv 6 --num_conv_layers 3 --scale_by_sigma --dynamic_max_cross --dropout 0.1 \
    --inference_steps 20 --water_ratio 15 \
    --lr 1e-3 --batch_size 8 --n_epochs 50 \
    --running_mode train --mad_prediction \
    --cache_creation_id 1 --cache_ids_to_combine 1
```

**The cache-building (setup) stage.** The first run is slow: it samples and caches water
positions for every train + val complex (under `--cache_path`, default
`data/cache_confidence`); later runs reuse that cache. The relevant flags:

| Flag | Meaning |
|------|---------|
| `--original_model_dir` | Trained score model whose checkpoint (`--ckpt`, default `best_model.pt`) samples the positions. Its `model_parameters.yml` also supplies the graph/arch params. |
| `--water_ratio` | Particles sampled per residue when building the cache. Lower (e.g. 10) on GPU-memory limits. |
| `--inference_steps` | Reverse-diffusion steps used while sampling the cache. |
| `--resample_steps` | Independent resampling passes per complex (total ratio = `water_ratio × resample_steps`). |
| `--cache_creation_id` | Tags this sampling pass; run multiple ids to accumulate more samples. |
| `--cache_ids_to_combine` | Which `cache_creation_id`s to concatenate into the training set (e.g. `1 2 3`). |
| `--cache_path` / `--cache_scope` | Where the sampled-position cache lives. |

**The classifier training stage.** Trained on the cache:

| Flag | Meaning |
|------|---------|
| `--mad_prediction` | Regress the (sigmoid-normalized) MAD with MSE loss — the shipped `water_confidence_res15_sigmoid` head. Omit for binary classification with `--mad_classification_cutoff`. |
| `--mad_classification_cutoff` | MAD threshold (Å) defining a positive when not using `--mad_prediction`; a list enables multi-bin cross-entropy. |
| `--balance` | Keep natural positive/negative ratio instead of balancing. |
| `--n_epochs`, `--batch_size`, `--lr`, `--scheduler`, `--scheduler_patience` | Optimization. |
| `--main_metric` / `--main_metric_goal` | Early-stopping metric (`confidence_loss`/`accuracy`/`ROC AUC`; default `min` confidence_loss). |
| `--transfer_weights` | Initialize from the score model's weights (uses `--original_model_dir` arch). |
| `--restart_dir` | Resume from a previous confidence run's `last_model.pt`. |

**Outputs** (`models/<run_name>/`): `best_model.pt`, `last_model.pt`,
`model_parameters.yml`, and (with `--model_save_frequency`/`--best_model_save_frequency`)
periodic `model_epoch<N>.pt` / `best_model_epoch<N>.pt`. Validation prints loss and, for the
classifier head, accuracy and ROC-AUC each epoch.

---

## 6. Confidence / full-pipeline inference

The full pipeline = score sampling → confidence scoring → clustering → final waters. Two
front-ends share the same core (`superwater.inference.run_inference`):

- **`superwater-predict`** — production, config-driven, structure in → waters out. Covered
  in the [README Quick start](../README.md#quick-start). Use this for real predictions.
- **`superwater-infer`** — evaluation over a dataset split, for benchmarking trained score +
  confidence models. Covered below.

### 6.1 `superwater-infer` — full-pipeline evaluation over a split

```bash
superwater-infer \
    --original_model_dir models/water_score_res15 \
    --confidence_dir     models/water_confidence_res15_sigmoid \
    --data_dir data/<dataset> \
    --esm_embeddings_path data/<dataset>_embeddings \
    --split_test examples/data/splits/test_res15.txt \
    --cache_path data/cache_infer \
    --water_ratio 10 --inference_steps 20 --cap 0.1 \
    --save_pos --output_format pdb
```

How it resolves configuration: the **score** model is built from
`--original_model_dir/model_parameters.yml` (and sampled with `--ckpt`, default
`best_model.pt`); the **confidence** model is built and loaded from
`--confidence_dir/model_parameters.yml` + `best_model.pt`; graph/featurization params come
from the score model's saved args. So you generally only set the data, ratio and threshold
flags here — not the architecture flags.

Inference-specific flags:

| Flag | Meaning |
|------|---------|
| `--water_ratio` | Particles sampled per residue (higher = more coverage + memory). |
| `--inference_steps` | Reverse-diffusion denoising steps. |
| `--resample_steps` | Independent resampling passes (total ratio = `water_ratio × resample_steps`). |
| `--cap` | Confidence keep-probability threshold (≈ 0.02–0.5; higher = stricter). |
| `--use_sigmoid` | Apply a sigmoid to the confidence output. **Leave off** for the shipped `water_confidence_res15_sigmoid` model, whose keep-probability is `1 − clamped MAD` (matching `superwater-predict`). |
| `--save_pos` | Write outputs to disk (required to keep results). |
| `--save_pos_path` | Output sub-folder name under `outputs/` (default `inferenced_pos_rr<ratio>_cap<cap>`). |
| `--output_format {pdb,cif}` | Structure format for the centroid file. |
| `--config <file.yml>` | Load any of the above from YAML. |

**Outputs** (`outputs/<save_pos_path>/<pdb>/`):

- `<pdb>_centroid.txt` — final clustered water coordinates (xyz).
- `<pdb>_centroid.{pdb,cif}` — final waters as a structure file.
- `<pdb>_filtered.txt` — every sampled position + its confidence probability.

A per-complex timing log is written to `outputs/logs/inference_log_rr<total_ratio>.txt`.

### 6.2 Production prediction (reference)

```bash
superwater-predict --config examples/configs/predict_5srf.yaml
```

Point one structure folder in, get clustered waters per structure out. The config maps
`prediction.water_ratio`/`inference_steps`/`confidence_cutoff` onto the same inference core
(`running_mode='test'`, `mad_prediction=True`). See the
[README](../README.md#quick-start) for the full config reference and the web app
(`python apps/webapp/app.py`).

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA is not available` | No supported GPU/driver; nothing runs. Check `python scripts/check_gpu.py`. |
| CUDA out of memory | Lower `--water_ratio`, `--batch_size`, and/or `--sampling_batch_size`. OOM on a single complex is caught and the complex is skipped. |
| First run very slow | Expected: the PyG graph cache (and, for confidence, the sampled-position cache) is built once and reused. Use `--resume`/`--skip_existing` to restart safely. |
| Corrupt protein/embedding files | Find them with `scripts/audit_dataset.py` (use `--delete_broken`), then re-run `setup_custom_dataset.py --skip_existing` to re-derive them as CIF (dataset-scope caching builds only the missing graphs). |
| Failed complexes during training | See `failed_complexes.txt` in the cache directory. |
| ESM download every run | First run only; the ~2.5 GB ESM-2 model is cached under `~/.cache/torch`. |
</content>
</invoke>
