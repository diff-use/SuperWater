#!/usr/bin/env bash
# TEMPORARY (do not commit): regenerate the conf-split structures with the new per-complex
# filters, embed only the missing ids, then 4-GPU DDP-train the score model at dataset scope.
# Delete after use.
set -euo pipefail
cd /home/dev/workspace/SuperWater

PY=.venv/bin/python
TORCHRUN=.venv/bin/torchrun
CACHE=/mnt/diffuse-shared/vratins/sw_cache
RAW=/mnt/diffuse-shared/migrated/diffuse-public/water_data/pdb_data
DATA=$CACHE/data/full_pdb_dataset_filtered
ESM=$CACHE/data/full_pdb_embeddings
SPLITS_OUT=$CACHE/data/conf_splits_filtered
CKPT=/mnt/diffuse-shared/vratins/sw_checkpoints

# 1) Regenerate conf structures in place with the new per-complex filters (overwrite the
#    existing conf ids, create the 153 conf-only ids). No graph-construction args (those are
#    training-only); embeddings handled in step 2. NO --skip_existing so filters re-apply.
$PY scripts/setup_custom_dataset.py \
  --raw_data_dir "$RAW" \
  --split_train splits/conf_dataset_train.txt \
  --split_val   splits/conf_dataset_valid.txt \
  --split_test  splits/conf_dataset_test.txt \
  --out_dir "$DATA" \
  --split_out_dir "$SPLITS_OUT" \
  --embeddings_dir "$ESM" \
  --out_format cif \
  --skip_embeddings \
  --num_workers "$(nproc)"

# 2) ESM embeddings for missing ids only (existing protein files are unchanged -> reused).
$PY -m superwater.embed --data_dir "$DATA" --out_dir "$ESM" --skip_existing

# 3) 4-GPU DDP score-model training, README 300-epoch params, dataset scope. The rank-0
#    barrier (train.py) builds the graph cache once; the other ranks just read it.
$TORCHRUN --nproc_per_node=4 -m superwater.train \
  --run_name conf_dataset_filtered \
  --data_dir "$DATA" \
  --esm_embeddings_path "$ESM" \
  --cache_path "$CACHE/data_filtered" --cache_scope dataset \
  --split_train "$SPLITS_OUT/train.txt" \
  --split_val   "$SPLITS_OUT/val.txt" \
  --split_test  "$SPLITS_OUT/test.txt" \
  --log_dir "$CKPT/" \
  --all_atoms --remove_hs --receptor_radius 15 --c_alpha_max_neighbors 24 \
  --ns 24 --nv 6 --num_conv_layers 3 \
  --distance_embed_dim 64 --cross_distance_embed_dim 64 --sigma_embed_dim 64 \
  --tr_sigma_min 0.1 --tr_sigma_max 30 --scale_by_sigma --dynamic_max_cross \
  --lr 1e-3 --batch_size 8 --n_epochs 300 \
  --scheduler plateau --scheduler_patience 30 --dropout 0.1 \
  --use_ema --cudnn_benchmark --test_sigma_intervals \
  --num_workers 10 --num_dataloader_workers 10 --pin_memory --checkpoint_freq 25 \
 --wandb --wandb_entity vratins --project superwater_final
