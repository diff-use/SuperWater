#!/usr/bin/env bash
# Score-inference sweep on combined_val_list across 3 checkpoints × 5 ratios.
# Runs sequentially on CUDA device 3; --resume makes it safe to restart.
#
# Usage:
#   bash scripts/run_val_sweep.sh [2>&1 | tee ~/sw_output_analysis/sweep.log]

set -euo pipefail

PYTHON=/home/dev/workspace/local_flow/.venv/bin/python
SCRIPT=/home/dev/workspace/SuperWater/scripts/score_inference.py
SW_SRC=/home/dev/workspace/SuperWater/src

DATA_DIR=/home/dev/sw_cache/data/full_pdb_dataset
ESM_DIR=/home/dev/sw_cache/data/full_pdb_embeddings
CACHE_PATH=/home/dev/sw_cache/score_inf_cache_val
SPLIT=/home/dev/workspace/SuperWater/splits/combined_val_list.txt
OUT_ROOT=~/sw_output_analysis

GPU=0  # within CUDA_VISIBLE_DEVICES=3

CHECKPOINTS=(
    "conf_dataset_clustered  models/conf_dataset_clustered   best_model.pt"
    "large_pdbs_clustered    models/large_pdbs_clustered     best_model.pt"
    "water_score_res15       models/water_score_res15        best_model.pt"
)
RATIOS=(1 2 5 10 15)

total=0
done_count=0

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    read -r ckpt_name score_dir ckpt_file <<< "$ckpt_entry"
    for ratio in "${RATIOS[@]}"; do
        total=$((total + 1))
    done
done

echo "===== Sweep: ${total} runs (${#CHECKPOINTS[@]} checkpoints × ${#RATIOS[@]} ratios) ====="
echo "Split:     $SPLIT  ($(wc -l < "$SPLIT") complexes)"
echo "Out root:  $OUT_ROOT"
echo "Started:   $(date)"
echo ""

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$SW_SRC

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    read -r ckpt_name score_dir ckpt_file <<< "$ckpt_entry"

    # Resolve score_dir to absolute path (relative to SuperWater root)
    abs_score_dir=/home/dev/workspace/SuperWater/$score_dir

    for ratio in "${RATIOS[@]}"; do
        done_count=$((done_count + 1))
        out_dir="$OUT_ROOT/$ckpt_name/ratio_${ratio}"
        echo "------------------------------------------------------------"
        echo "[$done_count/$total]  checkpoint=$ckpt_name  ratio=$ratio"
        echo "  score_dir: $abs_score_dir"
        echo "  ckpt:      $ckpt_file"
        echo "  out:       $out_dir"
        echo "  $(date)"
        echo ""

        $PYTHON "$SCRIPT" \
            --score_dir  "$abs_score_dir" \
            --ckpt       "$ckpt_file" \
            --data_dir   "$DATA_DIR" \
            --split      "$SPLIT" \
            --esm        "$ESM_DIR" \
            --cache_path "$CACHE_PATH" \
            --out        "$out_dir" \
            --water_ratio "$ratio" \
            --gpu        "$GPU" \
            --resume

        echo ""
        echo "[$done_count/$total] DONE  checkpoint=$ckpt_name  ratio=$ratio  $(date)"
        echo ""
    done
done

echo "===== ALL RUNS COMPLETE  $(date) ====="
