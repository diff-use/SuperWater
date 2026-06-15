"""Score-model inference: run reverse diffusion to generate raw water-candidate positions
and evaluate them against crystallographic waters.

The script is intentionally score-model-only (no confidence model).  Every sampled
particle is recorded; downstream analysis (PR curves, clustering, confidence scoring)
can be done separately on the saved outputs.

Outputs written to ``--out``:
  positions/<pdb>.npy       float32 (n_particles, 3) in original (un-centered) frame
  distances/<pdb>.npz       d_pred2true (M,), d_true2pred (K,)
  summary.csv               per-complex stats
  raw_distances.pkl         nested dict {pdb: {n_res, n_true, n_particles,
                                               d_pred2true, d_true2pred}}
                            (compatible with benchmark_score_pr.py downstream tools)
  run_args.json             CLI arguments for reproducibility

Use ``--resume`` to skip complexes that already have a positions file (safe to re-run
after an interruption on a large dataset).

Examples
--------
Smoke test – single PDB, ratio 1, epoch-25 checkpoint:

    .venv/bin/python scripts/score_inference.py \\
        --score_dir models/large_pdbs_clustered --ckpt model_epoch25.pt \\
        --data_dir /home/dev/sw_cache/data/full_pdb_dataset \\
        --split /home/dev/sw_cache/data/large_dataset_splits/test.txt \\
        --esm /home/dev/sw_cache/data/full_pdb_embeddings \\
        --cache_path /home/dev/sw_cache/score_inf_cache \\
        --out outputs/score_inference/epoch25_r1 \\
        --water_ratio 1 --gpu 2 --limit_complexes 1

Full test set, best checkpoint, ratio 10:

    .venv/bin/python scripts/score_inference.py \\
        --score_dir models/large_pdbs_clustered --ckpt best_model.pt \\
        --data_dir /home/dev/sw_cache/data/full_pdb_dataset \\
        --split /home/dev/sw_cache/data/large_dataset_splits/test.txt \\
        --esm /home/dev/sw_cache/data/full_pdb_embeddings \\
        --cache_path /home/dev/sw_cache/score_inf_cache \\
        --out outputs/score_inference/best_r10 \\
        --water_ratio 10 --gpu 2 --resume
"""
import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from functools import partial

import numpy as np
import torch

# PyG graph-cache objects are trusted local artifacts; suppress the torch >= 2.6
# weights_only warning without patching the whole process.
_ORIG_TORCH_LOAD = torch.load
def _trusting_load(*a, **k):
    k.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*a, **k)
torch.load = _trusting_load

from superwater.utils.utils import get_model, resolve_model_dir
from superwater.utils.parsing import parse_inference_args
from superwater.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule
from superwater.utils.sampling import sampling, randomize_position_multiple
from superwater.utils.nearest_point_dist import get_nearest_point_distances
from superwater.utils.find_water_pos import find_real_water_pos
from superwater.structure_io import structure_path
from superwater.confidence.dataset import get_args
from superwater.inference import construct_loader_origin, set_seed


PR_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]  # Angstroms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score-model-only water inference + distance evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- model ---
    p.add_argument("--score_dir", required=True,
                   help="Directory containing model_parameters.yml and the checkpoint file.")
    p.add_argument("--ckpt", default="best_model.pt",
                   help="Checkpoint filename inside --score_dir (e.g. best_model.pt, "
                        "model_epoch25.pt, best_ema_model.pt).")

    # --- data ---
    p.add_argument("--data_dir", required=True,
                   help="Root data directory (contains per-complex sub-folders).")
    p.add_argument("--split", required=True,
                   help="Path to split .txt file (one PDB id per line).")
    p.add_argument("--esm", required=True,
                   help="Directory with pre-computed ESM-2 embeddings.")
    p.add_argument("--cache_path", default="data/score_inf_cache",
                   help="Directory for the PyG graph cache (built once, reused on reruns).")

    # --- inference ---
    p.add_argument("--water_ratio", type=int, default=10,
                   help="Number of sampled water particles = n_residues * water_ratio.")
    p.add_argument("--inference_steps", type=int, default=20,
                   help="Number of reverse-diffusion denoising steps.")
    p.add_argument("--sampling_batch_size", type=int, default=32,
                   help="Batch size for the inner per-timestep forward pass in sampling().")

    # --- output ---
    p.add_argument("--out", required=True,
                   help="Output directory.  Sub-directories positions/ and distances/ "
                        "are created automatically.")
    p.add_argument("--resume", action="store_true",
                   help="Skip complexes whose positions/<pdb>.npy already exists.")
    p.add_argument("--no_distances", action="store_true",
                   help="Skip distance computation against true waters (faster; no summary stats).")

    # --- misc ---
    p.add_argument("--gpu", type=int, default=0,
                   help="CUDA device index.")
    p.add_argument("--limit_complexes", type=int, default=0,
                   help="Process only the first N complexes (0 = all).")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers for graph preprocessing.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_loader(args, score_args, t_to_sigma):
    """Construct the PyG DataLoader via the standard construct_loader_origin path."""
    inf = parse_inference_args([])
    inf.data_dir = args.data_dir
    inf.split_test = args.split
    inf.esm_embeddings_path = args.esm
    inf.batch_size_preprocessing = 1
    inf.num_workers = args.num_workers
    # Override cache_path and limit so graph cache lands where we want it.
    score_args.cache_path = args.cache_path
    score_args.limit_complexes = args.limit_complexes
    return construct_loader_origin(inf, score_args, t_to_sigma)


def load_score_model(score_dir, ckpt_name, device, score_args, t_to_sigma):
    ckpt_path = os.path.join(score_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    model = get_model(score_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ckpt_path}  ({n_params:,} parameters)")
    return model


def sample_one(model, batch, n_water, score_args, t_to_sigma, tr_schedule,
               inference_steps, sampling_batch_size, device):
    """Run score-only reverse diffusion for one complex.

    Returns predicted particle coordinates in the original (un-centered) frame,
    shape (n_water, 3).
    """
    data_list = [copy.deepcopy(batch)]
    randomize_position_multiple(data_list, False, score_args.tr_sigma_max, water_num=n_water)
    preds, _ = sampling(
        data_list=data_list,
        model=model,
        inference_steps=inference_steps,
        tr_schedule=tr_schedule,
        device=device,
        t_to_sigma=t_to_sigma,
        model_args=score_args,
        batch_size=sampling_batch_size,
    )
    center = batch.original_center.cpu().numpy()
    parts = np.concatenate([g["ligand"].pos.cpu().numpy() for g in preds], axis=0)
    return (parts + center).astype(np.float32)


def compute_pr_row(d_pred2true, d_true2pred, thresholds=PR_THRESHOLDS):
    """Return a dict of precision/recall values at each threshold."""
    row = {}
    for d in thresholds:
        row[f"prec_{d}A"] = float((d_pred2true <= d).mean())
        row[f"rec_{d}A"] = float((d_true2pred <= d).mean())
    return row


def write_summary_csv(rows, path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.5f}" if isinstance(r[c], float) else r[c]) for c in cols})


def print_aggregate(results, thresholds=PR_THRESHOLDS):
    """Print micro-averaged precision/recall over all processed complexes."""
    if not results:
        return
    print("\n--- Aggregate (micro-averaged) ---")
    header = f"{'threshold':>10}" + "".join(f"  {'prec':>8}  {'rec':>8}" for _ in thresholds)
    print(header)
    sep = "-" * len(header)
    print(sep)
    vals = []
    for d in thresholds:
        tp_p = sum(int((r["d_pred2true"] <= d).sum()) for r in results.values())
        n_p  = sum(int(r["n_particles"])              for r in results.values())
        tp_t = sum(int((r["d_true2pred"] <= d).sum()) for r in results.values())
        n_t  = sum(int(r["n_true"])                   for r in results.values())
        prec = tp_p / n_p if n_p else float("nan")
        rec  = tp_t / n_t if n_t else float("nan")
        vals.append((d, prec, rec))
    row_str = ""
    for d, prec, rec in vals:
        row_str += f"  {prec:8.4f}  {rec:8.4f}"
    print(f"{'':>10}" + row_str)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    pos_dir  = os.path.join(args.out, "positions")
    dist_dir = os.path.join(args.out, "distances")
    os.makedirs(pos_dir, exist_ok=True)
    if not args.no_distances:
        os.makedirs(dist_dir, exist_ok=True)

    # Save CLI args for reproducibility.
    with open(os.path.join(args.out, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    set_seed(args.seed)

    score_dir = resolve_model_dir(args.score_dir)
    score_args = get_args(score_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=score_args)

    print("Loading score model...")
    model = load_score_model(score_dir, args.ckpt, device, score_args, t_to_sigma)

    print("Building data loader...")
    loader = build_loader(args, score_args, t_to_sigma)
    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

    n_total = len(loader.dataset)
    print(f"Complexes to process: {n_total}  |  water_ratio: {args.water_ratio}  "
          f"|  inference_steps: {args.inference_steps}")

    results  = {}  # pdb -> dict with distance arrays
    summary_rows = []
    skipped = []
    failed  = []
    t_start = time.time()

    for i, batch in enumerate(loader):
        pdb = batch[0]["name"]
        pos_path  = os.path.join(pos_dir,  f"{pdb}.npy")
        dist_path = os.path.join(dist_dir, f"{pdb}.npz")

        if args.resume and os.path.exists(pos_path):
            skipped.append(pdb)
            print(f"[{i+1}/{n_total}] {pdb}: skipped (already done)")
            continue

        n_res   = int(batch[0]["receptor"].pos.shape[0])
        n_water = int(n_res * args.water_ratio)

        t0 = time.time()
        try:
            parts = sample_one(model, batch, n_water, score_args, t_to_sigma,
                               tr_schedule, args.inference_steps,
                               args.sampling_batch_size, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"[{i+1}/{n_total}] {pdb}: OOM — skipped")
                failed.append(pdb)
                continue
            raise
        elapsed = time.time() - t0

        # Save positions.
        np.save(pos_path, parts)

        row = {"pdb": pdb, "n_res": n_res, "n_particles": len(parts),
               "time_s": round(elapsed, 2)}

        if not args.no_distances:
            water_file = structure_path(os.path.join(args.data_dir, pdb), pdb, "_water")
            try:
                true_pos = find_real_water_pos(water_file)
                n_true   = len(true_pos)
                d_p2t, _ = get_nearest_point_distances(parts,    true_pos)  # (M,)
                d_t2p, _ = get_nearest_point_distances(true_pos, parts)     # (K,)
                np.savez(dist_path,
                         d_pred2true=d_p2t.astype(np.float32),
                         d_true2pred=d_t2p.astype(np.float32))
                results[pdb] = dict(n_res=n_res, n_true=n_true, n_particles=len(parts),
                                    d_pred2true=d_p2t.astype(np.float32),
                                    d_true2pred=d_t2p.astype(np.float32))
                row["n_true"] = n_true
                row["mean_d_pred2true"] = float(d_p2t.mean())
                row["min_d_pred2true"]  = float(d_p2t.min())
                row.update(compute_pr_row(d_p2t, d_t2p))
                dist_info = (f"n_true={n_true}  "
                             f"prec@1A={row['prec_1.0A']:.3f}  "
                             f"rec@1A={row['rec_1.0A']:.3f}")
            except Exception as e:
                dist_info = f"(distance eval failed: {e})"
        else:
            dist_info = ""

        mem = (f"  peak_GPU={torch.cuda.max_memory_allocated(device)/1e9:.2f}GB"
               if device.type == "cuda" else "")
        print(f"[{i+1}/{n_total}] {pdb}: n_res={n_res}  n_particles={len(parts)}  "
              f"{dist_info}  {elapsed:.1f}s{mem}")
        torch.cuda.reset_peak_memory_stats(device)

        summary_rows.append(row)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --- write aggregate outputs ---
    summary_path = os.path.join(args.out, "summary.csv")
    write_summary_csv(summary_rows, summary_path)
    print(f"\nWrote {summary_path}")

    if results:
        pkl_path = os.path.join(args.out, "raw_distances.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Wrote {pkl_path}")
        print_aggregate(results)

    n_done = len(summary_rows)
    print(f"Done: {n_done} processed, {len(skipped)} skipped, {len(failed)} failed  "
          f"— total {time.time() - t_start:.0f}s")
    if failed:
        print(f"Failed (OOM): {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
