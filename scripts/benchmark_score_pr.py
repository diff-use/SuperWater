"""Benchmark the SuperWater *score model only* (no confidence model) across water-to-residue
ratios, for precision/recall against crystallographic waters.

Method: "raw cloud" -- the score model samples ``n_residues * ratio`` water particles per
complex and reverse-diffuses them; every particle counts (no clustering, no confidence).
Matching to true waters is by nearest-neighbour distance, in both directions, so any cutoff
``d`` can be applied at plot time:

    precision(d) = mean( particle_to_nearest_true_water <= d )   # over all M particles
    recall(d)    = mean( true_water_to_nearest_particle <= d )   # over all K true waters

For each (ratio, pdb) we cache the two raw distance arrays plus n_res / n_true / n_particles.

Run:
    .venv/bin/python scripts/benchmark_score_pr.py \
        --data_dir data/bench62 --split data/bench62/bench62_ids.txt \
        --esm data/bench62_embeddings --score_dir models/water_score_res15 \
        --out ~/workspace/sw_results/score_pr
"""
import argparse
import copy
import os
import pickle
import time
from functools import partial

import numpy as np
import torch

# The PyG graph cache (HeteroData objects) is loaded via a bare torch.load in
# datasets/pdbbind.py. Torch >= 2.6 defaults weights_only=True, which refuses those
# objects. The cache is a trusted, locally-built artifact, so default weights_only=False
# for this (read-only) benchmark run. Kept contained to this script (no src edits).
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

DEFAULT_RATIOS = [1, 2, 3, 5, 10, 15]
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default="data/bench62")
    p.add_argument("--split", default="data/bench62/bench62_ids.txt")
    p.add_argument("--esm", default="data/bench62_embeddings")
    p.add_argument("--score_dir", default="models/water_score_res15")
    p.add_argument("--cache_path", default="data/bench62_cache")
    p.add_argument("--out", default=os.path.expanduser("~/workspace/sw_results/score_pr"))
    p.add_argument("--ratios", type=int, nargs="+", default=DEFAULT_RATIOS)
    p.add_argument("--inference_steps", type=int, default=20)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--limit_complexes", type=int, default=0, help="cap #complexes (smoke test)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_loader(args, score_args, t_to_sigma):
    inf = parse_inference_args([])
    inf.data_dir = args.data_dir
    inf.split_test = args.split
    inf.esm_embeddings_path = args.esm
    inf.batch_size_preprocessing = 1
    inf.num_workers = 0  # main-process loading (cache already built; keeps torch.load patch in effect)
    # The graph cache & dataset-construction params come from the score model's saved args.
    score_args.cache_path = args.cache_path
    score_args.limit_complexes = args.limit_complexes
    return construct_loader_origin(inf, score_args, t_to_sigma)


def sample_cloud(model, batch, ratio, n_res, score_args, t_to_sigma, tr_schedule, steps, device):
    """Run score-only reverse diffusion for one complex at a given water ratio.

    Returns predicted particle coordinates in the original (un-centered) frame, shape (M, 3).
    """
    n_water = int(n_res * ratio)
    data_list = [copy.deepcopy(batch)]
    randomize_position_multiple(data_list, False, score_args.tr_sigma_max, water_num=n_water)
    preds, _ = sampling(data_list=data_list, model=model, inference_steps=steps,
                        tr_schedule=tr_schedule, device=device, t_to_sigma=t_to_sigma,
                        model_args=score_args)  # no confidence_model -> score-only
    center = batch.original_center.cpu().numpy()
    parts = np.concatenate([g["ligand"].pos.cpu().numpy() for g in preds], axis=0)
    return parts + center


def summarize(results, thresholds=THRESHOLDS):
    """Build per-ratio precision/recall rows (micro + macro) at each threshold."""
    rows = []
    for ratio in sorted(results):
        per = results[ratio]
        if not per:
            continue
        for d in thresholds:
            # micro = pool every particle / every true water across all PDBs
            tp_p = sum(int((r["d_pred2true"] <= d).sum()) for r in per.values())
            n_p = sum(int(r["n_particles"]) for r in per.values())
            tp_t = sum(int((r["d_true2pred"] <= d).sum()) for r in per.values())
            n_t = sum(int(r["n_true"]) for r in per.values())
            micro_prec = tp_p / n_p if n_p else float("nan")
            micro_rec = tp_t / n_t if n_t else float("nan")
            # macro = mean of per-PDB precision/recall
            macro_prec = float(np.mean([(r["d_pred2true"] <= d).mean() for r in per.values()]))
            macro_rec = float(np.mean([(r["d_true2pred"] <= d).mean() for r in per.values()]))
            rows.append(dict(ratio=ratio, threshold=d,
                             micro_precision=micro_prec, micro_recall=micro_rec,
                             macro_precision=macro_prec, macro_recall=macro_rec,
                             mean_particles=n_p / len(per), mean_true=n_t / len(per),
                             n_pdb=len(per)))
    return rows


def write_summary_csv(rows, path):
    import csv
    cols = ["ratio", "threshold", "micro_precision", "micro_recall",
            "macro_precision", "macro_recall", "mean_particles", "mean_true", "n_pdb"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.5f}" if isinstance(r[c], float) else r[c]) for c in cols})


def write_readme(path, ratios):
    txt = f"""# Score-model precision/recall benchmark (raw cloud)

Benchmarks SuperWater's **score model only** (`water_score_res15`, no confidence model) on
62 test PDBs, across water-to-residue ratios {ratios}. For each complex the score model
samples `n_residues * ratio` particles and reverse-diffuses them; every particle counts
(no clustering). Matching to crystallographic waters is by nearest-neighbour distance.

## Files
- `raw_distances.pkl` -- primary artifact. `pickle.load` gives a nested dict:
  `{{ratio: {{pdb_id: {{n_res, n_true, n_particles,
                        d_pred2true: float32[M], d_true2pred: float32[K]}}}}}}`
  where `d_pred2true[i]` = distance from particle i to its nearest true water, and
  `d_true2pred[j]` = distance from true water j to its nearest particle.
- `summary.csv` -- precision/recall per (ratio, threshold), micro & macro aggregated.
- `pr_curve.png` -- sanity-check PR plot at d = 1.0 A (one point per ratio).

## Precision / recall at cutoff d
    precision(d) = mean(d_pred2true <= d)   # fraction of particles near a true water
    recall(d)    = mean(d_true2pred <= d)   # fraction of true waters covered by a particle

## Plot a PR curve at a chosen cutoff
```python
import pickle, numpy as np, matplotlib.pyplot as plt
data = pickle.load(open("raw_distances.pkl", "rb"))
d = 1.0  # angstroms
xs, ys = [], []
for ratio in sorted(data):
    per = data[ratio]
    # micro-averaged over all PDBs
    prec = np.concatenate([r["d_pred2true"] for r in per.values()]) <= d
    rec  = np.concatenate([r["d_true2pred"] for r in per.values()]) <= d
    xs.append(rec.mean()); ys.append(prec.mean())
    plt.annotate(f"r={{ratio}}", (rec.mean(), prec.mean()))
plt.plot(xs, ys, "o-"); plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"score-model PR @ {{d}} Angstrom"); plt.savefig("my_pr.png")
```
"""
    with open(path, "w") as f:
        f.write(txt)


def write_pr_plot(results, path, d=1.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipping pr_curve.png: matplotlib unavailable: {e})")
        return
    xs, ys, labels = [], [], []
    for ratio in sorted(results):
        per = results[ratio]
        if not per:
            continue
        prec = np.concatenate([r["d_pred2true"] for r in per.values()]) <= d
        rec = np.concatenate([r["d_true2pred"] for r in per.values()]) <= d
        xs.append(float(rec.mean())); ys.append(float(prec.mean())); labels.append(ratio)
    plt.figure(figsize=(6, 5))
    plt.plot(xs, ys, "o-", color="#2c7fb8")
    for x, y, r in zip(xs, ys, labels):
        plt.annotate(f"r={r}", (x, y), textcoords="offset points", xytext=(6, 4))
    plt.xlabel("recall"); plt.ylabel("precision")
    plt.title(f"Score-model raw-cloud PR @ {d} A (micro, {len(labels)} ratios)")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=140)
    print(f"wrote {path}")


def main():
    args = parse_cli()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    score_dir = resolve_model_dir(args.score_dir)
    score_args = get_args(score_dir)
    t_to_sigma = partial(t_to_sigma_compl, args=score_args)

    print("Loading score model...")
    model = get_model(score_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state = torch.load(f"{score_dir}/best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    loader = build_loader(args, score_args, t_to_sigma)
    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)

    results = {r: {} for r in args.ratios}
    skipped = []
    t0 = time.time()
    n_done = 0
    for batch in loader:
        pdb = batch[0]["name"]
        n_res = int(batch[0]["receptor"].pos.shape[0])
        true = find_real_water_pos(structure_path(os.path.join(args.data_dir, pdb), pdb, "_water"))
        K = int(true.shape[0])
        for ratio in args.ratios:
            try:
                parts = sample_cloud(model, batch, ratio, n_res, score_args,
                                     t_to_sigma, tr_schedule, args.inference_steps, device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  OOM: {pdb} ratio {ratio} -> skipped")
                    skipped.append((pdb, ratio))
                    continue
                raise
            d_p2t, _ = get_nearest_point_distances(parts, true)            # (M,)
            d_t2p, _ = get_nearest_point_distances(true, parts)            # (K,)
            results[ratio][pdb] = dict(
                n_res=n_res, n_true=K, n_particles=int(parts.shape[0]),
                d_pred2true=d_p2t.astype(np.float32), d_true2pred=d_t2p.astype(np.float32))
        n_done += 1
        torch.cuda.empty_cache()
        print(f"[{n_done}] {pdb}: n_res={n_res} K={K} "
              f"M={[results[r].get(pdb, {}).get('n_particles') for r in args.ratios]} "
              f"({time.time() - t0:.0f}s elapsed)")

    # --- write outputs ---
    pkl_path = os.path.join(args.out, "raw_distances.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nwrote {pkl_path} ({n_done} pdbs x {len(args.ratios)} ratios)")

    rows = summarize(results)
    write_summary_csv(rows, os.path.join(args.out, "summary.csv"))
    write_readme(os.path.join(args.out, "README.md"), args.ratios)
    write_pr_plot(results, os.path.join(args.out, "pr_curve.png"), d=1.0)
    if skipped:
        print(f"skipped (OOM): {skipped}")

    # console summary at d=1.0
    print("\nratio  micro_prec  micro_rec   (d=1.0 A)")
    for r in rows:
        if abs(r["threshold"] - 1.0) < 1e-9:
            print(f"{r['ratio']:>5}  {r['micro_precision']:.4f}      {r['micro_recall']:.4f}")
    print(f"\ntotal time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
