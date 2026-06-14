"""One-off fix for complexes the original setup wrote as corrupt PDBs.

These complexes use 5-character ligand CCD codes (e.g. ``A1ADA``) that overflow legacy
PDB columns, so their written ``_protein_processed.pdb`` -- and therefore their ESM
embeddings -- are broken. This re-derives ``_protein_processed.cif`` (and ``_water.cif``
when no water file exists yet) from the raw ``_final.cif`` (CIF preferred, PDB fallback)
and regenerates embeddings for just those ids. Afterwards, re-run training: dataset-scope
caching builds only the now-missing graphs.

By default the failed ids are read from the setup/embedding skip logs under
``<data_dir>/logs`` (the column-overflow ``invalid literal for int`` entries plus any
``skipped_write_error.txt`` ids); pass ``--ids_file`` to override.

    uv run python scripts/fix_failed_entries.py \
        --raw_data_dir /mnt/diffuse-shared/migrated/diffuse-public/water_data/pdb_data \
        --data_dir ~/sw_cache/data/full_pdb_dataset \
        --embeddings_dir ~/sw_cache/data/full_pdb_embeddings
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import setup_custom_dataset as scd  # noqa: E402  (sibling script, reuses its helpers)

from Bio.PDB.PDBExceptions import PDBConstructionWarning  # noqa: E402
from superwater.structure_io import parse_structure, candidate_structure_paths  # noqa: E402


def load_failed_ids(data_dir: str, ids_file: str | None) -> list[str]:
    if ids_file:
        return sorted({ln.strip().lower() for ln in open(ids_file) if ln.strip()})
    logs = Path(data_dir) / "logs"
    ids: set[str] = set()
    emb = logs / "skipped_embedding_errors.txt"
    if emb.exists():
        for line in open(emb):
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and "invalid literal for int" in parts[1]:
                ids.add(parts[0].strip().lower())
    write_err = logs / "skipped_write_error.txt"
    if write_err.exists():
        for line in open(write_err):
            cid = line.split("\t")[0].strip().lower()
            if cid:
                ids.add(cid)
    return sorted(ids)


def rebuild_one(raw_root: Path, data_dir: Path, cid: str, min_water_dist: float):
    """Write <cid>_protein_processed.cif (+ _water.cif if missing) from the raw source.

    Returns (status, detail). status == 'ok' on success.
    """
    entry = scd.Entry(raw=cid, norm=scd.normalize_id(cid))
    sources = scd.resolve_sources(raw_root, entry)
    if not sources:
        return "missing_source", ""

    structure, last_exc = None, None
    for src in sources:  # CIF first, fall back to PDB if it fails to parse
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PDBConstructionWarning)
                structure = parse_structure(str(src), permissive=True, structure_id=entry.norm)
            break
        except Exception as exc:
            last_exc = exc
    if structure is None:
        return "parse_error", str(last_exc)
    if not scd.has_protein_residue(structure):
        return "no_protein", ""

    dest = data_dir / entry.norm
    dest.mkdir(parents=True, exist_ok=True)
    try:
        scd.write_protein(structure, dest / f"{entry.norm}_protein_processed.cif", "cif")
    except Exception as exc:
        return "write_error", str(exc)

    # Only (re)write water if no usable water file already exists for this complex.
    if not candidate_structure_paths(str(dest), entry.norm, "_water"):
        waters = scd.extract_water_coords(structure, min_water_dist)
        if not waters:
            return "no_waters", ""
        scd.write_water(waters, dest / f"{entry.norm}_water.cif", "cif")
    return "ok", ""


def embed_ids(data_dir: Path, embeddings_dir: Path, ids: list[str], device_str: str):
    import torch
    from superwater.esm_embeddings import load_esm_model, embed_complex

    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    model, alphabet = load_esm_model(device)
    ok = bad = 0
    for cid in ids:
        candidates = candidate_structure_paths(str(data_dir / cid), cid, "_protein_processed")
        last_exc, embedded = None, False
        for path in candidates:  # CIF first, fall back to PDB
            try:
                embed_complex(cid, path, str(embeddings_dir), model, alphabet, device)
                embedded = True
                break
            except Exception as exc:
                last_exc = exc
        if embedded:
            ok += 1
        else:
            bad += 1
            print(f"  embed FAIL {cid}: {last_exc}")
    print(f"Embedded {ok}, failed {bad} -> {embeddings_dir}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw_data_dir", required=True)
    p.add_argument("--data_dir", required=True, help="Existing organized dataset dir to fix in place.")
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--ids_file", default=None,
                   help="Explicit id list; default reads the setup/embedding skip logs.")
    p.add_argument("--min_water_dist", type=float, default=1.9)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--skip_embeddings", action="store_true")
    args = p.parse_args(argv)

    raw_root = Path(args.raw_data_dir).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    embeddings_dir = Path(args.embeddings_dir).expanduser().resolve()
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    ids = load_failed_ids(str(data_dir), args.ids_file)
    print(f"Fixing {len(ids)} complexes from {raw_root}")

    rebuilt, by_status = [], {}
    for cid in ids:
        status, detail = rebuild_one(raw_root, data_dir, cid, args.min_water_dist)
        by_status.setdefault(status, []).append(f"{cid}\t{detail}" if detail else cid)
        if status == "ok":
            rebuilt.append(cid)
    for status, rows in sorted(by_status.items()):
        print(f"  {status}: {len(rows)}")
    print(f"Rebuilt _protein_processed.cif for {len(rebuilt)} complexes")

    if args.skip_embeddings or not rebuilt:
        return
    embed_ids(data_dir, embeddings_dir, rebuilt, args.device)


if __name__ == "__main__":
    main()
