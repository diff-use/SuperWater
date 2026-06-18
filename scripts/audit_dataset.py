"""Read-only health audit of an organized SuperWater dataset.

Walks each complex folder under ``--data_dir`` and reports problems without changing
anything. Per-complex categories:

  Errors (a complex is unusable / suspect):
    missing_protein       no <id>_protein_processed.{cif,pdb}
    missing_water         no <id>_water.{cif,pdb}
    protein_parse_error   the protein file fails to parse
    water_parse_error     the water file fails to parse
    no_waters             water file parses but holds zero oxygens
    missing_embedding     (only with --embeddings_dir) no <id>_chain_0.pt

  Warnings (works, but worth re-deriving):
    legacy_pdb_protein    protein stored as legacy .pdb; CIF is preferred because PDB
                          can silently corrupt 5-character ligand CCD codes (e.g. A1ADA)

Writes ``<data_dir>/logs/audit_report.tsv`` and exits non-zero when any complex has an
error (so a pipeline can gate on it). ``--strict`` also fails on warnings.

There is no separate repair script: the supported fix is to remove a broken complex's
processed files and re-run ``setup_custom_dataset.py --skip_existing`` (which re-derives
them as CIF and re-embeds). ``--delete_broken`` does that removal for you, in place.
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from collections import defaultdict
from multiprocessing import Pool

from Bio.PDB.PDBExceptions import PDBConstructionWarning
from tqdm import tqdm

from superwater.structure_io import candidate_structure_paths, parse_structure

# Categories treated as errors (vs. warnings) for exit-code and --delete_broken purposes.
ERROR_CATEGORIES = frozenset({
    "missing_protein", "missing_water", "protein_parse_error",
    "water_parse_error", "no_waters", "missing_embedding",
})


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Organized dataset dir (one subfolder per complex).")
    p.add_argument("--embeddings_dir", default=None,
                   help="If given, also flag complexes missing a <id>_chain_0.pt embedding.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--report", default=None,
                   help="Where to write the TSV report (default: <data_dir>/logs/audit_report.tsv). "
                        "Point elsewhere when --data_dir is read-only or shared.")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on warnings too, not just errors.")
    p.add_argument("--delete_broken", action="store_true",
                   help="DESTRUCTIVE: delete the processed files of complexes with errors so a "
                        "later `setup_custom_dataset.py --skip_existing` re-derives them as CIF.")
    p.add_argument("--limit", type=int, default=0, help="Audit at most this many complexes (0 = all).")
    return p.parse_args(argv)


def _short(exc: Exception, limit: int = 200) -> str:
    return str(exc).splitlines()[0][:limit] if str(exc) else exc.__class__.__name__


def _count_oxygens(path: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        structure = parse_structure(path, permissive=True)
    return sum(1 for atom in structure.get_atoms() if atom.element.strip().upper() == "O")


def audit_one(job):
    """Return ``(cid, errors, warnings_, detail)`` for one complex (read-only)."""
    data_dir, cid, embeddings_dir = job
    entry_dir = os.path.join(data_dir, cid)
    errors: list[str] = []
    warns: list[str] = []
    detail = ""

    protein = candidate_structure_paths(entry_dir, cid, "_protein_processed")
    if not protein:
        errors.append("missing_protein")
    else:
        if protein[0].lower().endswith(".pdb"):
            warns.append("legacy_pdb_protein")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PDBConstructionWarning)
                parse_structure(protein[0], permissive=True, structure_id=cid)
        except Exception as exc:  # noqa: BLE001 -- record, don't abort the audit
            errors.append("protein_parse_error")
            detail = _short(exc)

    water = candidate_structure_paths(entry_dir, cid, "_water")
    if not water:
        errors.append("missing_water")
    else:
        try:
            if _count_oxygens(water[0]) == 0:
                errors.append("no_waters")
        except Exception as exc:  # noqa: BLE001
            errors.append("water_parse_error")
            detail = detail or _short(exc)

    if embeddings_dir is not None and not os.path.exists(
        os.path.join(embeddings_dir, f"{cid}_chain_0.pt")
    ):
        errors.append("missing_embedding")

    return cid, errors, warns, detail


def delete_processed(data_dir: str, cid: str) -> None:
    """Remove a complex's processed structure files so prep re-derives them."""
    entry_dir = os.path.join(data_dir, cid)
    for suffix in ("_protein_processed", "_water"):
        for path in candidate_structure_paths(entry_dir, cid, suffix):
            os.remove(path)


def main(argv=None):
    args = parse_args(argv)
    data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    embeddings_dir = (os.path.abspath(os.path.expanduser(args.embeddings_dir))
                      if args.embeddings_dir else None)

    names = sorted(d for d in os.listdir(data_dir)
                   if d != "logs" and os.path.isdir(os.path.join(data_dir, d)))
    if args.limit:
        names = names[: args.limit]
    print(f"Auditing {len(names)} complexes in {data_dir}")

    jobs = [(data_dir, cid, embeddings_dir) for cid in names]
    results = []
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(audit_one, jobs, chunksize=16),
                            total=len(jobs), desc="auditing"):
                results.append(res)
    else:
        for job in tqdm(jobs, total=len(jobs), desc="auditing"):
            results.append(audit_one(job))

    by_category: dict[str, list[str]] = defaultdict(list)
    error_ids, ok = [], 0
    for cid, errors, warns, _ in results:
        for cat in errors + warns:
            by_category[cat].append(cid)
        if errors:
            error_ids.append(cid)
        elif not warns:
            ok += 1

    report = (os.path.abspath(os.path.expanduser(args.report)) if args.report
              else os.path.join(data_dir, "logs", "audit_report.tsv"))
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as f:
        f.write("id\terrors\twarnings\tdetail\n")
        for cid, errors, warns, detail in sorted(results):
            if errors or warns:
                f.write(f"{cid}\t{','.join(errors)}\t{','.join(warns)}\t{detail}\n")

    print(f"\n{ok}/{len(names)} complexes clean. Report -> {report}")
    for cat in sorted(by_category):
        kind = "ERROR" if cat in ERROR_CATEGORIES else "warn "
        print(f"  [{kind}] {cat}: {len(by_category[cat])}")

    if args.delete_broken and error_ids:
        print(f"\nDeleting processed files for {len(error_ids)} broken complexes "
              f"(re-run setup_custom_dataset.py --skip_existing to rebuild them)...")
        for cid in error_ids:
            delete_processed(data_dir, cid)

    warn_ids = [cid for cid, e, w, _ in results if w and not e]
    has_problem = bool(error_ids) or (args.strict and bool(warn_ids))
    raise SystemExit(1 if has_problem else 0)


if __name__ == "__main__":
    main()
