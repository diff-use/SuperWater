"""Prepare a custom SuperWater dataset from *_final CIF/PDB files.

Raw layout:
  <raw_data_dir>/<pdb_id>/<pdb_id>_final.cif
  <raw_data_dir>/<pdb_id>/<pdb_id>_final.pdb

Input split files may contain ``<pdb_id>_final``. Output split files contain the
normalized loader id (lowercase ``<pdb_id>``), because SuperWater uses each split row as
both the folder name and file prefix.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import warnings
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBIO import Select
from tqdm import tqdm

from superwater.datasets.water_quality import (
    compute_normalized_bfactors,
    filter_waters_by_quality,
    load_edia_for_pdb,
    normalize_ins_code,
)
from superwater.datasets.structure_quality import (
    check_chain_interactions,
    check_com_distance,
    check_water_clashes,
    check_water_residue_ratio,
)
from superwater.structure_io import parse_structure, candidate_structure_paths


SOURCE_EXTS = (".cif", ".mmcif", ".pdb")


class ProteinOnly(Select):
    def accept_residue(self, residue):
        return residue.get_resname().strip() != "HOH"


@dataclass(frozen=True)
class Entry:
    raw: str
    norm: str


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_data_dir", required=True)
    p.add_argument("--split_train", required=True)
    p.add_argument("--split_val", required=True)
    p.add_argument("--split_test", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split_out_dir", required=True)
    p.add_argument("--embeddings_dir", default=None)
    p.add_argument("--logs_dir", default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--min_water_dist", type=float, default=1.9)
    add_water_filter_args(p)
    add_structure_filter_args(p)
    p.add_argument("--out_format", choices=["cif", "pdb"], default="cif",
                   help="Format for the written _protein_processed and _water files. "
                        "CIF is the default because legacy PDB cannot hold 5-character "
                        "ligand CCD codes (e.g. A1ADA) without column overflow.")
    p.add_argument("--skip_embeddings", action="store_true")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip ids that already have output: a complex with both a "
                        "_protein_processed and a _water file is left untouched, and the "
                        "embedding stage skips complexes that already have _chain_0.pt. "
                        "Makes a re-run incremental (only new/missing ids are processed). "
                        "Note: it does NOT re-fix already-written-but-corrupt files.")
    p.add_argument("--download_missing", action="store_true",
                   help="For split ids absent from --raw_data_dir, download the entry's "
                        "_final.cif/.pdb from PDB-REDO into a temp dir, process it into "
                        "--out_dir, then delete the downloaded files. (Embedding of the "
                        "newly-processed complex happens in the normal embedding stage.)")
    p.add_argument("--tmp_dir", default=None,
                   help="Base dir for transient PDB-REDO downloads (default: a system "
                        "temp dir). Per-entry files are deleted right after processing.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--build_cache", action="store_true")
    p.add_argument("--cache_path", default=None)
    p.add_argument("--cache_scope", choices=["dataset", "split"], default="dataset")
    p.add_argument("--all_atoms", action="store_true", default=True)
    p.add_argument("--no_all_atoms", action="store_false", dest="all_atoms")
    p.add_argument("--remove_hs", action="store_true", default=True)
    p.add_argument("--keep_hs", action="store_false", dest="remove_hs")
    p.add_argument("--receptor_radius", type=float, default=15.0)
    p.add_argument("--c_alpha_max_neighbors", type=int, default=24)
    p.add_argument("--atom_radius", type=float, default=5.0)
    p.add_argument("--atom_max_neighbors", type=int, default=8)
    p.add_argument("--limit_complexes", type=int, default=0)
    return p.parse_args(argv)


def add_water_filter_args(p: argparse.ArgumentParser) -> None:
    """Per-water quality filters (ported from WaterFlow). All ON by default; a water is
    dropped if it fails any enabled filter. Disable individually with --no_filter_by_*."""
    p.add_argument("--max_protein_dist", type=float, default=5.0,
                   help="Drop waters farther than this (A) from the nearest protein atom.")
    p.add_argument("--min_edia", type=float, default=0.4,
                   help="Drop waters with EDIAm below this (from <id>_final.json; "
                        "no-op when the sidecar is absent).")
    p.add_argument("--max_bfactor_zscore", type=float, default=1.5,
                   help="Drop waters whose per-structure B-factor z-score exceeds this.")
    p.add_argument("--filter_by_distance", action="store_true", default=True)
    p.add_argument("--no_filter_by_distance", action="store_false", dest="filter_by_distance")
    p.add_argument("--filter_by_edia", action="store_true", default=True)
    p.add_argument("--no_filter_by_edia", action="store_false", dest="filter_by_edia")
    p.add_argument("--filter_by_bfactor", action="store_true", default=True)
    p.add_argument("--no_filter_by_bfactor", action="store_false", dest="filter_by_bfactor")


def water_filter_cfg(args) -> dict:
    """Build the filter-config dict consumed by ``quality_filtered_coords``."""
    return {
        "filter_by_distance": args.filter_by_distance,
        "filter_by_edia": args.filter_by_edia,
        "filter_by_bfactor": args.filter_by_bfactor,
        "max_protein_dist": args.max_protein_dist,
        "min_edia": args.min_edia,
        "max_bfactor_zscore": args.max_bfactor_zscore,
    }


def add_structure_filter_args(p: argparse.ArgumentParser) -> None:
    """Per-complex structural filters (ported from WaterFlow). All ON by default; a complex
    is dropped if it fails any enabled check. Disable individually with --no_filter_by_*."""
    p.add_argument("--interface_dist_threshold", type=float, default=4.0,
                   help="Multi-chain complex dropped if its closest inter-chain approach "
                        "exceeds this (A) — likely ASU copies, not a real interface.")
    p.add_argument("--max_com_dist", type=float, default=25.0,
                   help="Complex dropped if protein and water centers of mass are farther "
                        "apart than this (A).")
    p.add_argument("--clash_dist", type=float, default=2.0,
                   help="Distance (A) under which a water is counted as clashing with protein.")
    p.add_argument("--max_clash_fraction", type=float, default=0.05,
                   help="Complex dropped if more than this fraction of waters clash.")
    p.add_argument("--min_water_residue_ratio", type=float, default=0.1,
                   help="Complex dropped if waters/residues is below this.")
    p.add_argument("--filter_by_chain", action="store_true", default=True)
    p.add_argument("--no_filter_by_chain", action="store_false", dest="filter_by_chain")
    p.add_argument("--filter_by_com", action="store_true", default=True)
    p.add_argument("--no_filter_by_com", action="store_false", dest="filter_by_com")
    p.add_argument("--filter_by_clash", action="store_true", default=True)
    p.add_argument("--no_filter_by_clash", action="store_false", dest="filter_by_clash")
    p.add_argument("--filter_by_ratio", action="store_true", default=True)
    p.add_argument("--no_filter_by_ratio", action="store_false", dest="filter_by_ratio")


def structure_filter_cfg(args) -> dict:
    """Build the per-complex filter-config dict consumed by ``passes_structure_filters``."""
    return {
        "filter_by_chain": args.filter_by_chain,
        "filter_by_com": args.filter_by_com,
        "filter_by_clash": args.filter_by_clash,
        "filter_by_ratio": args.filter_by_ratio,
        "interface_dist_threshold": args.interface_dist_threshold,
        "max_com_dist": args.max_com_dist,
        "clash_dist": args.clash_dist,
        "max_clash_fraction": args.max_clash_fraction,
        "min_water_residue_ratio": args.min_water_residue_ratio,
    }


def normalize_id(raw: str) -> str:
    stem = Path(raw.strip()).stem.lower()
    if stem.endswith("_final"):
        stem = stem[:-6]
    return stem


def read_split(path: str) -> list[Entry]:
    with open(path) as f:
        return [Entry(line.strip(), normalize_id(line)) for line in f if line.strip()]


def unique_entries(splits: dict[str, list[Entry]]) -> list[Entry]:
    entries: OrderedDict[str, Entry] = OrderedDict()
    for split_entries in splits.values():
        for entry in split_entries:
            entries.setdefault(entry.norm, entry)
    return list(entries.values())


def resolve_sources(raw_root: Path, entry: Entry) -> list[Path]:
    """All existing raw source files for ``entry``, in CIF-first priority order.

    Callers try these in order and fall back to the next (e.g. PDB) when an earlier
    one fails to parse, so a present-but-corrupt CIF does not lose the complex.
    """
    folder = raw_root / entry.norm
    raw_stem = Path(entry.raw).stem.lower()
    stems = []
    for stem in (raw_stem, f"{entry.norm}_final", entry.norm):
        if stem not in stems:
            stems.append(stem)
    found = []
    for ext in SOURCE_EXTS:
        for stem in stems:
            path = folder / f"{stem}{ext}"
            if path.exists():
                found.append(path)
    return found


def resolve_source(raw_root: Path, entry: Entry) -> Path | None:
    sources = resolve_sources(raw_root, entry)
    return sources[0] if sources else None


# Raw sources mirror the PDB-REDO archive layout (<id>_final.cif/.pdb). When a split id
# is absent from --raw_data_dir we fetch that one entry's ZIP and extract just the
# structure files; see clustering/download_pdb_redo.py for the fuller downloader.
PDB_REDO_ZIP_URL = "https://pdb-redo.eu/db/{pdb_id}/zipped"
_DOWNLOAD_SESSION = None


def _download_session():
    global _DOWNLOAD_SESSION
    if _DOWNLOAD_SESSION is not None:
        return _DOWNLOAD_SESSION
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    retry = Retry(total=5, connect=5, read=5, status=5, backoff_factor=1.0,
                  allowed_methods=frozenset({"GET"}),
                  status_forcelist=(429, 500, 502, 503, 504),
                  raise_on_status=False, respect_retry_after_header=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "superwater-setup/1.0 (+https://pdb-redo.eu/)"})
    _DOWNLOAD_SESSION = session
    return session


def download_entry(pdb_id: str, dest_root: Path) -> Path | None:
    """Download ``<pdb_id>_final.cif`` (and ``.pdb`` when present) from PDB-REDO.

    Extracts into ``dest_root/<pdb_id>/`` so ``resolve_sources`` finds it like a raw dir.
    Returns the entry dir on success (at least the CIF or PDB present), else ``None``.
    """
    pdb_id = pdb_id.lower()
    entry_dir = Path(dest_root) / pdb_id
    entry_dir.mkdir(parents=True, exist_ok=True)
    # The _final.json sidecar carries EDIA scores used by the per-water quality filter.
    wanted = {f"{pdb_id}_final.cif", f"{pdb_id}_final.pdb", f"{pdb_id}_final.json"}
    url = PDB_REDO_ZIP_URL.format(pdb_id=pdb_id)

    tmp_zip = None
    try:
        session = _download_session()
        with session.get(url, stream=True, timeout=(30, 600)) as resp:
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(prefix="pdbredo_", suffix=".zip", delete=False) as tmp:
                tmp_zip = Path(tmp.name)
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        tmp.write(chunk)
        found = set()
        with zipfile.ZipFile(tmp_zip) as zf:
            members = {Path(name).name: name for name in zf.namelist()}
            for short in wanted:
                member = members.get(short)
                if member is None:
                    continue
                with zf.open(member) as src, open(entry_dir / short, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
                found.add(short)
    except Exception:
        shutil.rmtree(entry_dir, ignore_errors=True)
        return None
    finally:
        if tmp_zip is not None:
            tmp_zip.unlink(missing_ok=True)

    # Require at least one structure file; a lone JSON sidecar is not usable on its own.
    if not (found & {f"{pdb_id}_final.cif", f"{pdb_id}_final.pdb"}):
        shutil.rmtree(entry_dir, ignore_errors=True)
        return None
    return entry_dir


def is_water_residue(residue) -> bool:
    return residue.get_resname().strip() == "HOH"


def has_protein_residue(structure) -> bool:
    model = next(structure.get_models())
    for residue in model.get_residues():
        if is_water_residue(residue):
            continue
        atom_names = {atom.name for atom in residue}
        if {"CA", "N", "C"} <= atom_names:
            return True
    return False


@dataclass(frozen=True)
class WaterRecord:
    coord: tuple[float, float, float]
    key: tuple[str, int, str]  # (chain_id, res_id, ins_code)
    b_factor: float


def extract_waters(structure, min_water_dist: float) -> list[WaterRecord]:
    """One oxygen per water residue, deduplicated by ``min_water_dist``.

    Each kept water carries its residue key ``(chain_id, res_id, ins_code)`` and raw
    B-factor so downstream quality filters (EDIA / B-factor z-score) can be applied
    before the stripped ``_water`` file (which loses both) is written.
    """
    records: list[WaterRecord] = []
    min_d2 = min_water_dist * min_water_dist
    model = next(structure.get_models())
    for residue in model.get_residues():
        if not is_water_residue(residue):
            continue
        for atom in residue:
            atom_name = atom.name.strip().upper()
            element = atom.element.strip().upper()
            if atom_name != "O" and element != "O":
                continue
            x, y, z = (float(v) for v in atom.coord)
            if any((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < min_d2
                   for cx, cy, cz in (r.coord for r in records)):
                continue
            chain_id = str(residue.get_parent().id)
            key = (chain_id, int(residue.id[1]), normalize_ins_code(residue.id[2]))
            records.append(WaterRecord((x, y, z), key, float(atom.get_bfactor())))
            break
    return records


def extract_protein_coords(structure) -> list[tuple[float, float, float]]:
    """All non-water atom coordinates (used for the distance-to-protein filter)."""
    model = next(structure.get_models())
    return [tuple(float(v) for v in atom.coord)
            for residue in model.get_residues() if not is_water_residue(residue)
            for atom in residue]


def extract_protein_coords_by_chain(structure) -> dict[str, np.ndarray]:
    """Non-water atom coordinates grouped by chain id (for the multi-chain filter)."""
    model = next(structure.get_models())
    by_chain: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for residue in model.get_residues():
        if is_water_residue(residue):
            continue
        cid = str(residue.get_parent().id)
        for atom in residue:
            by_chain[cid].append(tuple(float(v) for v in atom.coord))
    return {cid: np.array(coords, dtype=float) for cid, coords in by_chain.items()}


def count_protein_residues(structure) -> int:
    """Number of non-water residues (denominator of the water/residue ratio)."""
    model = next(structure.get_models())
    return sum(1 for residue in model.get_residues() if not is_water_residue(residue))


def passes_structure_filters(structure, waters, filter_cfg, cache_key):
    """Run the enabled per-complex checks. Return ``(True, "")`` to keep, else the skip
    status + reason. ``waters`` are the surviving (quality-filtered) water coordinates."""
    water_coords = np.array(waters, dtype=float)
    protein_coords = np.array(extract_protein_coords(structure), dtype=float)

    if filter_cfg["filter_by_chain"]:
        ok, reason = check_chain_interactions(
            extract_protein_coords_by_chain(structure),
            interface_dist_threshold=filter_cfg["interface_dist_threshold"])
        if not ok:
            return False, "asu_copies", reason
    if filter_cfg["filter_by_com"]:
        ok, reason = check_com_distance(protein_coords, water_coords,
                                        max_com_dist=filter_cfg["max_com_dist"])
        if not ok:
            return False, "com_distance", reason
    if filter_cfg["filter_by_clash"]:
        ok, reason = check_water_clashes(protein_coords, water_coords,
                                         clash_dist=filter_cfg["clash_dist"],
                                         max_clash_fraction=filter_cfg["max_clash_fraction"])
        if not ok:
            return False, "water_clash", reason
    if filter_cfg["filter_by_ratio"]:
        ok, reason = check_water_residue_ratio(len(waters), count_protein_residues(structure),
                                               min_ratio=filter_cfg["min_water_residue_ratio"])
        if not ok:
            return False, "low_water_ratio", reason
    return True, "", ""


def write_water_pdb(coords: list[tuple[float, float, float]], path: Path) -> None:
    rows = []
    for i, (x, y, z) in enumerate(coords, start=1):
        resseq = (i - 1) % 9999 + 1
        rows.append(
            f"HETATM{i:>5}  O   HOH W{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}"
            "  1.00  0.00           O\n"
        )
    rows.append("END\n")
    path.write_text("".join(rows))


def write_water_cif(coords: list[tuple[float, float, float]], path: Path) -> None:
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.mmcifio import MMCIFIO

    structure = Structure("water")
    model = Model(0)
    chain = Chain("W")
    for i, (x, y, z) in enumerate(coords, start=1):
        # ('W', resseq, icode) marks each oxygen as its own water residue.
        residue = Residue(("W", i, " "), "HOH", " ")
        residue.add(Atom("O", [x, y, z], 0.0, 1.0, " ", "O", i, "O"))
        chain.add(residue)
    model.add(chain)
    structure.add(model)
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(str(path))


def write_protein(structure, path: Path, fmt: str) -> None:
    """Write the protein (non-HOH residues) to ``path`` in ``fmt`` ('cif' or 'pdb').

    CIF is preferred: BioPython's PDBIO truncates/overflows residue names longer than
    3 characters, which silently corrupts structures that use the newer 5-character
    ligand CCD codes (e.g. ``A1ADA``). mmCIF is free-format and round-trips them.
    """
    if fmt == "cif":
        from Bio.PDB.mmcifio import MMCIFIO
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    io.save(str(path), ProteinOnly())


def write_water(coords: list[tuple[float, float, float]], path: Path, fmt: str) -> None:
    if fmt == "cif":
        write_water_cif(coords, path)
    else:
        write_water_pdb(coords, path)


def quality_filtered_coords(records, structure, source: Path, filter_cfg: dict, cache_key: str):
    """Apply the enabled per-water quality filters and return surviving coordinates.

    ``filter_cfg`` carries the thresholds and the three ``filter_by_*`` toggles. EDIA is
    read from the ``<source_stem>.json`` sidecar; when any input is unavailable that
    filter degrades to a no-op (all waters pass it).
    """
    if not any((filter_cfg["filter_by_distance"], filter_cfg["filter_by_edia"],
                filter_cfg["filter_by_bfactor"])):
        return [r.coord for r in records]

    water_coords = np.array([r.coord for r in records], dtype=float)
    water_keys = [r.key for r in records]

    protein_coords = None
    if filter_cfg["filter_by_distance"]:
        protein_coords = np.array(extract_protein_coords(structure), dtype=float)

    edia_lookup = None
    if filter_cfg["filter_by_edia"]:
        edia_lookup = load_edia_for_pdb(source.with_suffix(".json"))

    bfactor_lookup = None
    if filter_cfg["filter_by_bfactor"]:
        bfactor_lookup = compute_normalized_bfactors([(r.key, r.b_factor) for r in records])

    keep_mask = filter_waters_by_quality(
        water_coords, water_keys, protein_coords, edia_lookup, bfactor_lookup,
        max_protein_dist=filter_cfg["max_protein_dist"],
        min_edia=filter_cfg["min_edia"],
        max_bfactor_zscore=filter_cfg["max_bfactor_zscore"],
        cache_key=cache_key,
    )
    return [r.coord for r, keep in zip(records, keep_mask) if keep]


def process_one(job):
    raw_root, out_dir, entry, min_water_dist, out_format, skip_existing, \
        download_missing, tmp_root, filter_cfg, struct_cfg = job
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    dest = out_dir / entry.norm
    if skip_existing \
            and candidate_structure_paths(str(dest), entry.norm, "_protein_processed") \
            and candidate_structure_paths(str(dest), entry.norm, "_water"):
        return entry.norm, "exists", ""

    downloaded_dir = None  # transient PDB-REDO download to delete once processed
    try:
        sources = resolve_sources(raw_root, entry)
        if not sources and download_missing and tmp_root is not None:
            downloaded_dir = download_entry(entry.norm, Path(tmp_root))
            if downloaded_dir is not None:
                sources = resolve_sources(Path(tmp_root), entry)
        if not sources:
            return entry.norm, "missing_source", ""

        structure, source, last_exc = None, None, None
        for candidate in sources:  # CIF first, fall back to PDB if it fails to parse
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", PDBConstructionWarning)
                    structure = parse_structure(str(candidate), permissive=True, structure_id=entry.norm)
                source = candidate
                break
            except Exception as exc:
                last_exc = exc
        if structure is None:
            return entry.norm, "parse_error", str(last_exc)

        if not has_protein_residue(structure):
            return entry.norm, "no_protein", ""

        records = extract_waters(structure, min_water_dist)
        if not records:
            return entry.norm, "no_waters", ""

        waters = quality_filtered_coords(records, structure, source, filter_cfg, entry.norm)
        if not waters:
            return entry.norm, "no_waters_after_filter", ""

        keep, status, reason = passes_structure_filters(structure, waters, struct_cfg, entry.norm)
        if not keep:
            return entry.norm, status, reason

        dest.mkdir(parents=True, exist_ok=True)
        protein_path = dest / f"{entry.norm}_protein_processed.{out_format}"
        water_path = dest / f"{entry.norm}_water.{out_format}"

        try:
            write_protein(structure, protein_path, out_format)
        except Exception as exc:
            print(f"[SKIP] {entry.norm}: write_error — {exc}")
            return entry.norm, "write_error", str(exc)
        write_water(waters, water_path, out_format)

        # waters column is "<kept>/<pre-filter>" so the log shows how many were dropped.
        meta = f"{source}\t{len(waters)}/{len(records)}\t{protein_path}\t{water_path}"
        return entry.norm, ("downloaded" if downloaded_dir is not None else "ok"), meta
    finally:
        if downloaded_dir is not None:
            shutil.rmtree(downloaded_dir, ignore_errors=True)


def write_lines(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows))


def prebuild_cache(args, split_paths: dict[str, Path], embeddings_dir: Path) -> None:
    from superwater.datasets.pdbbind import NoiseTransform, PDBBind
    from superwater.utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
    from superwater.utils.parsing import parse_train_args

    train_args = parse_train_args([])
    train_args.tr_sigma_min = 0.1
    train_args.tr_sigma_max = 30.0
    train_args.all_atoms = args.all_atoms
    transform = NoiseTransform(partial(t_to_sigma_compl, args=train_args), all_atom=args.all_atoms)
    cache_path = args.cache_path or str(Path(str(args.out_dir) + "_cache"))

    for name, split_path in split_paths.items():
        print(f"Prebuilding {name} graph cache: {split_path}")
        PDBBind(
            root=args.out_dir,
            transform=transform,
            cache_path=cache_path,
            split_path=str(split_path),
            limit_complexes=args.limit_complexes,
            receptor_radius=args.receptor_radius,
            c_alpha_max_neighbors=args.c_alpha_max_neighbors,
            remove_hs=args.remove_hs,
            num_workers=args.num_workers,
            all_atoms=args.all_atoms,
            atom_radius=args.atom_radius,
            atom_max_neighbors=args.atom_max_neighbors,
            esm_embeddings_path=str(embeddings_dir),
            cache_scope=args.cache_scope,
        )


def main(argv=None):
    args = parse_args(argv)
    if args.out_format == "pdb":
        # Legacy PDB truncates/overflows 5-character ligand CCD codes (e.g. A1ADA),
        # silently corrupting the protein file and any embedding derived from it. CIF
        # round-trips them; prefer it unless you are certain none of your structures use
        # 5-char codes. (Run scripts/audit_dataset.py later to spot any damage.)
        print("WARNING: --out_format pdb can corrupt structures with 5-character CCD "
              "codes; CIF (the default) is strongly recommended.")
    raw_root = Path(args.raw_data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    split_out_dir = Path(args.split_out_dir).expanduser().resolve()
    embeddings_dir = Path(args.embeddings_dir).expanduser().resolve() if args.embeddings_dir else Path(str(out_dir) + "_embeddings")
    logs_dir = Path(args.logs_dir).expanduser().resolve() if args.logs_dir else out_dir / "logs"

    splits = {
        "train": read_split(args.split_train),
        "val": read_split(args.split_val),
        "test": read_split(args.split_test),
    }
    all_entries = unique_entries(splits)
    print(f"Preparing {len(all_entries)} unique ids from {raw_root}")

    tmp_root = None
    if args.download_missing:
        tmp_root = Path(tempfile.mkdtemp(prefix="superwater_dl_",
                                         dir=args.tmp_dir if args.tmp_dir else None))

    filter_cfg = water_filter_cfg(args)
    struct_cfg = structure_filter_cfg(args)
    jobs = [(str(raw_root), str(out_dir), entry, args.min_water_dist, args.out_format,
             args.skip_existing, args.download_missing,
             str(tmp_root) if tmp_root is not None else None, filter_cfg, struct_cfg)
            for entry in all_entries]
    prepared, skipped, metadata = set(), defaultdict(list), {}
    n_exists = 0
    n_downloaded = 0

    def record(cid, status, meta):
        nonlocal n_exists, n_downloaded
        if status in ("ok", "downloaded"):
            prepared.add(cid)
            metadata[cid] = meta
            if status == "downloaded":
                n_downloaded += 1
        elif status == "exists":  # already on disk -> keep it in the split, don't redo
            prepared.add(cid)
            n_exists += 1
        else:
            skipped[status].append(f"{cid}\t{meta}" if meta else cid)

    try:
        if args.num_workers > 1:
            with Pool(args.num_workers) as pool:
                iterator = pool.imap_unordered(process_one, jobs, chunksize=16)
                for cid, status, meta in tqdm(iterator, total=len(jobs), desc="preparing"):
                    record(cid, status, meta)
        else:
            for job in tqdm(jobs, total=len(jobs), desc="preparing"):
                record(*process_one(job))
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)

    logs_dir.mkdir(parents=True, exist_ok=True)
    for reason, ids in sorted(skipped.items()):
        write_lines(logs_dir / f"skipped_{reason}.txt", sorted(ids))
    with open(logs_dir / "prepared.tsv", "w") as f:
        f.write("id\tsource\twaters_kept/total\tprotein\twater\n")
        for cid in sorted(metadata):
            f.write(f"{cid}\t{metadata[cid]}\n")

    split_paths = {}
    for split_name, entries in splits.items():
        ids = [entry.norm for entry in entries if entry.norm in prepared]
        path = split_out_dir / f"{split_name}.txt"
        write_lines(path, ids)
        split_paths[split_name] = path
        print(f"Wrote {len(ids)} {split_name} ids -> {path}")

    extra = []
    if n_exists:
        extra.append(f"{n_exists} already existed, skipped")
    if n_downloaded:
        extra.append(f"{n_downloaded} downloaded from PDB-REDO")
    suffix = f" ({'; '.join(extra)})" if extra else ""
    print(f"Prepared {len(prepared)} ids{suffix}. Logs -> {logs_dir}")

    if not args.skip_embeddings:
        import torch
        from superwater.esm_embeddings import embed_dataset

        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested for embeddings, but torch.cuda.is_available() is false.")
        print(f"Generating ESM embeddings -> {embeddings_dir}")
        embed_dataset(str(out_dir), str(embeddings_dir), torch.device(args.device),
                      skip_existing=args.skip_existing)

    if args.build_cache:
        if args.skip_embeddings and not embeddings_dir.exists():
            raise SystemExit("--build_cache needs embeddings; pass --embeddings_dir or omit --skip_embeddings.")
        prebuild_cache(args, split_paths, embeddings_dir)


if __name__ == "__main__":
    main()
