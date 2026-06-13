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
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from Bio.PDB import PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.PDBIO import Select
from tqdm import tqdm

from superwater.structure_io import parse_structure


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
    p.add_argument("--skip_embeddings", action="store_true")
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


def resolve_source(raw_root: Path, entry: Entry) -> Path | None:
    folder = raw_root / entry.norm
    raw_stem = Path(entry.raw).stem.lower()
    stems = []
    for stem in (raw_stem, f"{entry.norm}_final", entry.norm):
        if stem not in stems:
            stems.append(stem)
    for ext in SOURCE_EXTS:
        for stem in stems:
            path = folder / f"{stem}{ext}"
            if path.exists():
                return path
    return None


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


def extract_water_coords(structure, min_water_dist: float) -> list[tuple[float, float, float]]:
    coords = []
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
            if any((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < min_d2 for cx, cy, cz in coords):
                continue
            coords.append((x, y, z))
            break
    return coords


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


def process_one(job):
    raw_root, out_dir, entry, min_water_dist = job
    raw_root = Path(raw_root)
    out_dir = Path(out_dir)
    source = resolve_source(raw_root, entry)
    if source is None:
        return entry.norm, "missing_source", ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            structure = parse_structure(str(source), permissive=True, structure_id=entry.norm)
    except Exception as exc:
        return entry.norm, "parse_error", str(exc)

    if not has_protein_residue(structure):
        return entry.norm, "no_protein", ""

    waters = extract_water_coords(structure, min_water_dist)
    if not waters:
        return entry.norm, "no_waters", ""

    dest = out_dir / entry.norm
    dest.mkdir(parents=True, exist_ok=True)
    protein_path = dest / f"{entry.norm}_protein_processed.pdb"
    water_path = dest / f"{entry.norm}_water.pdb"

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(protein_path), ProteinOnly())
    write_water_pdb(waters, water_path)

    meta = f"{source}\t{len(waters)}\t{protein_path}\t{water_path}"
    return entry.norm, "ok", meta


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
    raw_root = Path(args.raw_data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    split_out_dir = Path(args.split_out_dir).expanduser().resolve()
    embeddings_dir = Path(args.embeddings_dir).expanduser().resolve() if args.embeddings_dir else Path(str(out_dir) + "_embeddings")
    logs_dir = Path(args.logs_dir).expanduser().resolve() if args.logs_dir else Path(str(out_dir) + "_setup_logs")

    splits = {
        "train": read_split(args.split_train),
        "val": read_split(args.split_val),
        "test": read_split(args.split_test),
    }
    all_entries = unique_entries(splits)
    print(f"Preparing {len(all_entries)} unique ids from {raw_root}")

    jobs = [(str(raw_root), str(out_dir), entry, args.min_water_dist) for entry in all_entries]
    prepared, skipped, metadata = set(), defaultdict(list), {}

    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            iterator = pool.imap_unordered(process_one, jobs, chunksize=16)
            for cid, status, meta in tqdm(iterator, total=len(jobs), desc="preparing"):
                if status == "ok":
                    prepared.add(cid)
                    metadata[cid] = meta
                else:
                    skipped[status].append(f"{cid}\t{meta}" if meta else cid)
    else:
        for job in tqdm(jobs, total=len(jobs), desc="preparing"):
            cid, status, meta = process_one(job)
            if status == "ok":
                prepared.add(cid)
                metadata[cid] = meta
            else:
                skipped[status].append(f"{cid}\t{meta}" if meta else cid)

    logs_dir.mkdir(parents=True, exist_ok=True)
    for reason, ids in sorted(skipped.items()):
        write_lines(logs_dir / f"skipped_{reason}.txt", sorted(ids))
    with open(logs_dir / "prepared.tsv", "w") as f:
        f.write("id\tsource\twaters\tprotein\twater\n")
        for cid in sorted(metadata):
            f.write(f"{cid}\t{metadata[cid]}\n")

    split_paths = {}
    for split_name, entries in splits.items():
        ids = [entry.norm for entry in entries if entry.norm in prepared]
        path = split_out_dir / f"{split_name}.txt"
        write_lines(path, ids)
        split_paths[split_name] = path
        print(f"Wrote {len(ids)} {split_name} ids -> {path}")

    print(f"Prepared {len(prepared)} ids. Logs -> {logs_dir}")

    if not args.skip_embeddings:
        import torch
        from superwater.esm_embeddings import embed_dataset

        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested for embeddings, but torch.cuda.is_available() is false.")
        print(f"Generating ESM embeddings -> {embeddings_dir}")
        embed_dataset(str(out_dir), str(embeddings_dir), torch.device(args.device))

    if args.build_cache:
        if args.skip_embeddings and not embeddings_dir.exists():
            raise SystemExit("--build_cache needs embeddings; pass --embeddings_dir or omit --skip_embeddings.")
        prebuild_cache(args, split_paths, embeddings_dir)


if __name__ == "__main__":
    main()
