"""Structure-format helpers: convert input CIF/mmCIF to PDB, and write predicted
water positions as PDB or mmCIF.

Input proteins may be supplied as .pdb, .cif or .mmcif; the rest of the pipeline works
on PDB, so CIF inputs are converted up front. Predicted waters (plain xyz text) can be
written back as either .pdb or .cif.
"""
import os
import shutil

import numpy as np

SUPPORTED_STRUCTURE_EXTS = (".pdb", ".cif", ".mmcif")


def structure_path(directory, name, suffix):
    """Resolve a per-complex structure file, preferring CIF over PDB.

    Returns ``<directory>/<name><suffix>.cif`` if it exists, else the ``.pdb`` path
    (which may or may not exist; callers handle a missing file as before). This lets the
    data layer read native CIF when present while staying backward-compatible with the
    older PDB-only layout.
    """
    cif = os.path.join(directory, f"{name}{suffix}.cif")
    if os.path.exists(cif):
        return cif
    return os.path.join(directory, f"{name}{suffix}.pdb")


def parse_structure(path, permissive=False, structure_id="structure"):
    """Parse a .pdb/.cif/.mmcif file into a Bio.PDB ``Structure``.

    CIF/mmCIF is parsed with ``MMCIFParser`` (which has no PERMISSIVE distinction), PDB
    with ``PDBParser(PERMISSIVE=permissive)``. The returned object type is identical, so
    downstream chain/residue/atom iteration is unaffected by the input format.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        from Bio.PDB.MMCIFParser import MMCIFParser
        return MMCIFParser(QUIET=True).get_structure(structure_id, path)
    from Bio.PDB import PDBParser
    return PDBParser(PERMISSIVE=permissive, QUIET=True).get_structure(structure_id, path)


def candidate_structure_paths(directory, name, suffix):
    """Existing structure files for ``<name><suffix>``, in CIF-first priority order.

    Returns ``[<dir>/<name><suffix>.cif, .mmcif, .pdb]`` filtered to those that exist.
    CIF is listed first because legacy PDB cannot hold 5-character ligand CCD codes
    without column overflow; PDB is the fallback when no CIF is present (or it fails).
    """
    paths = []
    for ext in (".cif", ".mmcif", ".pdb"):
        p = os.path.join(directory, f"{name}{suffix}{ext}")
        if os.path.exists(p):
            paths.append(p)
    return paths


def parse_structure_with_fallback(directory, name, suffix, permissive=False,
                                  structure_id="structure"):
    """Parse the first candidate file that parses successfully (CIF preferred, PDB fallback).

    Returns ``(structure, path)``. If a CIF exists but raises while parsing, the next
    candidate (e.g. the PDB) is tried instead of failing outright. Raises
    ``FileNotFoundError`` when no candidate exists, or re-raises the last parse error
    when every candidate fails.
    """
    candidates = candidate_structure_paths(directory, name, suffix)
    if not candidates:
        raise FileNotFoundError(f"No structure file for {name}{suffix} in {directory}")
    last_exc = None
    for path in candidates:
        try:
            return parse_structure(path, permissive=permissive, structure_id=structure_id), path
        except Exception as exc:  # try the next format (e.g. PDB) before giving up
            last_exc = exc
    raise last_exc


def convert_txt_to_pdb(txt_file_path, output_pdb_path):
    """Write water oxygen coordinates (one ``x y z`` per line) as HETATM/HOH records.

    Each water is written as its own HOH residue (resSeq 1..N, wrapping at 9999 to fit the
    4-column field) so the file round-trips through strict PDB parsers and the water count
    is preserved; viewers still render each oxygen as a separate water.
    """
    with open(txt_file_path, "r") as file:
        lines = [ln for ln in file.readlines() if ln.strip()]

    pdb_lines = []
    for i, line in enumerate(lines, start=1):
        x, y, z = map(float, line.split()[:3])
        resseq = (i - 1) % 9999 + 1
        pdb_lines.append(
            f"HETATM{i:>5}  O   HOH A{resseq:>4}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\n")

    with open(output_pdb_path, "w") as pdb_file:
        pdb_file.writelines(pdb_lines)
    print(f"Successfully saved PDB file to: {output_pdb_path}")


def convert_txt_to_cif(txt_file_path, output_cif_path):
    """Write water oxygen coordinates as an mmCIF file (one HOH residue per water)."""
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    from Bio.PDB.mmcifio import MMCIFIO

    coords = np.loadtxt(txt_file_path)
    if coords.size == 0:
        coords = np.empty((0, 3))
    elif coords.ndim == 1:
        coords = coords.reshape(1, -1)

    structure = Structure("water")
    model = Model(0)
    chain = Chain("A")
    for i, row in enumerate(coords, start=1):
        # ('W', resseq, icode) marks each oxygen as its own water residue.
        residue = Residue(("W", i, " "), "HOH", " ")
        residue.add(Atom("O", [float(row[0]), float(row[1]), float(row[2])],
                         0.0, 1.0, " ", "O", i, "O"))
        chain.add(residue)
    model.add(chain)
    structure.add(model)

    io = MMCIFIO()
    io.set_structure(structure)
    io.save(output_cif_path)
    print(f"Successfully saved CIF file to: {output_cif_path}")


def write_water_structure(txt_file_path, out_path, fmt="pdb"):
    """Write predicted waters to ``out_path`` in ``fmt`` ('pdb' or 'cif')."""
    fmt = fmt.lower()
    if fmt == "pdb":
        convert_txt_to_pdb(txt_file_path, out_path)
    elif fmt == "cif":
        convert_txt_to_cif(txt_file_path, out_path)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r} (use 'pdb' or 'cif')")


def _free_chain_id(model, preferred="WXYZ"):
    """Return a single-character chain id not already used in ``model``."""
    used = {chain.id for chain in model}
    candidates = list(preferred) + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
        + [chr(c) for c in range(ord("a"), ord("z") + 1)] + [str(d) for d in range(10)]
    for cid in candidates:
        if cid not in used:
            return cid
    return "W"


def write_protein_with_waters(protein_pdb_path, water_txt_path, out_path, fmt="pdb"):
    """Write the input protein plus predicted waters to ``out_path`` (pdb or cif).

    Waters are added as HOH oxygens on a dedicated chain so their residue numbering does
    not collide with the protein's. ``protein_pdb_path`` is the working PDB (already
    converted from CIF/mmCIF when needed).
    """
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.mmcifio import MMCIFIO
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom

    structure = PDBParser(QUIET=True).get_structure("complex", protein_pdb_path)
    model = structure[0]

    coords = np.loadtxt(water_txt_path)
    if coords.size == 0:
        coords = np.empty((0, 3))
    elif coords.ndim == 1:
        coords = coords.reshape(1, -1)

    water_chain = Chain(_free_chain_id(model))
    for i, row in enumerate(coords, start=1):
        residue = Residue(("W", i, " "), "HOH", " ")
        residue.add(Atom("O", [float(row[0]), float(row[1]), float(row[2])],
                         0.0, 1.0, " ", "O", i, "O"))
        water_chain.add(residue)
    model.add(water_chain)

    io = MMCIFIO() if fmt.lower() == "cif" else PDBIO()
    io.set_structure(structure)
    io.save(out_path)
    print(f"Successfully saved protein + {len(coords)} waters to: {out_path}")


def cif_to_pdb(cif_path, pdb_path):
    """Convert a protein .cif/.mmcif to .pdb via Biopython (MMCIFParser -> PDBIO)."""
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB import PDBIO

    structure = MMCIFParser(QUIET=True).get_structure("input", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def to_input_pdb(src_path, dst_pdb_path):
    """Materialize ``src_path`` (.pdb/.cif/.mmcif) as a PDB at ``dst_pdb_path``."""
    ext = os.path.splitext(src_path)[1].lower()
    if ext == ".pdb":
        shutil.copy(src_path, dst_pdb_path)
    elif ext in (".cif", ".mmcif"):
        cif_to_pdb(src_path, dst_pdb_path)
    else:
        raise ValueError(f"Unsupported structure format: {ext!r} "
                         f"(supported: {', '.join(SUPPORTED_STRUCTURE_EXTS)})")
