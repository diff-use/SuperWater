"""Prepare the 62-PDB benchmark dataset from the shared PDB-REDO mount.

For each ID in the test split, read <mount>/<id>/<id>_final.pdb and write the SuperWater
dataset layout expected by the PDBBind loader:
    data/bench62/<id>/<id>_protein_processed.pdb   (protein ATOM records, non-HOH)
    data/bench62/<id>/<id>_water.pdb               (crystallographic water oxygens)
Also writes data/bench62/bench62_ids.txt (lowercased IDs).

Splitting logic mirrors scripts/build_smoke_dataset.py; only the source path differs.
"""
import os
import sys

MOUNT = "/mnt/diffuse-shared/migrated/diffuse-public/water_data/pdb_data"
SPLIT = os.path.expanduser("~/workspace/total_splits/random_original_superwater_intersection_test.txt")
OUT = "data/bench62"


def split_final_pdb(src):
    protein, waters, seen = [], [], set()
    with open(src) as f:
        for ln in f:
            rec, resn = ln[:6].strip(), ln[17:20].strip()
            if rec == "ATOM" and resn != "HOH":
                protein.append(ln.rstrip("\n"))
            elif rec in ("ATOM", "HETATM") and resn == "HOH":
                el = (ln[76:78].strip() or ln[12:16].strip()[:1]).upper()
                if el != "O":
                    continue
                key = (ln[21], ln[22:26], ln[26])  # chain, resseq, icode
                if key in seen:
                    continue
                seen.add(key)
                waters.append(ln.rstrip("\n"))
    return protein, waters


def main():
    with open(SPLIT) as f:
        ids = [ln.strip().lower() for ln in f if ln.strip()]
    os.makedirs(OUT, exist_ok=True)
    rows, ok = [], []
    for pid in ids:
        src = os.path.join(MOUNT, pid, f"{pid}_final.pdb")
        if not os.path.exists(src):
            print(f"MISSING source: {pid}")
            continue
        protein, waters = split_final_pdb(src)
        if not protein:
            print(f"WARN {pid}: no protein atoms, skipping")
            continue
        if not waters:
            print(f"WARN {pid}: no water oxygens")
        dest = os.path.join(OUT, pid)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, f"{pid}_protein_processed.pdb"), "w") as f:
            f.write("\n".join(protein) + "\nEND\n")
        with open(os.path.join(dest, f"{pid}_water.pdb"), "w") as f:
            f.write("\n".join(waters) + "\nEND\n")
        rows.append((pid, len(protein), len(waters)))
        ok.append(pid)

    with open(os.path.join(OUT, "bench62_ids.txt"), "w") as f:
        f.write("\n".join(ok) + "\n")

    wc = sorted(w for _, _, w in rows)
    print(f"\nPrepared {len(rows)}/{len(ids)} complexes -> {OUT}")
    if wc:
        n = len(wc)
        print(f"water counts: min={wc[0]} median={wc[n // 2]} max={wc[-1]}")
    zero = [p for p, _, w in rows if w == 0]
    if zero:
        print(f"complexes with ZERO waters: {zero}")


if __name__ == "__main__":
    sys.exit(main())
