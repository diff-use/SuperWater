"""Build a tiny SuperWater dataset (pdb-only, no mol2) from raw *_final.pdb files.

For each id writes:
  <out>/<id>/<id>_protein_processed.pdb   (protein ATOM records only)
  <out>/<id>/<id>_water.pdb               (crystallographic water oxygens)
"""
import os
import sys

RAW = "/mnt/diffuse-shared/migrated/diffuse-public/water_data/pdb_data"


def build(out_dir, ids):
    for pid in ids:
        src = os.path.join(RAW, pid, f"{pid}_final.pdb")
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
        dest = os.path.join(out_dir, pid)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, f"{pid}_protein_processed.pdb"), "w") as f:
            f.write("\n".join(protein) + "\nEND\n")
        with open(os.path.join(dest, f"{pid}_water.pdb"), "w") as f:
            f.write("\n".join(waters) + "\nEND\n")
        print(f"{pid}: {len(protein)} protein atoms, {len(waters)} waters")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2:])
