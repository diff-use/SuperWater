"""Tests for the read-only dataset audit script."""
import importlib.util
import sys
from pathlib import Path

import pytest

_PROTEIN = (
    "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)
_WATER = "HETATM    1  O   HOH W   1       3.000   0.000   0.000  1.00  0.00           O\nEND\n"


def _load_audit():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_dataset.py"
    spec = importlib.util.spec_from_file_location("audit_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_complex(root: Path, cid: str, protein=True, water=True, water_text=_WATER, ext="cif"):
    d = root / cid
    d.mkdir(parents=True)
    # Write minimal PDB content under the requested extension; the audit only needs the
    # parser to read it, and BioPython's CIF/PDB choice keys off the file extension.
    if protein:
        (d / f"{cid}_protein_processed.{'pdb' if ext == 'pdb' else 'cif'}").write_text(_PROTEIN)
    if water:
        (d / f"{cid}_water.{'pdb' if ext == 'pdb' else 'cif'}").write_text(water_text)


def test_audit_flags_problems(tmp_path):
    audit = _load_audit()
    data = tmp_path / "data"
    # 1) healthy   2) missing water   3) zero-oxygen water
    _make_complex(data, "good", ext="pdb")
    _make_complex(data, "nowater", water=False, ext="pdb")
    _make_complex(data, "emptywater", water_text="END\n", ext="pdb")

    results = [audit.audit_one((str(data), cid, None))
               for cid in ("good", "nowater", "emptywater")]
    by_id = {cid: (errs, warns) for cid, errs, warns, _ in results}

    assert by_id["good"][0] == []  # no errors
    assert "legacy_pdb_protein" in by_id["good"][1]  # .pdb protein -> warning
    assert "missing_water" in by_id["nowater"][0]
    assert "no_waters" in by_id["emptywater"][0]


def test_audit_missing_embedding(tmp_path):
    audit = _load_audit()
    data = tmp_path / "data"
    emb = tmp_path / "emb"
    emb.mkdir()
    _make_complex(data, "x", ext="pdb")
    (emb / "x_chain_0.pt").write_bytes(b"")  # present -> no flag

    _, errors, _, _ = audit.audit_one((str(data), "x", str(emb)))
    assert "missing_embedding" not in errors

    _make_complex(data, "y", ext="pdb")  # no embedding written
    _, errors_y, _, _ = audit.audit_one((str(data), "y", str(emb)))
    assert "missing_embedding" in errors_y


def test_audit_main_exit_code_and_report(tmp_path):
    audit = _load_audit()
    data = tmp_path / "data"
    _make_complex(data, "good", ext="pdb")
    _make_complex(data, "bad", water=False, ext="pdb")

    with pytest.raises(SystemExit) as exc:
        audit.main(["--data_dir", str(data), "--num_workers", "1"])
    assert exc.value.code == 1  # an error complex -> non-zero exit

    report = (data / "logs" / "audit_report.tsv").read_text()
    assert "bad" in report and "missing_water" in report
    # "good" has only a legacy_pdb warning here, so it appears with an empty errors column.
    good_line = next(ln for ln in report.splitlines() if ln.startswith("good\t"))
    assert good_line.split("\t")[1] == ""  # no errors


def test_delete_broken_lets_rerun_reheal(tmp_path):
    audit = _load_audit()
    data = tmp_path / "data"
    _make_complex(data, "bad", water=False, ext="pdb")

    with pytest.raises(SystemExit):
        audit.main(["--data_dir", str(data), "--num_workers", "1", "--delete_broken"])

    # The broken complex's protein file is removed, so setup --skip_existing would redo it.
    assert not list((data / "bad").glob("bad_protein_processed.*"))
