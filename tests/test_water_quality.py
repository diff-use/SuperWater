"""Tests for per-water quality filtering (ported from WaterFlow) and its wiring into
the dataset-prep script.

Unit tests exercise ``filter_waters_by_quality`` directly; the prep-level test runs the
real ``extract_waters`` + ``quality_filtered_coords`` path from
``scripts/setup_custom_dataset.py`` against a synthetic structure.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from superwater.datasets.water_quality import (
    compute_normalized_bfactors,
    filter_waters_by_quality,
    normalize_ins_code,
)

# ---------------------------------------------------------------------------
# Unit tests for filter_waters_by_quality
# ---------------------------------------------------------------------------

WATER_COORDS = np.array(
    [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [15.0, 0.0, 0.0], [3.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
)
WATER_KEYS = [("A", 101, ""), ("A", 102, ""), ("A", 103, ""), ("B", 201, ""), ("B", 202, "")]
PROTEIN_COORDS = np.zeros((10, 3))  # protein centered at the origin


def test_distance_filtering():
    keep = filter_waters_by_quality(
        WATER_COORDS, WATER_KEYS, PROTEIN_COORDS, None, None, max_protein_dist=6.0
    )
    assert keep.sum() == 3  # 0, 5, 3 A pass; 15, 20 A fail


def test_edia_filtering():
    edia = {k: v for k, v in zip(WATER_KEYS, [0.85, 0.30, 0.50, 0.20, 0.60])}
    keep = filter_waters_by_quality(WATER_COORDS, WATER_KEYS, None, edia, None, min_edia=0.4)
    assert keep.sum() == 3


def test_bfactor_filtering():
    bf = {k: v for k, v in zip(WATER_KEYS, [1.0, 6.0, 2.5, 7.0, 0.5])}
    keep = filter_waters_by_quality(
        WATER_COORDS, WATER_KEYS, None, None, bf, max_bfactor_zscore=5.0
    )
    assert keep.sum() == 3


def test_combined_filters():
    edia = {k: v for k, v in zip(WATER_KEYS, [0.85, 0.85, 0.85, 0.30, 0.85])}
    bf = {k: v for k, v in zip(WATER_KEYS, [1.0, 6.0, 1.0, 1.0, 1.0])}
    keep = filter_waters_by_quality(
        WATER_COORDS, WATER_KEYS, PROTEIN_COORDS, edia, bf,
        max_protein_dist=6.0, min_edia=0.4, max_bfactor_zscore=5.0,
    )
    assert keep.sum() == 1  # only (A,101) passes all three


def test_missing_edia_data_keeps_water():
    edia = {("A", 101, ""): 0.85, ("A", 102, ""): 0.30}  # rest absent
    keep = filter_waters_by_quality(WATER_COORDS, WATER_KEYS, None, edia, None, min_edia=0.4)
    assert keep.sum() == 4  # 1 good + 3 with no EDIA data (conservative keep)


def test_empty_water_array():
    keep = filter_waters_by_quality(np.zeros((0, 3)), [], PROTEIN_COORDS, None, None)
    assert len(keep) == 0


def test_all_filters_disabled():
    keep = filter_waters_by_quality(WATER_COORDS, WATER_KEYS, None, None, None)
    assert keep.sum() == 5


def test_compute_normalized_bfactors():
    lookup = compute_normalized_bfactors(
        [(("A", 1, ""), 20.0), (("A", 2, ""), 20.0), (("A", 3, ""), 80.0)]
    )
    assert lookup[("A", 3, "")] > lookup[("A", 1, "")]
    assert compute_normalized_bfactors([]) is None
    # constant B-factors -> neutral z-score, no division blow-up
    flat = compute_normalized_bfactors([(("A", 1, ""), 30.0), (("A", 2, ""), 30.0)])
    assert flat[("A", 1, "")] == 0.0


def test_normalize_ins_code():
    assert normalize_ins_code(" ") == ""
    assert normalize_ins_code("?") == ""
    assert normalize_ins_code(None) == ""
    assert normalize_ins_code("A") == "A"


# ---------------------------------------------------------------------------
# Prep-level test: real extract_waters + quality_filtered_coords
# ---------------------------------------------------------------------------


def _load_setup_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "setup_custom_dataset.py"
    spec = importlib.util.spec_from_file_location("setup_custom_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so the module's dataclass can resolve annotations
    spec.loader.exec_module(mod)
    return mod


def _atom(serial, name, resname, chain, resseq, x, y, z, bfac, hetatm=False):
    rec = "HETATM" if hetatm else "ATOM  "
    elem = name.strip()[0]
    return (f"{rec}{serial:>5} {name:^4} {resname:>3} {chain}{resseq:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfac:6.2f}          {elem:>2}\n")


def test_prep_level_distance_and_bfactor_filter(tmp_path):
    setup = _load_setup_module()

    pdb = tmp_path / "x.pdb"
    pdb.write_text(
        _atom(1, " N  ", "ALA", "A", 1, 0.0, 0.0, 0.0, 0.0)
        + _atom(2, " CA ", "ALA", "A", 1, 1.0, 0.0, 0.0, 0.0)
        + _atom(3, " C  ", "ALA", "A", 1, 2.0, 0.0, 0.0, 0.0)
        + _atom(4, " O  ", "HOH", "W", 1, 3.0, 0.0, 0.0, 20.0, hetatm=True)   # keep
        + _atom(5, " O  ", "HOH", "W", 2, 30.0, 0.0, 0.0, 20.0, hetatm=True)  # far -> drop
        + _atom(6, " O  ", "HOH", "W", 3, 3.0, 3.0, 0.0, 80.0, hetatm=True)   # high B -> drop
        + "END\n"
    )
    from superwater.structure_io import parse_structure
    structure = parse_structure(str(pdb), permissive=True)

    records = setup.extract_waters(structure, min_water_dist=1.9)
    assert len(records) == 3

    cfg = {
        "filter_by_distance": True, "filter_by_edia": True, "filter_by_bfactor": True,
        "max_protein_dist": 5.0, "min_edia": 0.4, "max_bfactor_zscore": 1.0,
    }
    kept = setup.quality_filtered_coords(records, structure, pdb, cfg, "x")
    assert len(kept) == 1  # far water + high-B water dropped; EDIA json absent -> no-op
    assert kept[0][0] == pytest.approx(3.0)

    # All filters disabled -> all waters survive (only min_water_dist dedup applies).
    cfg_off = {**cfg, "filter_by_distance": False, "filter_by_edia": False,
               "filter_by_bfactor": False}
    assert len(setup.quality_filtered_coords(records, structure, pdb, cfg_off, "x")) == 3
