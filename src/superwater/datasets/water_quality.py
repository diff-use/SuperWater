"""Per-water quality filtering for crystallographic waters.

Ported from the WaterFlow project (``local_flow/src/dataset.py``) and re-expressed
against the plain ``(key, value)`` lookups SuperWater builds from BioPython structures.

A water is removed if it fails ANY enabled criterion:

1. Distance-to-protein  -- water is farther than ``max_protein_dist`` from the nearest
   protein atom (``scipy.spatial.distance.cdist``).
2. EDIA score           -- electron-density reliability ``EDIAm < min_edia``, read from
   the PDB-REDO ``<id>_final.json`` sidecar.
3. B-factor z-score     -- per-structure normalized B-factor ``> max_bfactor_zscore``.

Waters missing from a lookup PASS that lookup's filter (conservative), so the EDIA and
B-factor filters degrade gracefully to no-ops when the data is unavailable.

The water key is ``(chain_id: str, res_id: int, ins_code: str)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

WaterKey = tuple[str, int, str]


def normalize_ins_code(value) -> str:
    """Normalize an insertion code into a stable key token.

    Handles the various representations of "no insertion code" across data sources
    (BioPython uses ``" "``; PDB-REDO JSON may use ``""``/``"?"``/``"."``/``None``).
    Returns an empty string for "no insertion code", otherwise the stripped value.
    """
    if value is None:
        return ""
    ins = str(value).strip()
    if ins in {"", "?", "."}:
        return ""
    return ins


def load_edia_for_pdb(json_path: str | Path) -> dict[WaterKey, float] | None:
    """Load per-water EDIA scores from a PDB-REDO ``<id>_final.json`` file.

    Returns a dict mapping ``(chain_id, res_id, ins_code) -> EDIAm`` for water residues,
    an empty dict if the file holds no waters, or ``None`` if the file is missing or
    cannot be parsed (so the caller can skip EDIA filtering entirely).
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)

        edia_lookup: dict[WaterKey, float] = {}
        for entry in data:
            if entry.get("compID") not in ("HOH", "WAT"):
                continue
            pdb_info = entry.get("pdb", {})
            chain_id = str(pdb_info.get("strandID", ""))
            res_id = int(pdb_info.get("seqNum", 0))
            ins_code = normalize_ins_code(pdb_info.get("insCode", ""))
            edia_lookup[(chain_id, res_id, ins_code)] = float(entry.get("EDIAm", 0.0))

        return edia_lookup

    except Exception as e:  # noqa: BLE001 -- a bad sidecar must not abort dataset prep
        print(f"WARNING: could not load EDIA JSON {json_path}: {e}")
        return None


def compute_normalized_bfactors(
    water_records: list[tuple[WaterKey, float]],
) -> dict[WaterKey, float] | None:
    """Z-score normalize water B-factors using water-only statistics.

    ``water_records`` is an iterable of ``(key, raw_b_factor)`` for water atoms (one per
    water). The z-score is ``(b - mean) / max(std, 1e-3)``; when every B-factor is
    identical (``std == 0``) all waters get the neutral z-score ``0.0``. Returns a dict
    ``key -> z-score`` (first occurrence of a key wins) or ``None`` when there are no
    waters.
    """
    if not water_records:
        return None

    raw = np.array([b for _, b in water_records], dtype=float)
    water_mean = float(raw.mean())
    water_std = float(raw.std())

    bfactor_lookup: dict[WaterKey, float] = {}
    for key, b in water_records:
        if key in bfactor_lookup:
            continue
        bfactor_lookup[key] = (b - water_mean) / max(water_std, 1e-3) if water_std > 0 else 0.0
    return bfactor_lookup


def apply_threshold_filter(
    water_keys: list[WaterKey],
    lookup: dict[WaterKey, float],
    threshold: float,
    fail_if_below: bool,
) -> np.ndarray:
    """Boolean mask where ``True`` means the water FAILS the threshold.

    ``fail_if_below=True`` fails when ``value < threshold`` (e.g. EDIA); ``False`` fails
    when ``value > threshold`` (e.g. B-factor). Waters missing from ``lookup`` get
    ``np.nan`` and therefore PASS (NaN comparisons are ``False``).
    """
    values = np.array([lookup.get(key, np.nan) for key in water_keys])
    if fail_if_below:
        return values < threshold
    return values > threshold


def filter_waters_by_quality(
    water_coords: np.ndarray,
    water_keys: list[WaterKey],
    protein_coords: np.ndarray | None,
    edia_lookup: dict[WaterKey, float] | None,
    bfactor_lookup: dict[WaterKey, float] | None,
    max_protein_dist: float = 5.0,
    min_edia: float = 0.4,
    max_bfactor_zscore: float = 1.5,
    cache_key: str | None = None,
) -> np.ndarray:
    """Return a boolean keep-mask over waters (``True`` = keep).

    A water is kept only if it passes every enabled filter. Each filter is enabled by
    passing a non-None argument (``protein_coords``/``edia_lookup``/``bfactor_lookup``).
    Pass ``cache_key`` (e.g. the PDB id) to print a one-line per-structure summary.
    """
    n_waters = len(water_keys)
    if n_waters == 0:
        return np.array([], dtype=bool)

    stats = {"total": n_waters, "removed_distance": 0, "removed_edia": 0, "removed_bfactor": 0}

    # 1. distance-to-protein
    dist_fail = np.zeros(n_waters, dtype=bool)
    if protein_coords is not None and len(protein_coords):
        min_dists = cdist(water_coords, protein_coords).min(axis=1)
        dist_fail = min_dists > max_protein_dist
        stats["removed_distance"] = int(dist_fail.sum())

    # 2 + 3. lookup-based filters: (lookup, threshold, fail_if_below, stat_key)
    lookup_filters = [
        (edia_lookup, min_edia, True, "edia"),
        (bfactor_lookup, max_bfactor_zscore, False, "bfactor"),
    ]
    lookup_fail = np.zeros(n_waters, dtype=bool)
    for lookup, threshold, fail_if_below, name in lookup_filters:
        if lookup is not None:
            fail_mask = apply_threshold_filter(water_keys, lookup, threshold, fail_if_below)
            stats[f"removed_{name}"] = int(fail_mask.sum())
            lookup_fail |= fail_mask

    keep_mask = ~(dist_fail | lookup_fail)
    stats["kept"] = int(keep_mask.sum())

    if cache_key is not None:
        removed = stats["total"] - stats["kept"]
        if removed > 0:
            print(
                f"  {cache_key}: filtered {removed}/{stats['total']} waters "
                f"(dist:{stats['removed_distance']}, "
                f"edia:{stats['removed_edia']}, bfactor:{stats['removed_bfactor']})"
            )

    return keep_mask
