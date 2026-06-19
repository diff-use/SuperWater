"""Per-complex structural quality filters for crystallographic water datasets.

Ported from the WaterFlow project (``local_flow/src/dataset.py``) and re-expressed against
numpy/scipy (the originals used torch/biotite). Unlike the per-water filters in
``water_quality.py`` — which drop individual waters — these reject an entire complex:

1. Multi-chain interface  -- a complex with >=2 protein chains whose closest inter-chain
   approach exceeds ``interface_dist_threshold`` is likely separate asymmetric-unit (ASU)
   copies rather than a true interface.
2. CoM distance           -- protein and water centers of mass farther apart than
   ``max_com_dist`` indicate atoms in different reference frames (parse/placement errors).
3. Water clash            -- too large a fraction of waters sitting within ``clash_dist`` of
   a protein atom signals clashing/poorly-placed solvent.
4. Water/residue ratio    -- too few waters per protein residue signals sparse/under-modeled
   solvent.

Each check returns ``(is_valid: bool, reason: str)``; ``reason`` is empty when valid.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial.distance import cdist


def check_chain_interactions(
    chain_coords: dict[str, np.ndarray],
    interface_dist_threshold: float = 4.0,
) -> tuple[bool, str]:
    """Reject multi-chain complexes whose chains are non-interacting (ASU copies).

    ``chain_coords`` maps chain id -> ``(N_i, 3)`` protein atom coordinates. For >=2 chains
    the minimum pairwise inter-chain distance must be within ``interface_dist_threshold``;
    single-chain complexes always pass.
    """
    chain_ids = [cid for cid, c in chain_coords.items() if len(c) > 0]
    if len(chain_ids) < 2:
        return True, ""

    min_interface_dist = float("inf")
    for a, b in itertools.combinations(chain_ids, 2):
        d = cdist(chain_coords[a], chain_coords[b]).min()
        if d < min_interface_dist:
            min_interface_dist = d

    if min_interface_dist > interface_dist_threshold:
        return False, (
            f"multi-chain ({len(chain_ids)} chains) min interface distance "
            f"{min_interface_dist:.1f}A > {interface_dist_threshold}A (likely ASU copies)"
        )
    return True, ""


def check_com_distance(
    protein_coords: np.ndarray,
    water_coords: np.ndarray,
    max_com_dist: float = 25.0,
) -> tuple[bool, str]:
    """Reject if protein and water centers of mass are farther apart than ``max_com_dist``."""
    if len(water_coords) == 0 or len(protein_coords) == 0:
        return True, ""
    com_dist = float(np.linalg.norm(protein_coords.mean(axis=0) - water_coords.mean(axis=0)))
    if com_dist > max_com_dist:
        return False, f"CoM distance {com_dist:.1f}A exceeds threshold {max_com_dist}A"
    return True, ""


def check_water_clashes(
    protein_coords: np.ndarray,
    water_coords: np.ndarray,
    clash_dist: float = 2.0,
    max_clash_fraction: float = 0.05,
) -> tuple[bool, str]:
    """Reject if more than ``max_clash_fraction`` of waters lie within ``clash_dist`` of protein."""
    if len(water_coords) == 0 or len(protein_coords) == 0:
        return True, ""
    min_dists = cdist(water_coords, protein_coords).min(axis=1)
    n_clashing = int((min_dists < clash_dist).sum())
    clash_fraction = n_clashing / len(water_coords)
    if clash_fraction > max_clash_fraction:
        return False, (
            f"water clash fraction {clash_fraction:.1%} ({n_clashing}/{len(water_coords)}) "
            f"exceeds threshold {max_clash_fraction:.0%}"
        )
    return True, ""


def check_water_residue_ratio(
    num_waters: int,
    num_residues: int,
    min_ratio: float = 0.1,
) -> tuple[bool, str]:
    """Reject if the water/residue ratio is below ``min_ratio`` (or there are no residues)."""
    if num_residues == 0:
        return False, "no protein residues"
    ratio = num_waters / num_residues
    if ratio < min_ratio:
        return False, (
            f"water/residue ratio {ratio:.2f} ({num_waters}/{num_residues}) "
            f"below threshold {min_ratio}"
        )
    return True, ""
