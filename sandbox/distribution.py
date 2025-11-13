"""Analytical functions pertaining to molecular distribution."""

import numpy as np


def radial_distribution(
    xyz: np.ndarray,
    ref: np.ndarray,
    unit_cell: np.ndarray,
    distinct: bool = True,
    cutoff: float = 1.5,
) -> np.ndarray:
    """
    Compute pairwise radial distances between `ref` and `xyz` with PBC.

    :param xyz: Coordinates of particles, shape (N,3).
    :param ref: Coordinates of reference particles, shape (M,3).
    :param unit_cell: Unit cell vectors for periodic boundary conditions, shape (3,3).
    :param distinct: If True, ignore self-pairs when xyz and ref are identical.
    :param cutoff: Maximum distance to consider.

    Returns
    -------
    distances : np.ndarray
        Array of pairwise distances (up to cutoff).
    """
    xyz = np.asarray(xyz)
    ref = np.asarray(ref)
    unit_cell = np.asarray(unit_cell)

    inv_uc = np.linalg.inv(unit_cell)
    xyz_frac = xyz @ inv_uc
    ref_frac = ref @ inv_uc

    diff = xyz_frac[:, None, :] - ref_frac[None, :, :]  # shape (N, M, 3)

    diff -= np.round(diff)

    diff_cart = diff @ unit_cell
    distances = np.linalg.norm(diff_cart, axis=-1)

    mask = distances <= cutoff
    distances = distances[mask]
    distances = distances[distances > 1e-8]  # ignore zero distances

    hist, bins = np.histogram(distances, bins=1000, range=(0.0, cutoff))
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if not distinct:
        hist = hist / 2

    return hist, bin_centers
