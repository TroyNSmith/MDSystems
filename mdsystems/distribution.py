"""Analytical functions pertaining to molecular distribution."""

import numpy as np


def radial_distribution(
    xyz: np.ndarray,
    ref: np.ndarray,
    unit_cell: np.ndarray,
    distinct: bool = True,
    cutoff: float = 1.5,
    nbins: int = 1000
):
    """
    Compute radial distribution function (RDF) between xyz and ref with PBC using CPU + Numba.

    Parameters
    ----------
    xyz : (N,3) array
        Coordinates of particles.
    ref : (M,3) array
        Reference coordinates.
    unit_cell : (3,3) array
        Unit cell vectors for PBC.
    xp : module
        np (for compatibility, Numba only works with CPU arrays)
    distinct : bool
        If True, ignore self-pairs (xyz == ref).
    cutoff : float
        Maximum distance to consider.
    nbins : int
        Number of histogram bins.

    Returns
    -------
    hist : (nbins,) array
        Histogram counts of distances.
    bin_centers : (nbins,) array
        Centers of the bins.
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    unit_cell = np.asarray(unit_cell, dtype=np.float32)

    # Convert to fractional coordinates for PBC
    inv_uc = np.linalg.inv(unit_cell).astype(np.float32)
    xyz_frac = xyz @ inv_uc
    ref_frac = ref @ inv_uc

    hist = np.zeros(nbins, dtype=np.int64)
    bin_width = cutoff / nbins

    _compute_rdf_loop(xyz_frac, ref_frac, unit_cell.astype(np.float32),
                      hist, bin_width, cutoff, distinct)

    bin_centers = (np.arange(nbins, dtype=np.float32) + 0.5) * bin_width
    if not distinct:
        hist = hist / 2
    return hist, bin_centers