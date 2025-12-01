"""Analytical functions pertaining to molecular distribution."""

import numpy as np
from scipy.spatial.distance import cdist

from typing import Any


def radial_distribution(
    xyz: np.ndarray,
    ref: np.ndarray,
    unitcell: np.ndarray,
    volume: float,
    distinct: bool = True,
    cutoff: float = 1.5,
    nbins: int = 1000,
    mode: str = "total",
    **kwargs,
):
    """
    Compute the radial distribution function (RDF) g(r) for a set of coordinates.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Coordinates of the reference set (or set A).
    ref : np.ndarray, shape (M, 3)
        Coordinates of the comparison set (or set B). Can be the same as xyz.
    unitcell : np.ndarray, shape (3,) or (3,3)
        Box lengths along each axis for PBC corrections.
    volume : float
        Volume of the system (used for normalization).
    distinct : bool, default True
        If False and xyz is ref, ignores the lower triangle of the distance matrix
        to avoid double counting.
    cutoff : float, default 1.5
        Maximum distance for RDF calculation.
    nbins : int, default 1000
        Number of bins for the histogram.
    mode : str, default "total"
        Which distances to include:
        - "total" : all distances
        - "intra" : only diagonal distances (self-distances, usually 0)
        - "inter" : only off-diagonal distances (excludes self-distances)
    **kwargs : dict
        Additional arguments (currently unused).

    Returns
    -------
    rdf : np.ndarray, shape (nbins,)
        Radial distribution function values.
    bin_centers : np.ndarray, shape (nbins,)
        Bin centers corresponding to the RDF values.
    """
    box = np.asarray(unitcell).diagonal() if unitcell.ndim == 2 else np.asarray(unitcell)

    dmat = pairwise_distances_pbc(xyz, ref, box)

    if not distinct and np.all(xyz is ref):
        i_upper = np.triu_indices(len(xyz), k=1)
        dists = dmat[i_upper]
    elif mode == "total":
        dists = dmat.ravel()
    elif mode == "intra":
        dists = np.diag(dmat)
    elif mode == "inter":
        dists = dmat[~np.eye(dmat.shape[0], dtype=bool)]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'total', 'intra', or 'inter'.")

    hist, bins = np.histogram(dists, bins=nbins, range=(0.01, cutoff))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    dr = bins[1] - bins[0]
    ideal_counts = (len(xyz) / volume) * 4 * np.pi * bin_centers**2 * dr

    rdf = hist / ideal_counts
    return rdf, bin_centers

def pairwise_distances_pbc(
    x: np.ndarray[Any, np.float32],
    y: np.ndarray[Any, np.float32],
    box: np.ndarray[Any, np.float32],
) -> np.ndarray:
    """
    Compute all pairwise Euclidean distances between points in X and Y
    under periodic boundary conditions (PBC) using the minimum-image convention.

    Parameters
    ----------
    X : np.ndarray, shape (N, 3)
        Coordinates of the first set of points.
    Y : np.ndarray, shape (M, 3)
        Coordinates of the second set of points.
    box : float or np.ndarray, shape (3,)
        Box dimensions along each axis.

    Returns
    -------
    dmat : np.ndarray, shape (N, M)
        Euclidean distance matrix considering PBC.
    """  # noqa: D205
    delta = x[:, None, :] - y[None, :, :]
    # If delta / box > 0.5, offset the distance vectors by box.
    delta -= box * np.round(delta / box)

    dmat = np.linalg.norm(delta, axis=-1)

    return dmat
