"""Pickleable functions for process pooling and/or jit compiling."""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


def rdf_distances(
    set_a: np.ndarray,
    set_b: np.ndarray,
    box: np.ndarray,
    mode: str,
    nbins: int,
    rmin: float,
    rmax: float,
    **kwargs,
):
    """Compute distances between two sets of coordinates."""

    def __flatten_pairs(pairs, mode):
        I = []
        J = []
        for i, js in enumerate(pairs):
            for j in js:
                if mode == "inter" and i == j:
                    continue
                if mode == "intra" and i != j:
                    continue
                else:
                    I.append(i)
                    J.append(j)

        return np.array(I, dtype=np.int32), np.array(J, dtype=np.int32)

    @njit
    def __pbc_distances(I, J, set_a, set_b, box):
        n = len(I)
        out = np.empty(n, dtype=np.float32)

        for k in range(n):
            i = I[k]
            j = J[k]

            dx = set_b[j, 0] - set_a[i, 0]
            dy = set_b[j, 1] - set_a[i, 1]
            dz = set_b[j, 2] - set_a[i, 2]

            dx -= np.round(dx / box[0]) * box[0]
            dy -= np.round(dy / box[1]) * box[1]
            dz -= np.round(dz / box[2]) * box[2]

            out[k] = np.sqrt(dx * dx + dy * dy + dz * dz)

        return out

    if box.shape == (3, 3):
        box = box.diagonal()

    set_a = set_a % box
    set_b = set_b % box

    tree_a = cKDTree(set_a, boxsize=box)
    tree_b = cKDTree(set_b, boxsize=box)

    pairs = tree_a.query_ball_tree(tree_b, rmax)
    I, J = __flatten_pairs(pairs, mode)

    dists = __pbc_distances(I, J, set_a, set_b, box)

    hist, bins = np.histogram(dists, nbins, range=(rmin, rmax))
    bin_centers = (bins[1:] + bins[:-1]) / 2

    vol = np.prod(box)
    rho = len(set_b) / vol
    shell_vol = 4 * np.pi * bin_centers**2 * (bins[1] - bins[0])

    ideal_counts = rho * shell_vol * len(set_a)

    return hist / ideal_counts
