from __future__ import annotations
from collections import OrderedDict
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from itertools import product

from .logging import logger

if TYPE_CHECKING:
    from mdevaluate.coordinates import CoordinateFrame


def pbc_diff(
    coords_a: NDArray, coords_b: NDArray, box: Optional[NDArray] = None
) -> NDArray:
    if box is None:
        out = coords_a - coords_b
    elif len(getattr(box, "shape", [])) == 1:
        out = pbc_diff_rect(coords_a, coords_b, box)
    elif len(getattr(box, "shape", [])) == 2:
        out = pbc_diff_tric(coords_a, coords_b, box)
    else:
        raise NotImplementedError("cannot handle box")
    return out


def pbc_diff_rect(coords_a: NDArray, coords_b: NDArray, box: NDArray) -> NDArray:
    """
    Calculate the difference of two vectors, considering periodic boundary conditions.
    """
    v = coords_a - coords_b
    s = v / box
    v = box * (s - np.round(s))
    return v


def pbc_diff_tric(coords_a: NDArray, coords_b: NDArray, box: NDArray) -> NDArray:
    """
    Difference vector for arbitrary pbc

        Args:
        box_matrix: CoordinateFrame.box
    """
    if len(box.shape) == 1:
        box = np.diag(box)
    if coords_a.shape == (3,):
        coords_a = coords_a.reshape((1, 3))  # quick 'n dirty
    if coords_b.shape == (3,):
        coords_b = coords_b.reshape((1, 3))
    if box is not None:
        r3 = np.subtract(coords_a, coords_b)
        r2 = np.subtract(
            r3,
            (np.rint(np.divide(r3[:, 2], box[2][2])))[:, np.newaxis]
            * box[2][np.newaxis, :],
        )
        r1 = np.subtract(
            r2,
            (np.rint(np.divide(r2[:, 1], box[1][1])))[:, np.newaxis]
            * box[1][np.newaxis, :],
        )
        v = np.subtract(
            r1,
            (np.rint(np.divide(r1[:, 0], box[0][0])))[:, np.newaxis]
            * box[0][np.newaxis, :],
        )
    else:
        v = coords_a - coords_b
    return v


def pbc_dist(
    atoms_a: NDArray, atoms_b: NDArray, box: Optional[NDArray] = None
) -> ArrayLike:
    return ((pbc_diff(atoms_a, atoms_b, box) ** 2).sum(axis=1)) ** 0.5


def pbc_backfold_compact(act_frame: NDArray, box_matrix: NDArray) -> NDArray:
    """
    mimics "trjconv ... -pbc atom -ur compact"

    folds coords of act_frame in wigner-seitz-cell (e.g. dodecahedron)
    """
    c = act_frame
    box = box_matrix
    ctr = box.sum(0) / 2
    c = np.asarray(c)
    shape = c.shape
    if shape == (3,):
        c = c.reshape((1, 3))
        shape = (1, 3)  # quick 'n dirty
    comb = np.array(
        [np.asarray(i) for i in product([0, -1, 1], [0, -1, 1], [0, -1, 1])]
    )
    b_matrices = comb[:, :, np.newaxis] * box[np.newaxis, :, :]
    b_vectors = b_matrices.sum(axis=1)[np.newaxis, :, :]
    sc = c[:, np.newaxis, :] + b_vectors
    w = np.argsort(((sc - ctr) ** 2).sum(2), 1)[:, 0]
    return sc[range(shape[0]), w]


def whole(frame: CoordinateFrame) -> CoordinateFrame:
    """
    Apply ``-pbc whole`` to a CoordinateFrame.
    """
    residue_ids = frame.coordinates.atom_subset.residue_ids
    box = frame.box.diagonal()

    # make sure, residue_ids are sorted, then determine indices at which the res id changes
    # kind='stable' assures that any existent ordering is preserved
    logger.debug("Using first atom as reference for whole.")
    sort_ind = residue_ids.argsort(kind="stable")
    i = np.concatenate([[0], np.where(np.diff(residue_ids[sort_ind]) > 0)[0] + 1])
    coms = frame[sort_ind[i]][residue_ids - 1]

    cor = np.zeros_like(frame)
    cd = frame - coms
    n, d = np.where(cd > box / 2 * 0.9)
    cor[n, d] = -box[d]
    n, d = np.where(cd < -box / 2 * 0.9)
    cor[n, d] = box[d]

    return frame + cor


NOJUMP_CACHESIZE = 128


def nojump(frame: CoordinateFrame, usecache: bool = True) -> CoordinateFrame:
    """
    Return the nojump coordinates of a frame, based on a jump matrix.
    """
    selection = frame.selection
    reader = frame.coordinates.frames
    if usecache:
        if not hasattr(reader, "_nojump_cache"):
            reader._nojump_cache = OrderedDict()
        # make sure to use absolute (non negative) index
        abstep = frame.step % len(frame.coordinates)
        i0s = [x for x in reader._nojump_cache if x <= abstep]
        if len(i0s) > 0:
            i0 = max(i0s)
            delta = reader._nojump_cache[i0]
            i0 += 1
        else:
            i0 = 0
            delta = 0

        delta = (
            delta
            + np.array(
                np.vstack(
                    [m[i0 : abstep + 1].sum(axis=0) for m in reader.nojump_matrices]
                ).T
            )
            @ frame.box
        )

        reader._nojump_cache[abstep] = delta
        while len(reader._nojump_cache) > NOJUMP_CACHESIZE:
            reader._nojump_cache.popitem(last=False)
        delta = delta[selection, :]
    else:
        delta = (
            np.array(
                np.vstack(
                    [
                        m[: frame.step + 1, selection].sum(axis=0)
                        for m in reader.nojump_matrices
                    ]
                ).T
            )
            @ frame.box
        )
    return frame - delta


def pbc_points(
    coordinates: ArrayLike,
    box: Optional[NDArray] = None,
    thickness: Optional[float] = None,
    index: bool = False,
    shear: bool = False,
) -> Union[NDArray, tuple[NDArray, NDArray]]:
    """
    Returns the points their first periodic images. Does not fold
    them back into the box.
    Thickness 0 means all 27 boxes. Positive means the box+thickness.
    Negative values mean that less than the box is returned.
    index=True also returns the indices with indices of images being their
    original values.
    """
    if box is None:
        box = coordinates.box
    if shear:
        box[2, 0] = box[2, 0] % box[0, 0]
        # Shifts the box images in the other directions if they moved more than
        # half the box length
        if box[2, 0] > box[2, 2] / 2:
            box[2, 0] = box[2, 0] - box[0, 0]

    grid = np.array(
        [[i, j, k] for k in [-1, 0, 1] for j in [1, 0, -1] for i in [-1, 0, 1]]
    )
    indices = np.tile(np.arange(len(coordinates)), 27)
    coordinates_pbc = np.concatenate([coordinates + v @ box for v in grid], axis=0)
    size = np.diag(box)

    if thickness is not None:
        mask = np.all(coordinates_pbc > -thickness, axis=1)
        coordinates_pbc = coordinates_pbc[mask]
        indices = indices[mask]
        mask = np.all(coordinates_pbc < size + thickness, axis=1)
        coordinates_pbc = coordinates_pbc[mask]
        indices = indices[mask]
    if index:
        return coordinates_pbc, indices
    return coordinates_pbc
