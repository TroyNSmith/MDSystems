from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import legendre, jn
import dask.array as darray
from functools import partial
from scipy.spatial import KDTree

from .autosave import autosave_data
from .utils import coherent_sum
from .pbc import pbc_diff, pbc_points
from .coordinates import Coordinates, CoordinateFrame, displacements_without_drift


def log_indices(first: int, last: int, num: int = 100) -> np.ndarray:
    ls = np.logspace(0, np.log10(last - first + 1), num=num)
    return np.unique(np.int_(ls) - 1 + first)


@autosave_data(2)
def shifted_correlation(
    function: Callable,
    frames: Coordinates,
    selector: Optional[Callable] = None,
    segments: int = 10,
    skip: float = 0.1,
    window: float = 0.5,
    average: bool = True,
    points: int = 100,
) -> (np.ndarray, np.ndarray):
    """
    Calculate the time series for a correlation function.

    The times at which the correlation is calculated are determined by
    a logarithmic distribution.

    Args:
        function: The function that should be correlated
        frames: The coordinates of the simulation data
        selector (opt.):
                    A function that returns the indices depending on
                    the staring frame for which particles the
                    correlation should be calculated.
        segments (int, opt.):
                    The number of segments the time window will be
                    shifted
        skip (float, opt.):
                    The fraction of the trajectory that will be skipped
                    at the beginning, if this is None the start index
                    of the frames slice will be used, which defaults
                    to 0.1.
        window (float, opt.):
                    The fraction of the simulation the time series will
                    cover
        average (bool, opt.):
                    If True, returns averaged correlation function
        points (int, opt.):
                    The number of timeshifts for which the correlation
                    should be calculated
    Returns:
        tuple:
            A list of length N that contains the timeshiftes of the frames at which
            the time series was calculated and a numpy array of shape (segments, N)
            that holds the (non-avaraged) correlation data

    Example:
        Calculating the mean square displacement of a coordinate object
        named ``coords``:

        >>> time, data = shifted_correlation(msd, coords)
    """

    def get_correlation(
        frames: CoordinateFrame,
        start_frame: CoordinateFrame,
        index: np.ndarray,
        shifted_idx: np.ndarray,
    ) -> np.ndarray:
        if len(index) == 0:
            correlation = np.zeros(len(shifted_idx))
        else:
            start = frames[start_frame][index]
            correlation = np.array(
                [function(start, frames[frame][index]) for frame in shifted_idx]
            )
        return correlation

    def apply_selector(
        start_frame: CoordinateFrame,
        frames: CoordinateFrame,
        idx: np.ndarray,
        selector: Optional[Callable] = None,
    ):
        shifted_idx = idx + start_frame

        if selector is None:
            index = np.arange(len(frames[start_frame]))
            return get_correlation(frames, start_frame, index, shifted_idx)
        else:
            index = selector(frames[start_frame])
            if len(index) == 0:
                return np.zeros(len(shifted_idx))

            elif (
                isinstance(index[0], int)
                or isinstance(index[0], bool)
                or isinstance(index[0], np.integer)
                or isinstance(index[0], np.bool_)
            ):
                return get_correlation(frames, start_frame, index, shifted_idx)
            else:
                correlations = []
                for ind in index:
                    if len(ind) == 0:
                        correlations.append(np.zeros(len(shifted_idx)))

                    elif (
                        isinstance(ind[0], int)
                        or isinstance(ind[0], bool)
                        or isinstance(ind[0], np.integer)
                        or isinstance(ind[0], np.bool_)
                    ):
                        correlations.append(
                            get_correlation(frames, start_frame, ind, shifted_idx)
                        )
                    else:
                        raise ValueError(
                            "selector has more than two dimensions or does not "
                            "contain int or bool types"
                        )
                return correlations

    if 1 - skip < window:
        window = 1 - skip

    start_frames = np.unique(
        np.linspace(
            len(frames) * skip,
            len(frames) * (1 - window),
            num=segments,
            endpoint=False,
            dtype=int,
        )
    )

    num_frames = int(len(frames) * window)
    ls = np.logspace(0, np.log10(num_frames + 1), num=points)
    idx = np.unique(np.int_(ls) - 1)
    t = np.array([frames[i].time for i in idx]) - frames[0].time

    result = np.array(
        [
            apply_selector(start_frame, frames=frames, idx=idx, selector=selector)
            for start_frame in start_frames
        ]
    )

    if average:
        clean_result = []
        for entry in result:
            if np.all(entry == 0):
                continue
            else:
                clean_result.append(entry)
        result = np.array(clean_result)
        result = np.average(result, axis=0)
    return t, result


def msd(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    trajectory: Coordinates = None,
    axis: str = "all",
) -> float:
    """
    Mean square displacement
    """
    if trajectory is None:
        displacements = start_frame - end_frame
    else:
        displacements = displacements_without_drift(start_frame, end_frame, trajectory)
    if axis == "all":
        return (displacements**2).sum(axis=1).mean()
    elif axis == "xy" or axis == "yx":
        return (displacements[:, [0, 1]]**2).sum(axis=1).mean()
    elif axis == "xz" or axis == "zx":
        return (displacements[:, [0, 2]]**2).sum(axis=1).mean()
    elif axis == "yz" or axis == "zy":
        return (displacements[:, [1, 2]]**2).sum(axis=1).mean()
    elif axis == "x":
        return (displacements[:, 0] ** 2).mean()
    elif axis == "y":
        return (displacements[:, 1] ** 2).mean()
    elif axis == "z":
        return (displacements[:, 2] ** 2).mean()
    else:
        raise ValueError('Parameter axis has to be ether "all", "x", "y", or "z"!')


def isf(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    q: float = 22.7,
    trajectory: Coordinates = None,
    axis: str = "all",
) -> float:
    """
    Incoherent intermediate scattering function. To specify q, use
    water_isf = functools.partial(isf, q=22.77) # q has the value 22.77 nm^-1
    """
    if trajectory is None:
        displacements = start_frame - end_frame
    else:
        displacements = displacements_without_drift(start_frame, end_frame, trajectory)
    if axis == "all":
        distance = (displacements**2).sum(axis=1) ** 0.5
        return np.sinc(distance * q / np.pi).mean()
    elif axis == "xy" or axis == "yx":
        distance = (displacements[:, [0, 1]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean()
    elif axis == "xz" or axis == "zx":
        distance = (displacements[:, [0, 2]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean()
    elif axis == "yz" or axis == "zy":
        distance = (displacements[:, [1, 2]]**2).sum(axis=1) ** 0.5
        return np.real(jn(0, distance * q)).mean()
    elif axis == "x":
        distance = np.abs(displacements[:, 0])
        return np.mean(np.cos(np.abs(q * distance)))
    elif axis == "y":
        distance = np.abs(displacements[:, 1])
        return np.mean(np.cos(np.abs(q * distance)))
    elif axis == "z":
        distance = np.abs(displacements[:, 2])
        return np.mean(np.cos(np.abs(q * distance)))
    else:
        raise ValueError('Parameter axis has to be ether "all", "x", "y", or "z"!')


def rotational_autocorrelation(
    start_frame: CoordinateFrame, end_frame: CoordinateFrame, order: int = 2
) -> float:
    """
    Compute the rotational autocorrelation of the legendre polynomial for the
    given vectors.

    Args:
        start_frame, end_frame: CoordinateFrames of vectors
        order (opt.): Order of the legendre polynomial.

    Returns:
        Scalar value of the correlation function.
    """
    scalar_prod = (start_frame * end_frame).sum(axis=-1)
    poly = legendre(order)
    return poly(scalar_prod).mean()


def van_hove_self(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    bins: ArrayLike,
    trajectory: Coordinates = None,
    axis: str = "all",
) -> np.ndarray:
    r"""
    Compute the self-part of the Van Hove autocorrelation function.

    ..math::
      G(r, t) = \sum_i \delta(|\vec r_i(0) - \vec r_i(t)| - r)
    """
    if trajectory is None:
        vectors = start_frame - end_frame
    else:
        vectors = displacements_without_drift(start_frame, end_frame, trajectory)
    if axis == "all":
        delta_r = (vectors**2).sum(axis=1) ** 0.5
    elif axis == "xy" or axis == "yx":
        delta_r = (vectors[:, [0, 1]]**2).sum(axis=1) ** 0.5
    elif axis == "xz" or axis == "zx":
        delta_r = (vectors[:, [0, 2]]**2).sum(axis=1) ** 0.5
    elif axis == "yz" or axis == "zy":
        delta_r = (vectors[:, [1, 2]]**2).sum(axis=1) ** 0.5
    elif axis == "x":
        delta_r = np.abs(vectors[:, 0])
    elif axis == "y":
        delta_r = np.abs(vectors[:, 1])
    elif axis == "z":
        delta_r = np.abs(vectors[:, 2])
    else:
        raise ValueError('Parameter axis has to be ether "all", "x", "y", or "z"!')
    hist = np.histogram(delta_r, bins, range=(bins[0], bins[-1]))[0]
    hist = hist / (bins[1:] - bins[:-1])
    return hist / len(start_frame)


def van_hove_distinct(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    bins: ArrayLike,
    box: ArrayLike = None,
    use_dask: bool = True,
    comp: bool = False,
) -> np.ndarray:
    r"""
    Compute the distinct part of the Van Hove autocorrelation function.

    ..math::
      G(r, t) = \sum_{i, j} \delta(|\vec r_i(0) - \vec r_j(t)| - r)
    """
    if box is None:
        box = start_frame.box.diagonal()
    dimension = len(box)
    N = len(start_frame)
    if use_dask:
        start_frame = darray.from_array(start_frame, chunks=(500, dimension)).reshape(
            1, N, dimension
        )
        end_frame = darray.from_array(end_frame, chunks=(500, dimension)).reshape(
            N, 1, dimension
        )
        dist = (
            (pbc_diff(start_frame, end_frame, box) ** 2).sum(axis=-1) ** 0.5
        ).ravel()
        if np.diff(bins).std() < 1e6:
            dx = bins[0] - bins[1]
            hist = darray.bincount((dist // dx).astype(int), minlength=(len(bins) - 1))
        else:
            hist = darray.histogram(dist, bins=bins)[0]
        return hist.compute() / N
    else:
        if comp:
            dx = bins[1] - bins[0]
            minlength = len(bins) - 1

            def f(x):
                d = (pbc_diff(x, end_frame, box) ** 2).sum(axis=-1) ** 0.5
                return np.bincount((d // dx).astype(int), minlength=minlength)[
                    :minlength
                ]

            hist = sum(f(x) for x in start_frame)
        else:
            dist = (
                pbc_diff(
                    start_frame.reshape(1, -1, 3), end_frame.reshape(-1, 1, 3), box
                )
                ** 2
            ).sum(axis=-1) ** 0.5
            hist = np.histogram(dist, bins=bins)[0]
        return hist / N


def overlap(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    radius: float = 0.1,
    mode: str = "self",
) -> float:
    """
    Compute the overlap with a reference configuration defined in a CoordinatesTree.

    Args:
        start_frame: Initial frame, this is only used to get the frame index
        end_frame: The current configuration
        radius: The cutoff radius for the overlap
        mode: Select between "indifferent", "self" or "distict" part of the overlap

    This function is intended to be used with :func:`shifted_correlation`.
    As usual, the first two arguments are used internally, and the remaining ones
    should be defined with :func:`functools.partial`.

    If the overlap of a subset of the system should be calculated, this has to be
    defined through a selection of the reference configurations in the CoordinatesTree.

    Example:
        >>> shifted_correlation(
        ...     partial(overlap, crds_tree=CoordinatesTree(traj), radius=0.11),
        ...     traj
        ... )
    """
    start_PBC, index_PBC = pbc_points(
        start_frame, start_frame.box, index=True, thickness=2 * radius
    )
    start_tree = KDTree(start_PBC)
    dist, index_dist = start_tree.query(end_frame, 1, distance_upper_bound=radius)
    if mode == "indifferent":
        return np.sum(dist <= radius) / len(start_frame)
    index_dist = index_PBC[index_dist]
    index = np.arange(len(start_frame))
    index = index[dist <= radius]
    index_dist = index_dist[dist <= radius]
    if mode == "self":
        return np.sum(index == index_dist) / len(start_frame)
    elif mode == "distinct":
        return np.sum(index != index_dist) / len(start_frame)


def coherent_scattering_function(
    start_frame: CoordinateFrame, end_frame: CoordinateFrame, q: float
) -> np.ndarray:
    """
    Calculate the coherent scattering function.
    """
    box = start_frame.box.diagonal()
    dimension = len(box)

    def scfunc(x, y):
        sqdist = 0
        for i in range(dimension):
            d = x[i] - y[i]
            if d > box[i] / 2:
                d -= box[i]
            if d < -box[i] / 2:
                d += box[i]
            sqdist += d**2
        x = sqdist**0.5 * q
        if x == 0:
            return 1.0
        else:
            return np.sin(x) / x

    return coherent_sum(scfunc, start_frame.pbc, end_frame.pbc) / len(start_frame)


def non_gaussian_parameter(
    start_frame: CoordinateFrame,
    end_frame: CoordinateFrame,
    trajectory: Coordinates = None,
    axis: str = "all",
) -> float:
    """
    Calculate the non-Gaussian parameter.
    ..math:
      \alpha_2 (t) =
        \frac{3}{5}\frac{\langle r_i^4(t)\rangle}{\langle r_i^2(t)\rangle^2} - 1
    """
    if trajectory is None:
        vectors = start_frame - end_frame
    else:
        vectors = displacements_without_drift(start_frame, end_frame, trajectory)
    if axis == "all":
        r = (vectors**2).sum(axis=1)
        dimensions = 3
    elif axis == "xy" or axis == "yx":
        r = (vectors[:, [0, 1]]**2).sum(axis=1)
        dimensions = 2
    elif axis == "xz" or axis == "zx":
        r = (vectors[:, [0, 2]]**2).sum(axis=1)
        dimensions = 2
    elif axis == "yz" or axis == "zy":
        r = (vectors[:, [1, 2]]**2).sum(axis=1)
        dimensions = 2
    elif axis == "x":
        r = vectors[:, 0] ** 2
        dimensions = 1
    elif axis == "y":
        r = vectors[:, 1] ** 2
        dimensions = 1
    elif axis == "z":
        r = vectors[:, 2] ** 2
        dimensions = 1
    else:
        raise ValueError('Parameter axis has to be ether "all", "x", "y", or "z"!')

    return (np.mean(r**2) / ((1 + 2 / dimensions) * (np.mean(r) ** 2))) - 1
