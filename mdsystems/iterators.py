"""Functions for performing analysis on iterated frames."""

from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np

from .atoms import System


def time_average(
    fn: Callable,
    sys: System,
    ref: System | None = None,
    com: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> np.ndarray:
    """Return time average of function results.

    Parameters
    ----------
    fn : Callable
        Function to call.
    sys : System
        Trajectory to analyze.
    ref : System | None (default = None)
        Trajectory to compare with sys.
        If None and a second System is necessary, sys is compared to itself.
    com : Bool (default = False)
        If True, reduce coordinates to residue-level centers of mass.
    start_frame : int (default = 0)
        Index of first frame to analyze in trajectory.
    end_frame : int | None (default = None)
        Index of final frame to analyze in trajectory.

    Returns
    -------
    results : np.ndarray
        1D array of time averaged results.
    """
    end_frame = sys.num_frames if end_frame is None else end_frame

    ref = sys if ref is None else ref

    xyz_a = sys.centers_of_mass if com else sys.coordinates
    xyz_b = ref.centers_of_mass if com else sys.coordinates

    unitcell = sys.unitcell

    with ProcessPoolExecutor() as pool:
        futures = []
        for i in range(start_frame, end_frame):
            futures.append(pool.submit(fn, xyz_a[i], xyz_b[i], unitcell[i]))

        results = [f.result() for f in futures]

    return np.mean(results, axis=0)


def shifted_correlation(
    fn,
    sys: System,
    com: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
    windows: int = 10,
    points: int = 100,
):
    """Return time average of function results.

    Parameters
    ----------
    fn : Callable
        Function to call.
    sys : System
        Trajectory to analyze.
    com : Bool (default = False)
        If True, reduce coordinates to residue-level centers of mass.
    start_frame : int (default = 0)
        Index of first frame to analyze in trajectory.
    end_frame : int | None (default = None)
        Index of final frame to analyze in trajectory.
    windows : int (default = 10)
        Number of starting points in the analysis.
    points : int (default = 100)
        Number of points to analyze.

    Returns
    -------
    results : np.ndarray
        1D array of averaged results.
    """
    end_frame = sys.num_frames if end_frame is None else end_frame

    xyz = sys.centers_of_mass if com else sys.coordinates

    t = sys.times
    unitcell = sys.unitcell

    origin_indices = np.unique(
        np.linspace(start_frame, end_frame, windows, endpoint=False, dtype=int)
    )

    lag_indices = np.unique(
        (np.logspace(0, np.log10(end_frame - start_frame + 1), num=points) - 1).astype(int)
    )

    lag_indices = lag_indices[lag_indices < len(t)]

    accumulated = np.zeros(len(lag_indices))
    with ProcessPoolExecutor() as pool:
        futures = []
        for origin in origin_indices:
            frame_indices = origin + lag_indices
            frame_indices = frame_indices[frame_indices < len(t)]

            futures.append(pool.submit(fn, xyz[origin], xyz[frame_indices], unitcell[origin]))

        results = [f.result() for f in futures]

    padded = []
    if results[0].ndim == 1:
        for r in results:
            pad = np.full(len(lag_indices), np.nan)
            pad[: len(r)] = r
            padded.append(pad)
    else:
        max_t = max(r.shape[0] for r in results)
        nbins = results[0].shape[1]
        padded = []
        for r in results:
            pad = np.full((max_t, nbins), np.nan)
            pad[: r.shape[0], :] = r
            padded.append(pad)

    accumulated = np.nanmean(padded, axis=0)
    times = t[lag_indices] - t[0]

    return accumulated, times
