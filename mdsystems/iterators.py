"""Functions for performing analysis on iterated frames."""

import copy
from typing import Callable

import numpy as np

from . import coordinates


def time_average(
    function: Callable,
    system: "coordinates.System",
    ref_system: "coordinates.System" = None,
    stride: int | None = None,
    com: bool = False,
    vecs: tuple[str, str] | None = None,
):
    """
    Compute time-averaged value of a function over a trajectory.

    Parameters
    ----------
    function : Callable
        Function that takes xyz, ref, first_xyz, first_ref, unitcell, volume, distinct
        and returns a NumPy array (per-frame result).
    system : coordinates.System
        System to analyze.
    ref_system : coordinates.System, optional
        Reference system for comparison. Defaults to a copy of `system`.
    stride : int | None, optional
        Stride to subsample trajectory frames.
    com : bool, default False
        If True, compute centers of mass for residues.
    vecs : tuple(str, str) | None, optional
        If provided, compute vectors between specified atoms.

    Returns
    -------
    mean_result : np.ndarray
        Time-averaged result over frames processed.
    """
    if com and vecs is not None:
        raise ValueError("com=True and vecs!=None are mutually exclusive.")

    distinct = True
    if ref_system is None:
        ref_system = copy.copy(system)
        distinct = False

    if stride is not None:
        system.stride = stride
        ref_system.stride = stride

    unitcell = system.unitcell
    if not np.allclose(unitcell, ref_system.unitcell):
        raise ValueError("Unitcells for system and reference system do not match.")

    volume = system.volume

    results = None

    for i, ((first_xyz, xyz), (first_ref, ref)) in enumerate(
        zip(system.load_frame(com=com, vecs=vecs),
            ref_system.load_frame(com=com, vecs=vecs),
            strict=True)
    ):
        result = function(
            xyz=xyz,
            ref=ref,
            first_xyz=first_xyz,
            first_ref=first_ref,
            unitcell=unitcell,
            volume=volume,
            distinct=distinct,
        )

        result = np.asarray(result, dtype=np.float32)
        if results is None:
            results = np.zeros_like(result, dtype=np.float32)

        results += result
        raise ValueError(results)
