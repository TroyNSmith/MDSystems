"""Functions for performing analysis on iterated frames."""

from typing import Callable

import numpy as np

from . import coordinates


def time_average(
    function: Callable,
    system: coordinates.System,
    ref_system: coordinates.System = None,
    stride: int = 0,
    com: bool = False,
):
    """Compute time average of analytical function."""
    distinct = True
    if ref_system is None:
        ref_system = system.model_copy()
        distinct = False

    system.stride = stride
    ref_system.stride = stride

    unit_cells = coordinates.unit_cells(ref_system)

    if com:
        xyzs, refs = (
            coordinates.com_coordinates(system),
            coordinates.com_coordinates(ref_system),
        )
    else:
        xyzs, refs = (
            coordinates.raw_coordinates(system),
            coordinates.raw_coordinates(ref_system),
        )

    results = []
    for n, (xyz, ref, uc) in enumerate(zip(xyzs, refs, unit_cells, strict=True)):
        result, bin_centers = function(xyz=xyz, ref=ref, unit_cell=uc, distinct=distinct)
        results.append(result)

    time_avg = np.mean(results, axis=0)
    return time_avg, bin_centers
