"""Assistant functions."""

import numpy as np


def find_nearest(arr: list | set | np.ndarray, value: float):
    """Return the nearest element to value in arr."""
    arr = np.asarray(arr)
    idx = (np.abs(arr - value)).argmin()
    return arr[idx]
