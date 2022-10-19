from numba import njit
import numpy as np


@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@njit(cache=True)
def np_sum_axis1(x):
    """Numba compatible version of np.sum(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.int64)
    for i in range(x.shape[1]):
        out = np.sum(out, x[:, i])
    return out
