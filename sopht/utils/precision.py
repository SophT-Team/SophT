"""Precision and tolerance details."""
import numpy as np


def get_real_t(precision: str = "single") -> type:
    """Return the real data type based on precision."""
    if precision == "single":
        return np.float32
    elif precision == "double":
        return np.float64
    else:
        raise ValueError("Precision argument must be single or double")


def get_test_tol(precision: str = "single") -> float:
    """Return the testing tolerance based on precision."""
    real_t = get_real_t(precision=precision)
    return real_t(1e3) * np.finfo(real_t).eps
