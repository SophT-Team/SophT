import numpy as np
import pytest
from sopht.utils import get_real_t, get_test_tol


@pytest.mark.parametrize("precision", ["half", "single", "double"])
def test_get_real_t(precision):
    if precision == "single":
        assert get_real_t(precision) == np.float32
    elif precision == "double":
        assert get_real_t(precision) == np.float64
    else:
        with pytest.raises(
            ValueError, match="Precision argument must be single or double"
        ):
            get_real_t(precision)


@pytest.mark.parametrize("precision", ["single", "double"])
def test_get_test_tol(precision):
    test_tol = get_test_tol(precision)
    if precision == "single":
        assert test_tol == np.float32(1e3 * np.finfo(np.float32).eps)
    elif precision == "double":
        assert test_tol == np.float64(1e3 * np.finfo(np.float64).eps)
