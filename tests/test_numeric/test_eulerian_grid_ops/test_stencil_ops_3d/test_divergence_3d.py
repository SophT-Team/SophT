import numpy as np
import psutil
import pytest
from sopht.numeric.eulerian_grid_ops import (
    gen_divergence_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def divergence_reference(field, inv_dx):
    divergence = np.zeros_like(field[0])
    divergence[1:-1, 1:-1, 1:-1] = (
        (
            field[0, 1:-1, 1:-1, 2:]
            - field[0, 1:-1, 1:-1, :-2]
            + field[1, 1:-1, 2:, 1:-1]
            - field[1, 1:-1, :-2, 1:-1]
            + field[2, 2:, 1:-1, 1:-1]
            - field[2, :-2, 1:-1, 1:-1]
        )
        * 0.5
        * inv_dx
    )
    return divergence


class DivergenceSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.dim = 3
        self.ref_field = np.random.randn(
            self.dim, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.inv_dx = real_t(0.1)
        self.ref_divergence = divergence_reference(self.ref_field, self.inv_dx)

    def check_field_equals(self, divergence):
        np.testing.assert_allclose(
            self.ref_divergence,
            divergence,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("reset_ghost_zone", [True, False])
def test_divergence_3d(n_values, precision, reset_ghost_zone):
    real_t = get_real_t(precision)
    solution = DivergenceSolution(n_values, precision)
    divergence = (
        np.ones_like(solution.ref_divergence)
        if reset_ghost_zone
        else np.zeros_like(solution.ref_divergence)
    )
    divergence_pyst_kernel_3d = gen_divergence_pyst_kernel_3d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        reset_ghost_zone=reset_ghost_zone,
    )
    divergence_pyst_kernel_3d(
        divergence=divergence,
        field=solution.ref_field,
        inv_dx=solution.inv_dx,
    )
    solution.check_field_equals(divergence)
