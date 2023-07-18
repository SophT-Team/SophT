import numpy as np

import psutil

import pytest
from sopht.utils.precision import get_real_t, get_test_tol
import sys

sys.path.append("/")
import sopht.numeric.eulerian_grid_ops as spne


def update_passive_field_from_forcing_reference(
    passive_field, forcing_field, prefactor
):
    new_passive_field = np.zeros_like(passive_field)
    new_passive_field[1:-1, 1:-1] = (
        passive_field[1:-1, 1:-1] + forcing_field[1:-1, 1:-1] * prefactor
    )

    return new_passive_field


class UpdatePassiveFieldFromForcingSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_passive_field = np.random.rand(n_samples, n_samples).astype(real_t)
        self.ref_forcing_field = np.random.rand(n_samples, n_samples).astype(real_t)
        self.prefactor = real_t(0.1)
        self.ref_new_passive_field = update_passive_field_from_forcing_reference(
            self.ref_passive_field, self.ref_forcing_field, self.prefactor
        )

    def check_equals(self, new_passive_field):
        np.testing.assert_allclose(
            self.ref_new_passive_field[1:-1, 1:-1],
            new_passive_field[1:-1, 1:-1],
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_update_passive_field_from_forcing_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = UpdatePassiveFieldFromForcingSolution(n_values, precision)
    passive_field = solution.ref_passive_field.copy()
    update_vorticity_from_velocity_forcing_pyst_kernel = (
        spne.gen_update_passive_field_from_forcing_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    update_vorticity_from_velocity_forcing_pyst_kernel(
        passive_field=passive_field,
        forcing_field=solution.ref_forcing_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(passive_field)
