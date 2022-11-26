import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_update_vorticity_from_penalised_velocity_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def update_vorticity_from_velocity_forcing_reference(
    vorticity_field, velocity_forcing_field, prefactor
):
    new_vorticity_field = np.zeros_like(vorticity_field)
    new_vorticity_field[1:-1, 1:-1] = (
        vorticity_field[1:-1, 1:-1]
        + (
            velocity_forcing_field[1, 1:-1, 2:]
            - velocity_forcing_field[1, 1:-1, :-2]
            - velocity_forcing_field[0, 2:, 1:-1]
            + velocity_forcing_field[0, :-2, 1:-1]
        )
        * prefactor
    )

    return new_vorticity_field


class UpdateVorticityFromVelocityForcingSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_vorticity_field = np.random.rand(n_samples, n_samples).astype(real_t)
        self.ref_velocty_forcing_field = np.random.rand(2, n_samples, n_samples).astype(
            real_t
        )
        self.prefactor = real_t(0.1)
        self.ref_new_vorticity_field = update_vorticity_from_velocity_forcing_reference(
            self.ref_vorticity_field, self.ref_velocty_forcing_field, self.prefactor
        )

    def check_equals(self, new_vorticity_field):
        np.testing.assert_allclose(
            self.ref_new_vorticity_field[1:-1, 1:-1],
            new_vorticity_field[1:-1, 1:-1],
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_update_vorticity_from_velocity_forcing_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = UpdateVorticityFromVelocityForcingSolution(n_values, precision)
    vorticity_field = solution.ref_vorticity_field.copy()
    update_vorticity_from_velocity_forcing_pyst_kernel = (
        gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    update_vorticity_from_velocity_forcing_pyst_kernel(
        vorticity_field=vorticity_field,
        velocity_forcing_field=solution.ref_velocty_forcing_field,
        prefactor=solution.prefactor,
    )
    solution.check_equals(vorticity_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_update_vorticity_from_penalised_velocity_2d(n_values, precision):
    real_t = get_real_t(precision)
    vorticity_field = np.random.rand(n_values, n_values).astype(real_t)
    velocity_field = np.random.rand(2, n_values, n_values).astype(real_t)
    penalised_velocity_field = np.random.rand(2, n_values, n_values).astype(real_t)
    prefactor = real_t(0.1)
    ref_new_vorticity_field = update_vorticity_from_velocity_forcing_reference(
        vorticity_field=vorticity_field,
        velocity_forcing_field=penalised_velocity_field - velocity_field,
        prefactor=prefactor,
    )
    update_vorticity_from_penalised_vorticity_kernel = (
        gen_update_vorticity_from_penalised_velocity_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    update_vorticity_from_penalised_vorticity_kernel(
        vorticity_field=vorticity_field,
        penalised_velocity_field=penalised_velocity_field,
        velocity_field=velocity_field,
        prefactor=prefactor,
    )
    np.testing.assert_allclose(
        ref_new_vorticity_field[1:-1, 1:-1],
        vorticity_field[1:-1, 1:-1],
        atol=get_test_tol(precision),
    )
