import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_advection_flux_conservative_eno3_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def advection_flux_conservative_eno_ord3_reference(
    field, velocity_x, velocity_y, inv_dx, real_t
):
    kernel_w = 2
    advection_flux = np.zeros_like(field)
    face_velocity_x_east = real_t(0.5) * (
        velocity_x[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + velocity_x[kernel_w:-kernel_w, kernel_w + 1 : -kernel_w + 1]
    )
    face_velocity_x_west = real_t(0.5) * (
        velocity_x[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + velocity_x[kernel_w:-kernel_w, kernel_w - 1 : -kernel_w - 1]
    )
    face_velocity_y_north = real_t(0.5) * (
        velocity_y[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + velocity_y[kernel_w + 1 : -kernel_w + 1, kernel_w:-kernel_w]
    )
    face_velocity_y_south = real_t(0.5) * (
        velocity_y[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + velocity_y[kernel_w - 1 : -kernel_w - 1, kernel_w:-kernel_w]
    )
    nodal_flux = velocity_x * field
    upwind_switch = face_velocity_x_east > 0
    advection_flux_east = (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w + 1 : -kernel_w + 1]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w - 1 : -kernel_w - 1]
    ) * upwind_switch + (real_t(1) - upwind_switch) * (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w + 1 : -kernel_w + 1]
        - real_t(1 / 6) * nodal_flux[kernel_w:-kernel_w, (2 * kernel_w) :]
    )
    upwind_switch = face_velocity_x_west > 0
    advection_flux_west = (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w - 1 : -kernel_w - 1]
        - real_t(1 / 6) * nodal_flux[kernel_w:-kernel_w, : -(2 * kernel_w)]
    ) * upwind_switch + (real_t(1) - upwind_switch) * (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w - 1 : -kernel_w - 1]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w + 1 : -kernel_w + 1]
    )
    nodal_flux[...] = velocity_y * field
    upwind_switch = face_velocity_y_north > 0
    advection_flux_north = (
        real_t(1 / 3) * nodal_flux[kernel_w + 1 : -kernel_w + 1, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[kernel_w - 1 : -kernel_w - 1, kernel_w:-kernel_w]
    ) * upwind_switch + (real_t(1) - upwind_switch) * (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w + 1 : -kernel_w + 1, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[(2 * kernel_w) :, kernel_w:-kernel_w]
    )
    upwind_switch = face_velocity_y_south > 0
    advection_flux_south = (
        real_t(1 / 3) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w - 1 : -kernel_w - 1, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[: -(2 * kernel_w), kernel_w:-kernel_w]
    ) * upwind_switch + (real_t(1) - upwind_switch) * (
        real_t(1 / 3) * nodal_flux[kernel_w - 1 : -kernel_w - 1, kernel_w:-kernel_w]
        + real_t(5 / 6) * nodal_flux[kernel_w:-kernel_w, kernel_w:-kernel_w]
        - real_t(1 / 6) * nodal_flux[kernel_w + 1 : -kernel_w + 1, kernel_w:-kernel_w]
    )
    advection_flux[kernel_w:-kernel_w, kernel_w:-kernel_w] = inv_dx * (
        advection_flux_east
        - advection_flux_west
        + advection_flux_north
        - advection_flux_south
    )
    return advection_flux


class AdvectionFluxSolution:
    def __init__(self, n_samples, flux_type="conservative_eno3", precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_velocity_x = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_velocity_y = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_velocity = np.zeros((2, n_samples, n_samples)).astype(real_t)
        self.ref_velocity[0] = self.ref_velocity_x
        self.ref_velocity[1] = self.ref_velocity_y
        self.inv_dx = real_t(0.1)
        self.flux_type = flux_type
        if self.flux_type == "conservative_eno3":
            self.kernel_width = 2
            self.ref_advection_flux = advection_flux_conservative_eno_ord3_reference(
                self.ref_field,
                self.ref_velocity_x,
                self.ref_velocity_y,
                self.inv_dx,
                real_t=real_t,
            )

    def check_equals(self, advection_flux):
        np.testing.assert_allclose(
            self.ref_advection_flux,
            advection_flux,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_advection_flux_conservative_eno3(n_values, precision):
    real_t = get_real_t(precision)
    solution = AdvectionFluxSolution(
        n_values, flux_type="conservative_eno3", precision=precision
    )
    advection_flux = np.zeros_like(solution.ref_advection_flux)
    advection_flux_conservative_eno3_kernel = (
        gen_advection_flux_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
        )
    )
    advection_flux_conservative_eno3_kernel(
        advection_flux=advection_flux,
        field=solution.ref_field,
        velocity=solution.ref_velocity,
        inv_dx=solution.inv_dx,
    )
    solution.check_equals(advection_flux)
