import numpy as np
import pytest
from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol
from test_advection_flux_2d import advection_flux_conservative_eno_ord3_reference


def advection_timestep_conservative_eno3_euler_forward_reference(
    field, velocity_x, velocity_y, inv_dx, dt, real_t
):
    advection_flux = advection_flux_conservative_eno_ord3_reference(
        field=field,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        inv_dx=inv_dx,
        real_t=real_t,
    )
    return field - dt * advection_flux


class AdvectionTimestepSolution:
    def __init__(
        self,
        rng_generator: np.random.Generator,
        n_samples: int,
        timestepper: str = "euler_forward",
        flux_type: str = "conservative_eno3",
        precision: str = "single",
    ):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = rng_generator.standard_normal((n_samples, n_samples)).astype(real_t)
        self.ref_velocity = rng_generator.standard_normal((2, n_samples, n_samples)).astype(real_t)
        self.inv_dx = real_t(0.2)
        self.dt = real_t(0.1)
        if timestepper == "euler_forward" and flux_type == "conservative_eno3":
            self.kernel_width = 2
            self.ref_new_field = advection_timestep_conservative_eno3_euler_forward_reference(
                field=self.ref_field,
                velocity_x=self.ref_velocity[0],
                velocity_y=self.ref_velocity[1],
                inv_dx=self.inv_dx,
                dt=self.dt,
                real_t=real_t,
            )

    def check_equals(self, new_field):
        np.testing.assert_allclose(
            self.ref_new_field,
            new_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_euler_forward_conservative_eno3_2d(n_values, precision, rng, max_cpu_count):
    real_t = get_real_t(precision)
    solution = AdvectionTimestepSolution(
        rng,
        n_values,
        timestepper="euler_forward",
        flux_type="conservative_eno3",
        precision=precision,
    )
    advection_flux = np.zeros_like(solution.ref_field)
    dt_by_dx = real_t(solution.dt * solution.inv_dx)
    advection_timestep_euler_forward_conservative_eno3_kernel = (
        gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=max_cpu_count,
        )
    )
    field = solution.ref_field.copy()
    advection_timestep_euler_forward_conservative_eno3_kernel(
        field=field,
        advection_flux=advection_flux,
        velocity=solution.ref_velocity,
        dt_by_dx=dt_by_dx,
    )
    solution.check_equals(field)
