import logging

import numpy as np

from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d,
    gen_diffusion_timestep_euler_forward_pyst_kernel_2d,
    gen_penalise_field_boundary_pyst_kernel_2d,
    gen_outplane_field_curl_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d,
    UnboundedPoissonSolverPYFFTW2D,
)
from sopht.utils.precision import get_test_tol


class UnboundedFlowSimulator2D:
    """Class for 2D unbounded flow simulator"""

    def __init__(
        self,
        grid_size,
        x_range,
        kinematic_viscosity,
        CFL=0.1,
        flow_type="passive_scalar",
        real_t=np.float32,
        num_threads=1,
        **kwargs,
    ):
        """Class initialiser

        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param kinematic_viscosity: kinematic viscosity of the fluid
        :param CFL: Courant Freidrich Lewy number (advection timestep)
        :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
        "navier_stokes" or "navier_stokes_with_forcing"
        :param real_t: precision of the solver
        :param num_threads: number of threads

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.grid_size = grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.num_threads = num_threads
        self.flow_type = flow_type
        self.kinematic_viscosity = kinematic_viscosity
        self.CFL = CFL
        self.init_domain()
        self.init_fields()
        if (
            self.flow_type == "navier_stokes"
            or self.flow_type == "navier_stokes_with_forcing"
        ):
            if "penalty_zone_width" in kwargs:
                self.penalty_zone_width = kwargs.get("penalty_zone_width")
            else:
                self.penalty_zone_width = 2
        self.compile_kernels()
        self.finalise_flow_timestep()

    def init_domain(self):
        """Initialize the domain i.e. grid coordinates. etc."""
        grid_size_y = self.grid_size[0]
        grid_size_x = self.grid_size[1]
        self.y_range = self.x_range * grid_size_y / grid_size_x
        self.dx = self.real_t(self.x_range / grid_size_x)
        eul_grid_shift = self.dx / 2
        x = np.linspace(
            eul_grid_shift, self.x_range - eul_grid_shift, grid_size_x
        ).astype(self.real_t)
        y = np.linspace(
            eul_grid_shift, self.y_range - eul_grid_shift, grid_size_y
        ).astype(self.real_t)
        self.x_grid, self.y_grid = np.meshgrid(x, y)
        log = logging.getLogger()
        log.warning(
            "==============================================="
            "\n2D flow domain initialized with:"
            f"\nX axis from 0.0 to {self.x_range}"
            f"\nY axis from 0.0 to {self.y_range}"
            "\nPlease initialize bodies within these bounds!"
            "\n==============================================="
        )

    def init_fields(self):
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field = np.zeros_like(self.x_grid)
        self.velocity_field = np.zeros((2, *self.grid_size), dtype=self.real_t)
        # we use the same buffer for advection, diffusion and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.primary_scalar_field)

        if (
            self.flow_type == "navier_stokes"
            or self.flow_type == "navier_stokes_with_forcing"
        ):
            self.vorticity_field = self.primary_scalar_field.view()
            self.stream_func_field = np.zeros_like(self.vorticity_field)
        if self.flow_type == "navier_stokes_with_forcing":
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def compile_kernels(self):
        """Compile necessary kernels based on flow type"""
        self.diffusion_timestep = gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
            real_t=self.real_t,
            fixed_grid_size=self.grid_size,
            num_threads=self.num_threads,
        )
        self.advection_timestep = (
            gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )
        )

        if (
            self.flow_type == "navier_stokes"
            or self.flow_type == "navier_stokes_with_forcing"
        ):
            self.unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW2D(
                grid_size_y=self.grid_size[0],
                grid_size_x=self.grid_size[1],
                x_range=self.x_range,
                real_t=self.real_t,
                num_threads=self.num_threads,
            )
            self.outplane_field_curl = gen_outplane_field_curl_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                fixed_grid_size=self.grid_size,
            )
            self.penalise_field_towards_boundary = (
                gen_penalise_field_boundary_pyst_kernel_2d(
                    width=self.penalty_zone_width,
                    dx=self.dx,
                    x_grid_field=self.x_grid,
                    y_grid_field=self.y_grid,
                    real_t=self.real_t,
                    num_threads=self.num_threads,
                    fixed_grid_size=self.grid_size,
                )
            )

        if self.flow_type == "navier_stokes_with_forcing":
            self.update_vorticity_from_velocity_forcing = (
                gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                )
            )
            self.set_field = gen_set_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                field_type="vector",
            )

    def finalise_flow_timestep(self):
        # defqult time step
        self.time_step = self.advection_and_diffusion_timestep

        if self.flow_type == "navier_stokes":
            self.time_step = self.navier_stokes_timestep
        elif self.flow_type == "navier_stokes_with_forcing":
            self.time_step = self.navier_stokes_with_forcing_timestep

    def advection_and_diffusion_timestep(self, dt):
        self.advection_timestep(
            field=self.primary_scalar_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        self.diffusion_timestep(
            field=self.primary_scalar_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )

    def compute_velocity_from_vorticity(
        self,
    ):
        self.penalise_field_towards_boundary(field=self.vorticity_field)
        self.unbounded_poisson_solver.solve(
            solution_field=self.stream_func_field,
            rhs_field=self.vorticity_field,
        )
        self.outplane_field_curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )

    def navier_stokes_timestep(self, dt):
        self.advection_and_diffusion_timestep(dt=dt)
        self.compute_velocity_from_vorticity()

    def navier_stokes_with_forcing_timestep(self, dt):
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.navier_stokes_timestep(dt=dt)
        self.set_field(vector_field=self.eul_grid_forcing_field, fixed_vals=[0.0, 0.0])

    def compute_stable_timestep(self, dt_prefac=1, precision="single"):
        """Compute stable timestep based on advection and diffusion limits."""
        # This may need a numba or pystencil version
        velocity_mag_field = self.buffer_scalar_field.view()
        velocity_mag_field[...] = np.sqrt(
            self.velocity_field[0] ** 2 + self.velocity_field[1] ** 2
        )
        dt = min(
            self.CFL
            * self.dx
            / (np.amax(velocity_mag_field) + get_test_tol(precision)),
            self.dx**2 / 4 / self.kinematic_viscosity,
        )
        return dt * dt_prefac
