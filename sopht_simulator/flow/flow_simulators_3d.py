import logging

import numpy as np

from sopht.numeric.eulerian_grid_ops import (
    gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d,
    gen_add_fixed_val_pyst_kernel_3d,
    gen_diffusion_timestep_euler_forward_pyst_kernel_3d,
    gen_penalise_field_boundary_pyst_kernel_3d,
    gen_curl_pyst_kernel_3d,
    gen_set_fixed_val_pyst_kernel_3d,
    gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d,
    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d,
    gen_elementwise_cross_product_pyst_kernel_3d,
    UnboundedPoissonSolverPYFFTW3D,
    gen_divergence_pyst_kernel_3d,
)
from sopht.utils.precision import get_test_tol


# TODO refactor 2D and 3D with a common base simulator class
class UnboundedFlowSimulator3D:
    """Class for 3D unbounded flow simulator"""

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
        "passive_vector", "navier_stokes" or "navier_stokes_with_forcing"
        :param real_t: precision of the solver
        :param num_threads: number of threads

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.grid_dim = 3
        self.grid_size = grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.num_threads = num_threads
        self.flow_type = flow_type
        self.kinematic_viscosity = kinematic_viscosity
        self.CFL = CFL
        supported_flow_types = [
            "passive_scalar",
            "passive_vector",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]
        if self.flow_type not in supported_flow_types:
            raise ValueError("Invalid flow type given")
        self.init_domain()
        self.init_fields()
        if (
            self.flow_type == "navier_stokes"
            or self.flow_type == "navier_stokes_with_forcing"
        ):
            self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
            self.with_free_stream_flow = kwargs.get("with_free_stream_flow", False)
            self.navier_stokes_inertial_term_form = kwargs.get(
                "navier_stokes_inertial_term_form", "advection_stretching_split"
            )
            supported_navier_stokes_inertial_term_forms = [
                "advection_stretching_split",
                "rotational",
            ]
            if (
                self.navier_stokes_inertial_term_form
                not in supported_navier_stokes_inertial_term_forms
            ):
                raise ValueError("Invalid Navier Stokes inertial treatment form given")
        self.compile_kernels()
        self.finalise_flow_timestep()

    def init_domain(self):
        """Initialize the domain i.e. grid coordinates. etc."""
        grid_size_z, grid_size_y, grid_size_x = self.grid_size
        self.y_range = self.x_range * grid_size_y / grid_size_x
        self.z_range = self.x_range * grid_size_z / grid_size_x
        self.dx = self.real_t(self.x_range / grid_size_x)
        eul_grid_shift = self.dx / 2.0
        x = np.linspace(
            eul_grid_shift, self.x_range - eul_grid_shift, grid_size_x
        ).astype(self.real_t)
        y = np.linspace(
            eul_grid_shift, self.y_range - eul_grid_shift, grid_size_y
        ).astype(self.real_t)
        z = np.linspace(
            eul_grid_shift, self.z_range - eul_grid_shift, grid_size_z
        ).astype(self.real_t)
        self.z_grid, self.y_grid, self.x_grid = np.meshgrid(z, y, x, indexing="ij")
        log = logging.getLogger()
        log.warning(
            "==============================================="
            f"\n{self.grid_dim}D flow domain initialized with:"
            f"\nX axis from 0.0 to {self.x_range}"
            f"\nY axis from 0.0 to {self.y_range}"
            f"\nZ axis from 0.0 to {self.z_range}"
            "\nPlease initialize bodies within these bounds!"
            "\n==============================================="
        )

    def init_fields(self):
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field = np.zeros_like(self.x_grid)
        self.velocity_field = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )
        # we use the same buffer for advection, diffusion, stretching
        # and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.primary_scalar_field)

        if self.flow_type in [
            "passive_vector",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]:
            self.primary_vector_field = np.zeros_like(self.velocity_field)
            del self.primary_scalar_field  # not needed

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.vorticity_field = self.primary_vector_field.view()
            self.stream_func_field = np.zeros_like(self.vorticity_field)
            self.buffer_vector_field = np.zeros_like(self.vorticity_field)
        if self.flow_type == "navier_stokes_with_forcing":
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def compile_kernels(self):
        """Compile necessary kernels based on flow type"""
        if self.flow_type == "passive_scalar":
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="scalar",
                )
            )
            self.advection_timestep = (
                gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="scalar",
                )
            )
        elif self.flow_type == "passive_vector":
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="vector",
                )
            )
            self.advection_timestep = (
                gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="vector",
                )
            )

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.diffusion_timestep = (
                gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="vector",
                )
            )
            grid_size_z, grid_size_y, grid_size_x = self.grid_size
            self.unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
                grid_size_z=grid_size_z,
                grid_size_y=grid_size_y,
                grid_size_x=grid_size_x,
                x_range=self.x_range,
                real_t=self.real_t,
                num_threads=self.num_threads,
            )
            self.curl = gen_curl_pyst_kernel_3d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                fixed_grid_size=self.grid_size,
            )
            self.penalise_field_towards_boundary = (
                gen_penalise_field_boundary_pyst_kernel_3d(
                    width=self.penalty_zone_width,
                    dx=self.dx,
                    x_grid_field=self.x_grid,
                    y_grid_field=self.y_grid,
                    z_grid_field=self.z_grid,
                    real_t=self.real_t,
                    num_threads=self.num_threads,
                    fixed_grid_size=self.grid_size,
                    field_type="vector",
                )
            )
            if self.navier_stokes_inertial_term_form == "advection_stretching_split":
                self.advection_timestep = gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="vector",
                )
                self.vorticity_stretching_timestep = (
                    gen_vorticity_stretching_timestep_euler_forward_pyst_kernel_3d(
                        real_t=self.real_t,
                        num_threads=self.num_threads,
                        fixed_grid_size=self.grid_size,
                    )
                )
            elif self.navier_stokes_inertial_term_form == "rotational":
                self.elementwise_cross_product = (
                    gen_elementwise_cross_product_pyst_kernel_3d(
                        real_t=self.real_t,
                        num_threads=self.num_threads,
                        fixed_grid_size=self.grid_size,
                    )
                )
                self.update_vorticity_from_velocity_forcing = (
                    gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d(
                        real_t=self.real_t,
                        fixed_grid_size=self.grid_size,
                        num_threads=self.num_threads,
                    )
                )
            # check if vorticity stays divergence free
            self.compute_divergence = gen_divergence_pyst_kernel_3d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )

        if self.flow_type == "navier_stokes_with_forcing":
            self.set_field = gen_set_fixed_val_pyst_kernel_3d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                field_type="vector",
            )
            # Avoid double kernel compilation
            # TODO have a cleaner way for this
            if self.navier_stokes_inertial_term_form != "rotational":
                self.update_vorticity_from_velocity_forcing = (
                    gen_update_vorticity_from_velocity_forcing_pyst_kernel_3d(
                        real_t=self.real_t,
                        fixed_grid_size=self.grid_size,
                        num_threads=self.num_threads,
                    )
                )
        # free stream velocity stuff
        if self.with_free_stream_flow:
            add_fixed_val = gen_add_fixed_val_pyst_kernel_3d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
                field_type="vector",
            )

            def update_velocity_with_free_stream(free_stream_velocity):
                add_fixed_val(
                    sum_field=self.velocity_field,
                    vector_field=self.velocity_field,
                    fixed_vals=free_stream_velocity,
                )

        else:

            def update_velocity_with_free_stream(free_stream_velocity):
                ...

        self.update_velocity_with_free_stream = update_velocity_with_free_stream

    def finalise_navier_stokes_timestep(self):
        self.navier_stokes_timestep = None
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            if self.navier_stokes_inertial_term_form == "advection_stretching_split":
                self.navier_stokes_timestep = (
                    self.advection_stretching_split_navier_stokes_timestep
                )
            elif self.navier_stokes_inertial_term_form == "rotational":
                self.navier_stokes_timestep = (
                    self.rotational_form_navier_stokes_timestep
                )

    def finalise_flow_timestep(self):
        self.finalise_navier_stokes_timestep()
        # defqult time step
        self.time_step = self.scalar_advection_and_diffusion_timestep
        if self.flow_type == "passive_vector":
            self.time_step = self.vector_advection_and_diffusion_timestep
        elif self.flow_type == "navier_stokes":
            self.time_step = self.navier_stokes_timestep
        elif self.flow_type == "navier_stokes_with_forcing":
            self.time_step = self.navier_stokes_with_forcing_timestep

    def scalar_advection_and_diffusion_timestep(self, dt, **kwargs):
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

    def vector_advection_and_diffusion_timestep(self, dt, **kwargs):
        self.advection_timestep(
            vector_field=self.primary_vector_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        self.diffusion_timestep(
            vector_field=self.primary_vector_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )

    def compute_velocity_from_vorticity(
        self,
    ):
        self.penalise_field_towards_boundary(vector_field=self.vorticity_field)
        self.unbounded_poisson_solver.vector_field_solve(
            solution_vector_field=self.stream_func_field,
            rhs_vector_field=self.vorticity_field,
        )
        self.curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )

    def advection_stretching_split_navier_stokes_timestep(
        self, dt, free_stream_velocity=(0.0, 0.0, 0.0)
    ):
        self.vorticity_stretching_timestep(
            vorticity_field=self.vorticity_field,
            velocity_field=self.velocity_field,
            vorticity_stretching_flux_field=self.buffer_vector_field,
            dt_by_2_dx=self.real_t(dt / (2 * self.dx)),
        )
        self.vector_advection_and_diffusion_timestep(dt=dt)
        self.compute_velocity_from_vorticity()
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def rotational_form_navier_stokes_timestep(
        self, dt, free_stream_velocity=(0.0, 0.0, 0.0)
    ):
        velocity_cross_vorticity = self.buffer_vector_field.view()
        self.elementwise_cross_product(
            result_field=velocity_cross_vorticity,
            field_1=self.velocity_field,
            field_2=self.vorticity_field,
        )
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=velocity_cross_vorticity,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.diffusion_timestep(
            vector_field=self.vorticity_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )
        self.compute_velocity_from_vorticity()
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def navier_stokes_with_forcing_timestep(
        self, dt, free_stream_velocity=(0.0, 0.0, 0.0)
    ):
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.navier_stokes_timestep(dt=dt, free_stream_velocity=free_stream_velocity)
        self.set_field(
            vector_field=self.eul_grid_forcing_field, fixed_vals=[0.0] * self.grid_dim
        )

    def compute_stable_timestep(self, dt_prefac=1, precision="single"):
        """Compute stable timestep based on advection and diffusion limits."""
        # This may need a numba or pystencil version
        velocity_mag_field = self.buffer_scalar_field.view()
        tol = get_test_tol(precision)
        velocity_mag_field[...] = np.sum(np.fabs(self.velocity_field), axis=0)
        dt = min(
            self.CFL * self.dx / (np.amax(velocity_mag_field) + tol),
            0.9 * self.dx**2 / (2 * self.grid_dim) / (self.kinematic_viscosity + tol),
        )
        return dt * dt_prefac

    def get_vorticity_divergence_l2_norm(self):
        """Return L2 norm of divergence of the vorticity field."""
        divergence_field = self.buffer_scalar_field.view()
        self.compute_divergence(
            divergence=divergence_field,
            field=self.vorticity_field,
            inv_dx=(1.0 / self.dx),
        )
        vorticity_divg_l2_norm = np.linalg.norm(divergence_field) * self.dx**1.5
        return vorticity_divg_l2_norm
