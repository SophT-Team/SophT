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
    gen_laplacian_filter_kernel_3d,
    FastDiagPoissonSolver3D,
)
from sopht.utils.precision import get_test_tol
from sopht.utils.field import VectorField
from typing import Tuple, Type, Union, Callable


# TODO refactor 2D and 3D with a common base simulator class
class UnboundedFlowSimulator3D:
    """Class for 3D unbounded flow simulator"""

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        x_range: float,
        kinematic_viscosity: float,
        CFL: float = 0.1,
        flow_type: str = "passive_scalar",
        real_t: Type = np.float32,
        num_threads: int = 1,
        filter_vorticity=False,
        poisson_solver_type="greens_function_convolution",
        time: float = 0.0,
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
        :param filter_vorticity: flag to determine if vorticity should be filtered or not,
        needed for stability sometimes
        :param poisson_solver_type: Type of the poisson solver algorithm, can be
        "greens_function_convolution" or "fast_diagonalisation"
        :param time: simulator time at initialisation

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
        self.time = time
        self.filter_vorticity = filter_vorticity
        self.poisson_solver_type = poisson_solver_type
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
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
            self.with_free_stream_flow = kwargs.get("with_free_stream_flow", False)
            self.navier_stokes_inertial_term_form = kwargs.get(
                "navier_stokes_inertial_term_form", "rotational"
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
            if self.filter_vorticity:
                log = logging.getLogger()
                log.warning(
                    "==============================================="
                    "\nVorticity filtering is turned on."
                )
                self.filter_setting_dict = kwargs.get("filter_setting_dict")
                if self.filter_setting_dict is None:
                    # set default values for the filter setting dictionary
                    self.filter_setting_dict = {"order": 2, "type": "multiplicative"}
                    log.warning(
                        "Since a dict named filter_setting with keys "
                        "\n'order' and 'type' is not provided, setting "
                        f"\ndefault filter order = {self.filter_setting_dict['order']}"
                        f"\nand type: {self.filter_setting_dict['type']}"
                    )
                log.warning("===============================================")
            # check validity of poisson solver types
            supported_poisson_solver_types = [
                "greens_function_convolution",
                "fast_diagonalisation",
            ]
            if self.poisson_solver_type not in supported_poisson_solver_types:
                raise ValueError("Invalid Poisson solver type given")
        self.compile_kernels()
        self.finalise_flow_timestep()

    def init_domain(self) -> None:
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
        # reversing because meshgrid generates in order Z, Y and X
        self.position_field = np.flipud(np.array(np.meshgrid(z, y, x, indexing="ij")))
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

    def init_fields(self) -> None:
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field = np.zeros(self.grid_size, dtype=self.real_t)
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

    def compile_kernels(self) -> None:
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
            self.unbounded_poisson_solver: Union[
                UnboundedPoissonSolverPYFFTW3D, FastDiagPoissonSolver3D
            ]
            if self.poisson_solver_type == "greens_function_convolution":
                self.unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW3D(
                    grid_size_z=grid_size_z,
                    grid_size_y=grid_size_y,
                    grid_size_x=grid_size_x,
                    x_range=self.x_range,
                    real_t=self.real_t,
                    num_threads=self.num_threads,
                )
            elif self.poisson_solver_type == "fast_diagonalisation":
                self.unbounded_poisson_solver = FastDiagPoissonSolver3D(
                    grid_size_z=grid_size_z,
                    grid_size_y=grid_size_y,
                    grid_size_x=grid_size_x,
                    dx=self.dx,
                    real_t=self.real_t,
                    # TODO add different options here later
                    bc_type="homogenous_neumann_along_xyz",
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
                    x_grid_field=self.position_field[VectorField.x_axis_idx()],
                    y_grid_field=self.position_field[VectorField.y_axis_idx()],
                    z_grid_field=self.position_field[VectorField.z_axis_idx()],
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

            # filter kernel compilation
            def filter_vector_field(vector_field: np.ndarray) -> None:
                ...

            self.filter_vector_field = filter_vector_field
            if self.filter_vorticity and self.filter_setting_dict is not None:
                self.filter_vector_field = gen_laplacian_filter_kernel_3d(
                    filter_order=self.filter_setting_dict["order"],
                    filter_flux_buffer=self.buffer_vector_field[0],
                    field_buffer=self.buffer_vector_field[1],
                    real_t=self.real_t,
                    num_threads=self.num_threads,
                    fixed_grid_size=self.grid_size,
                    field_type="vector",
                    filter_type=self.filter_setting_dict["type"],
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
        # free stream velocity stuff (only meaningful in navier stokes problems)
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            if self.with_free_stream_flow:
                add_fixed_val = gen_add_fixed_val_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type="vector",
                )

                def update_velocity_with_free_stream(
                    free_stream_velocity: np.ndarray,
                ) -> None:
                    add_fixed_val(
                        sum_field=self.velocity_field,
                        vector_field=self.velocity_field,
                        fixed_vals=free_stream_velocity,
                    )

            else:

                def update_velocity_with_free_stream(
                    free_stream_velocity: np.ndarray,
                ) -> None:
                    ...

            self.update_velocity_with_free_stream = update_velocity_with_free_stream

    def finalise_navier_stokes_timestep(self) -> None:
        def default_navier_stokes_timestep(
            dt: float, free_stream_velocity: np.ndarray
        ) -> None:
            ...

        self.navier_stokes_timestep = default_navier_stokes_timestep
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            # default value corresponds to
            # if self.navier_stokes_inertial_term_form == "advection_stretching_split":
            self.navier_stokes_timestep = (
                self.advection_stretching_split_navier_stokes_timestep
            )
            if self.navier_stokes_inertial_term_form == "rotational":
                self.navier_stokes_timestep = (
                    self.rotational_form_navier_stokes_timestep
                )

    def finalise_flow_timestep(self) -> None:
        self.finalise_navier_stokes_timestep()
        self.flow_time_step: Callable
        # defqult time step
        self.flow_time_step = self.scalar_advection_and_diffusion_timestep
        if self.flow_type == "passive_vector":
            self.flow_time_step = self.vector_advection_and_diffusion_timestep
        elif self.flow_type == "navier_stokes":
            self.flow_time_step = self.navier_stokes_timestep
        elif self.flow_type == "navier_stokes_with_forcing":
            self.flow_time_step = self.navier_stokes_with_forcing_timestep

    def update_simulator_time(self, dt: float) -> None:
        """Updates simulator time."""
        self.time += dt

    def time_step(self, dt: float, **kwargs) -> None:
        """Final simulator time step"""
        self.flow_time_step(dt=dt, **kwargs)
        self.update_simulator_time(dt=dt)

    def scalar_advection_and_diffusion_timestep(self, dt: float, **kwargs) -> None:
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

    def vector_advection_and_diffusion_timestep(self, dt: float, **kwargs) -> None:
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

    def compute_flow_velocity(self, free_stream_velocity: np.ndarray) -> None:
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
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def advection_stretching_split_navier_stokes_timestep(
        self,
        dt: float,
        free_stream_velocity: np.ndarray = np.zeros(3),
    ) -> None:
        self.vorticity_stretching_timestep(
            vorticity_field=self.vorticity_field,
            velocity_field=self.velocity_field,
            vorticity_stretching_flux_field=self.buffer_vector_field,
            dt_by_2_dx=self.real_t(dt / (2 * self.dx)),
        )
        self.vector_advection_and_diffusion_timestep(dt=dt)
        self.filter_vector_field(vector_field=self.vorticity_field)
        self.compute_flow_velocity(free_stream_velocity=free_stream_velocity)

    def rotational_form_navier_stokes_timestep(
        self,
        dt: float,
        free_stream_velocity: np.ndarray = np.zeros(3),
    ) -> None:
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
        self.filter_vector_field(vector_field=self.vorticity_field)
        self.compute_flow_velocity(free_stream_velocity=free_stream_velocity)

    def navier_stokes_with_forcing_timestep(
        self,
        dt: float,
        free_stream_velocity: np.ndarray = np.zeros(3),
    ) -> None:
        self.update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self.navier_stokes_timestep(dt=dt, free_stream_velocity=free_stream_velocity)
        self.set_field(
            vector_field=self.eul_grid_forcing_field, fixed_vals=[0.0] * self.grid_dim
        )

    def compute_stable_timestep(
        self, dt_prefac: float = 1, precision: str = "single"
    ) -> float:
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

    def get_vorticity_divergence_l2_norm(self) -> float:
        """Return L2 norm of divergence of the vorticity field."""
        divergence_field = self.buffer_scalar_field.view()
        self.compute_divergence(
            divergence=divergence_field,
            field=self.vorticity_field,
            inv_dx=(1.0 / self.dx),
        )
        vorticity_divg_l2_norm = np.linalg.norm(divergence_field) * self.dx**1.5
        return vorticity_divg_l2_norm
