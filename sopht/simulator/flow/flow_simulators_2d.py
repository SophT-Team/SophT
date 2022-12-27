import logging
import numpy as np
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable, Literal


class UnboundedFlowSimulator2D:
    """Class for 2D unbounded flow simulator"""

    def __init__(
        self,
        grid_size: tuple[int, int],
        x_range: float,
        kinematic_viscosity: float,
        CFL: float = 0.1,
        flow_type: Literal[
            "passive_scalar", "navier_stokes", "navier_stokes_with_forcing"
        ] = "passive_scalar",
        real_t: type = np.float32,
        num_threads: int = 1,
        time: float = 0.0,
        **kwargs,
    ) -> None:
        """Class initialiser

        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param kinematic_viscosity: kinematic viscosity of the fluid
        :param CFL: Courant Freidrich Lewy number (advection timestep)
        :param flow_type: Nature of the simulator, can be "passive_scalar" (default value),
        "navier_stokes" or "navier_stokes_with_forcing"
        :param real_t: precision of the solver
        :param num_threads: number of threads
        :param time: simulator time at initialisation

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.grid_dim = 2
        self.grid_size = grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.num_threads = num_threads
        self.flow_type = flow_type
        self.kinematic_viscosity = kinematic_viscosity
        self.CFL = CFL
        self.time = time
        supported_flow_types = [
            "passive_scalar",
            "navier_stokes",
            "navier_stokes_with_forcing",
        ]
        if self.flow_type not in supported_flow_types:
            raise ValueError("Invalid flow type given")
        self._init_domain()
        self._init_fields()
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.with_free_stream_flow = kwargs.get("with_free_stream_flow", False)
            self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
        self._compile_kernels()
        self._finalise_flow_timestep()

    def _init_domain(self) -> None:
        """Initialize the domain i.e. grid coordinates. etc."""
        grid_size_y, grid_size_x = self.grid_size
        self.y_range = self.x_range * grid_size_y / grid_size_x
        self.dx = self.real_t(self.x_range / grid_size_x)
        eul_grid_shift = self.dx / 2.0
        x: np.ndarray = np.linspace(
            eul_grid_shift, self.x_range - eul_grid_shift, grid_size_x
        ).astype(self.real_t)
        y: np.ndarray = np.linspace(
            eul_grid_shift, self.y_range - eul_grid_shift, grid_size_y
        ).astype(self.real_t)
        # reversing because meshgrid generates in order Y and X
        self.position_field = np.flipud(np.array(np.meshgrid(y, x, indexing="ij")))
        log = logging.getLogger()
        log.warning(
            "==============================================="
            f"\n{self.grid_dim}D flow domain initialized with:"
            f"\nX axis from 0.0 to {self.x_range}"
            f"\nY axis from 0.0 to {self.y_range}"
            "\nPlease initialize bodies within these bounds!"
            "\n==============================================="
        )

    def _init_fields(self) -> None:
        """Initialize the necessary field arrays, i.e. vorticity, velocity, etc."""
        # Initialize flow field
        self.primary_scalar_field: np.ndarray = np.zeros(
            self.grid_size, dtype=self.real_t
        )
        self.velocity_field: np.ndarray = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )
        # we use the same buffer for advection, diffusion and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.primary_scalar_field)

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            self.vorticity_field = self.primary_scalar_field.view()
            self.stream_func_field = np.zeros_like(self.vorticity_field)
        if self.flow_type == "navier_stokes_with_forcing":
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def _compile_kernels(self) -> None:
        """Compile necessary kernels based on flow type"""
        self.diffusion_timestep = (
            spne.gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )
        )
        self.advection_timestep = (
            spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )
        )

        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            grid_size_y, grid_size_x = self.grid_size
            self.unbounded_poisson_solver = spne.UnboundedPoissonSolverPYFFTW2D(
                grid_size_y=grid_size_y,
                grid_size_x=grid_size_x,
                x_range=self.x_range,
                real_t=self.real_t,
                num_threads=self.num_threads,
            )
            self.curl = spne.gen_outplane_field_curl_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                fixed_grid_size=self.grid_size,
            )
            self.penalise_field_towards_boundary = (
                spne.gen_penalise_field_boundary_pyst_kernel_2d(
                    width=self.penalty_zone_width,
                    dx=self.dx,
                    x_grid_field=self.position_field[spu.VectorField.x_axis_idx()],
                    y_grid_field=self.position_field[spu.VectorField.y_axis_idx()],
                    real_t=self.real_t,
                    num_threads=self.num_threads,
                    fixed_grid_size=self.grid_size,
                )
            )

        if self.flow_type == "navier_stokes_with_forcing":
            self.update_vorticity_from_velocity_forcing = (
                spne.gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                )
            )
            self.set_field = spne.gen_set_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                field_type="vector",
            )
        # free stream velocity stuff (only meaningful in navier stokes problems)
        if self.flow_type in ["navier_stokes", "navier_stokes_with_forcing"]:
            if self.with_free_stream_flow:
                add_fixed_val = spne.gen_add_fixed_val_pyst_kernel_2d(
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

    def _finalise_flow_timestep(self) -> None:
        self.flow_time_step: Callable
        # defqult time step
        self.flow_time_step = self.advection_and_diffusion_timestep
        if self.flow_type == "navier_stokes":
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

    def advection_and_diffusion_timestep(self, dt: float, **kwargs) -> None:
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
    ) -> None:
        self.penalise_field_towards_boundary(field=self.vorticity_field)
        self.unbounded_poisson_solver.solve(
            solution_field=self.stream_func_field,
            rhs_field=self.vorticity_field,
        )
        self.curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )

    def navier_stokes_timestep(
        self, dt: float, free_stream_velocity: np.ndarray = np.zeros(2)
    ):
        self.advection_and_diffusion_timestep(dt=dt)
        self.compute_velocity_from_vorticity()
        self.update_velocity_with_free_stream(free_stream_velocity=free_stream_velocity)

    def navier_stokes_with_forcing_timestep(
        self, dt: float, free_stream_velocity: np.ndarray = np.zeros(2)
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
        velocity_mag_field[...] = np.sum(np.fabs(self.velocity_field), axis=0)
        dt = min(
            self.CFL
            * self.dx
            / (np.amax(velocity_mag_field) + spu.get_test_tol(precision)),
            0.9 * self.dx**2 / (2 * self.grid_dim) / self.kinematic_viscosity,
        )
        return dt * dt_prefac
