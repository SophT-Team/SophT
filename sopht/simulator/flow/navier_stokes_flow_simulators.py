import numpy as np
from .flow_simulators import FlowSimulator
from .passive_transport_flow_simulators import (
    compute_advection_diffusion_stable_time_step,
)
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable


class UnboundedNavierStokesFlowSimulator2D(FlowSimulator):
    """Class for 2D unbounded Navier Stokes flow simulator"""

    def __init__(
        self,
        grid_size: tuple[int, int],
        x_range: float,
        kinematic_viscosity: float,
        cfl: float = 0.1,
        real_t: type = np.float32,
        num_threads: int = 1,
        time: float = 0.0,
        with_forcing: bool = False,
        with_free_stream_flow: bool = False,
        **kwargs,
    ) -> None:
        """Class initialiser

        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param kinematic_viscosity: kinematic viscosity of the fluid
        :param cfl: Courant Freidrich Lewy number (advection timestep)
        :param real_t: precision of the solver
        :param num_threads: number of threads
        :param time: simulator time at initialisation
        :param with_forcing: flag indicating presence of body forcing
        :param with_free_stream_flow: flag indicating presence of free stream flow

        Notes
        -----
        Currently only supports Euler forward timesteps :(
        """
        self.kinematic_viscosity = kinematic_viscosity
        self.cfl = cfl
        self.with_forcing = with_forcing
        self.with_free_stream_flow = with_free_stream_flow
        self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
        super().__init__(
            grid_dim=2,
            grid_size=grid_size,
            x_range=x_range,
            real_t=real_t,
            num_threads=num_threads,
            time=time,
        )

    def _init_fields(self) -> None:
        """Initialize the necessary field arrays"""
        # Initialize flow field
        self.vorticity_field: np.ndarray = np.zeros(self.grid_size, dtype=self.real_t)
        self.velocity_field: np.ndarray = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )
        # we use the same buffer for advection, diffusion and velocity recovery
        self.buffer_scalar_field = np.zeros_like(self.vorticity_field)
        self.stream_func_field = np.zeros_like(self.vorticity_field)
        if self.with_forcing:
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.velocity_field)

    def _compile_kernels(self) -> None:
        """Compile necessary kernels based on simulator flags"""
        self._diffusion_timestep = (
            spne.gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )
        )
        self._advection_timestep = (
            spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                real_t=self.real_t,
                fixed_grid_size=self.grid_size,
                num_threads=self.num_threads,
            )
        )
        grid_size_y, grid_size_x = self.grid_size
        self._unbounded_poisson_solver = spne.UnboundedPoissonSolverPYFFTW2D(
            grid_size_y=grid_size_y,
            grid_size_x=grid_size_x,
            x_range=self.x_range,
            real_t=self.real_t,
            num_threads=self.num_threads,
        )
        self._curl = spne.gen_outplane_field_curl_pyst_kernel_2d(
            real_t=self.real_t,
            num_threads=self.num_threads,
            fixed_grid_size=self.grid_size,
        )
        self._penalise_field_towards_boundary = (
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
        if self.with_forcing:
            self._update_vorticity_from_velocity_forcing = (
                spne.gen_update_vorticity_from_velocity_forcing_pyst_kernel_2d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                )
            )
            self._set_field = spne.gen_set_fixed_val_pyst_kernel_2d(
                real_t=self.real_t,
                num_threads=self.num_threads,
                field_type="vector",
            )
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

        self._update_velocity_with_free_stream = update_velocity_with_free_stream

    def _finalise_flow_time_step(self) -> None:
        # defqult time step
        self._flow_time_step: Callable = self._navier_stokes_time_step
        if self.with_forcing:
            self._flow_time_step = self._navier_stokes_with_forcing_time_step

    def _navier_stokes_time_step(
        self, dt: float, free_stream_velocity: np.ndarray = np.zeros(2)
    ):
        # advect vorticity
        self._advection_timestep(
            field=self.vorticity_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        # diffusion vorticity
        self._diffusion_timestep(
            field=self.vorticity_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )
        # compute velocity from vorticity
        self._penalise_field_towards_boundary(field=self.vorticity_field)
        self._unbounded_poisson_solver.solve(
            solution_field=self.stream_func_field,
            rhs_field=self.vorticity_field,
        )
        self._curl(
            curl=self.velocity_field,
            field=self.stream_func_field,
            prefactor=self.real_t(0.5 / self.dx),
        )
        self._update_velocity_with_free_stream(
            free_stream_velocity=free_stream_velocity
        )

    def _navier_stokes_with_forcing_time_step(
        self, dt: float, free_stream_velocity: np.ndarray = np.zeros(2)
    ) -> None:
        self._update_vorticity_from_velocity_forcing(
            vorticity_field=self.vorticity_field,
            velocity_forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt / (2 * self.dx)),
        )
        self._navier_stokes_time_step(dt=dt, free_stream_velocity=free_stream_velocity)
        self._set_field(
            vector_field=self.eul_grid_forcing_field, fixed_vals=[0.0] * self.grid_dim
        )

    def compute_stable_time_step(self, dt_prefac: float = 1.0) -> float:
        """Compute upper limit for stable time-stepping."""
        dt = compute_advection_diffusion_stable_time_step(
            velocity_field=self.velocity_field,
            velocity_magnitude_field=self.buffer_scalar_field,
            grid_dim=self.grid_dim,
            dx=self.dx,
            cfl=self.cfl,
            kinematic_viscosity=self.kinematic_viscosity,
            real_t=self.real_t,
        )
        return dt * dt_prefac
