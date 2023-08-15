import numpy as np
from sopht.simulator.flow.flow_simulators import FlowSimulator
import sopht.numeric.eulerian_grid_ops as spne
import sopht.utils as spu
from typing import Callable, Literal, Type
from sopht.simulator.flow.passive_transport_flow_simulators import (
    compute_advection_diffusion_stable_timestep,
)


class PassiveTransportScalarFieldFlowSimulator(FlowSimulator):
    """Class for passive transport flow simulator.

    Solves advection diffusion equations for a passive scalar
    or vector field.
    """

    def __init__(
        self,
        diffusivity_constant: float,
        grid_dim: int,
        grid_size: tuple[int, int] | tuple[int, int, int],
        x_range: float,
        cfl: float = 0.1,
        real_t: Type = np.float32,
        num_threads: int = 1,
        time: float = 0.0,
        field_type: Literal["scalar"] = "scalar",
        with_forcing: bool = False,
        velocity_field: np.ndarray = np.array([None]),
        **kwargs,
    ) -> None:
        """Class initialiser

        :param diffusivity_constant: kinematic viscosity or thermal diffusivity
        :param grid_dim: grid dimensions
        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param cfl: Courant Fredreich Lewy number
        :param real_t: precision of the solver
        :param num_threads: number of threads
        :param time: simulator time at initialisation
        :param field_type: type of primary field (scalar or vector)
        :param with_forcing: flag indicating presence of body forcing
        :param with_free_stream_flow: flag indicating presence of free stream flow

        """
        if field_type not in ["scalar"]:
            raise ValueError("Invalid field type. Supported values include 'scalar' ")

        self.diffusivity_constant = diffusivity_constant
        self.cfl = cfl
        self.field_type = field_type
        self.with_forcing = with_forcing
        self.penalty_zone_width = kwargs.get("penalty_zone_width", 2)
        # Create reference to the velocity field. Velocity field will be computed by the Navier-Stokes Flow simulator.
        self.velocity_field = np.ndarray.view(velocity_field)

        super().__init__(
            grid_dim=grid_dim,
            grid_size=grid_size,
            x_range=x_range,
            real_t=real_t,
            num_threads=num_threads,
            time=time,
        )

    def _init_fields(self) -> None:
        """Initialize the necessary field arrays"""
        self.primary_field = np.zeros(self.grid_size, dtype=self.real_t)

        # self.velocity_field = np.zeros(
        #     (self.grid_dim, *self.grid_size), dtype=self.real_t
        # )
        # we use the same buffer for advection and diffusion fluxes
        self.buffer_scalar_field = np.zeros(self.grid_size, dtype=self.real_t)

        self.gradient_of_primary_field = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )

        if self.with_forcing:
            # this one holds the forcing from bodies
            self.eul_grid_forcing_field = np.zeros_like(self.primary_field)

    def _compile_kernels(self) -> None:
        """Compile necessary kernels based on flow type"""
        match self.grid_dim:
            case 2:
                self._diffusion_timestep = (
                    spne.gen_diffusion_timestep_euler_forward_pyst_kernel_2d(
                        real_t=self.real_t,
                        fixed_grid_size=self.grid_size,
                        num_threads=self.num_threads,
                    )
                )
                self._advection_timestep = spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_2d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
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
                    self._update_passive_field_from_forcing = (
                        spne.gen_update_passive_field_from_forcing_pyst_kernel_2d(
                            real_t=self.real_t,
                            fixed_grid_size=self.grid_size,
                            num_threads=self.num_threads,
                        )
                    )
                    self._set_field = spne.gen_set_fixed_val_pyst_kernel_2d(
                        real_t=self.real_t,
                        num_threads=self.num_threads,
                        field_type=self.field_type,
                    )

            case 3:
                # raise ValueError("3D not supported yet.")

                self._diffusion_timestep = (
                    spne.gen_diffusion_timestep_euler_forward_pyst_kernel_3d(
                        real_t=self.real_t,
                        fixed_grid_size=self.grid_size,
                        num_threads=self.num_threads,
                        field_type=self.field_type,
                    )
                )
                self._advection_timestep = spne.gen_advection_timestep_euler_forward_conservative_eno3_pyst_kernel_3d(
                    real_t=self.real_t,
                    fixed_grid_size=self.grid_size,
                    num_threads=self.num_threads,
                    field_type=self.field_type,
                )

                self._penalise_field_towards_boundary = (
                    spne.gen_penalise_field_boundary_pyst_kernel_3d(
                        width=self.penalty_zone_width,
                        dx=self.dx,
                        x_grid_field=self.position_field[spu.VectorField.x_axis_idx()],
                        y_grid_field=self.position_field[spu.VectorField.y_axis_idx()],
                        z_grid_field=self.position_field[spu.VectorField.z_axis_idx()],
                        real_t=self.real_t,
                        num_threads=self.num_threads,
                        fixed_grid_size=self.grid_size,
                        field_type=self.field_type,
                    )
                )

                # self._gradient_of_field = NotImplemented

                if self.with_forcing:
                    self._update_passive_field_from_forcing = (
                        spne.gen_update_passive_field_from_forcing_pyst_kernel_3d(
                            real_t=self.real_t,
                            fixed_grid_size=self.grid_size,
                            num_threads=self.num_threads,
                        )
                    )
                    self._set_field = spne.gen_set_fixed_val_pyst_kernel_3d(
                        real_t=self.real_t,
                        num_threads=self.num_threads,
                        field_type=self.field_type,
                    )

    def _advection_and_diffusion_time_step(self, dt: float) -> None:
        """Advection and diffusion time step"""
        self._advection_timestep(
            self.primary_field,
            advection_flux=self.buffer_scalar_field,
            velocity=self.velocity_field,
            dt_by_dx=self.real_t(dt / self.dx),
        )
        self._diffusion_timestep(
            self.primary_field,
            diffusion_flux=self.buffer_scalar_field,
            nu_dt_by_dx2=self.real_t(
                self.diffusivity_constant * dt / self.dx / self.dx
            ),
        )
        self._penalise_field_towards_boundary(field=self.primary_field)

    def _advection_and_diffusion_with_forcing_time_step(self, dt: float) -> None:
        self._update_passive_field_from_forcing(
            passive_field=self.primary_field,
            forcing_field=self.eul_grid_forcing_field,
            prefactor=self.real_t(dt),
        )
        self._advection_and_diffusion_time_step(dt)
        self._set_field(self.eul_grid_forcing_field, 0.0)

    def _finalise_flow_time_step(self) -> None:
        """Finalise the flow time step"""
        self._flow_time_step: Callable = self._advection_and_diffusion_time_step

        if self.with_forcing:
            self._flow_time_step = self._advection_and_diffusion_with_forcing_time_step

    def compute_stable_timestep(self, dt_prefac: float = 1.0) -> float:
        """Compute upper limit for stable time-stepping."""
        dt = compute_advection_diffusion_stable_timestep(
            velocity_field=self.velocity_field,
            velocity_magnitude_field=self.buffer_scalar_field,
            grid_dim=self.grid_dim,
            dx=self.dx,
            cfl=self.cfl,
            kinematic_viscosity=self.diffusivity_constant,
            real_t=self.real_t,
        )
        return dt * dt_prefac
