import numpy as np
from .flow_simulators import FlowSimulator
import sopht.numeric.eulerian_grid_ops as spne
from typing import Callable, Literal, Type


class PassiveTransportFlowSimulator(FlowSimulator):
    """Class for passive transport flow simulator.

    Solves advection diffusion equations for a passive scalar
    or vector field.
    """

    def __init__(
        self,
        kinematic_viscosity: float,
        grid_dim: int,
        grid_size: tuple[int, int] | tuple[int, int, int],
        x_range: float,
        cfl: float = 0.1,
        real_t: Type = np.float32,
        num_threads: int = 1,
        time: float = 0.0,
        field_type: Literal["scalar", "vector"] = "scalar",
    ) -> None:
        """Class initialiser

        :param kinematic_viscosity: kinematic viscosity
        :param grid_dim: grid dimensions
        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param cfl: Courant Fredreich Lewy number
        :param real_t: precision of the solver
        :param num_threads: number of threads
        :param time: simulator time at initialisation
        :param field_type: type of primary field (scalar or vector)

        """
        if field_type not in ["scalar", "vector"]:
            raise ValueError(
                "Invalid field type. Supported values include 'scalar' and 'vector'"
            )
        # TODO add support for passive transport of vector field in 2D
        if grid_dim == 2 and field_type == "vector":
            raise ValueError("Passive transport of vector 2D fields not supported yet.")
        self.kinematic_viscosity = kinematic_viscosity
        self.cfl = cfl
        self.field_type = field_type
        super().__init__(grid_dim, grid_size, x_range, real_t, num_threads, time)

    def _init_fields(self) -> None:
        """Initialize the necessary field arrays"""
        match self.field_type:
            case "scalar":
                self.primary_field = np.zeros(self.grid_size, dtype=self.real_t)
            case "vector":
                self.primary_field = np.zeros(
                    (self.grid_dim, *self.grid_size), dtype=self.real_t
                )
        self.velocity_field = np.zeros(
            (self.grid_dim, *self.grid_size), dtype=self.real_t
        )
        # we use the same buffer for advection and diffusion fluxes
        self.buffer_scalar_field = np.zeros(self.grid_size, dtype=self.real_t)

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
            case 3:
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

    def _advection_and_diffusion_time_step(self, dt: float, **kwargs) -> None:
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
            nu_dt_by_dx2=self.real_t(self.kinematic_viscosity * dt / self.dx / self.dx),
        )

    def _finalise_flow_time_step(self) -> None:
        """Finalise the flow time step"""
        self._flow_time_step: Callable = self._advection_and_diffusion_time_step

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


def compute_advection_diffusion_stable_time_step(
    velocity_field: np.ndarray,
    velocity_magnitude_field: np.ndarray,
    grid_dim: int,
    dx: float,
    cfl: float,
    kinematic_viscosity: float,
    real_t: type = np.float32,
) -> float:
    """Compute stable timestep based on advection and diffusion limits."""
    # This may need a numba or pystencil version
    tol = 10 * np.finfo(real_t).eps
    velocity_magnitude_field[...] = np.sum(np.fabs(velocity_field), axis=0)
    dt = min(
        cfl * dx / (np.amax(velocity_magnitude_field) + tol),
        0.9 * dx**2 / (2 * grid_dim) / kinematic_viscosity + tol,
    )
    return dt
