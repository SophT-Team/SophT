import logging
import numpy as np
from typing import Type
from abc import abstractmethod


class FlowSimulator:
    """Class for base flow simulator

    All flow simulators will have the interface of this class and
    should be derived from this class.
    """

    def __init__(
        self,
        grid_dim: int,
        # TODO fix mypy error and enable type hint
        grid_size,  # : tuple[int, int] | tuple[int, int, int],
        x_range: float,
        real_t: Type = np.float32,
        num_threads: int = 1,
        time: float = 0.0,
    ) -> None:
        """Class initialiser

        :param grid_dim: grid dimensions
        :param grid_size: Grid size of simulator
        :param x_range: Range of X coordinate of the grid
        :param real_t: precision of the solver
        :param num_threads: number of threads
        :param time: simulator time at initialisation

        """
        if grid_dim not in [2, 3]:
            raise ValueError(
                "Invalid grid dimensions. Supported values include 2 and 3."
            )
        self.grid_dim = grid_dim
        self.grid_size = grid_size
        self.x_range = x_range
        self.real_t = real_t
        self.num_threads = num_threads
        self.time = time
        self._init_domain()
        self._init_fields()
        self._compile_kernels()
        self._finalise_flow_time_step()

    def _init_domain(self) -> None:
        """Initialize the domain i.e. grid coordinates. etc."""
        grid_size_x = self.grid_size[-1]
        self.dx = self.real_t(self.x_range / grid_size_x)
        eul_grid_shift = self.dx / 2.0
        x = np.linspace(
            eul_grid_shift, self.x_range - eul_grid_shift, grid_size_x
        ).astype(self.real_t)
        match self.grid_dim:
            case 2:
                grid_size_y, grid_size_x = self.grid_size
                self.y_range = self.x_range * grid_size_y / grid_size_x
                y = np.linspace(
                    eul_grid_shift, self.y_range - eul_grid_shift, grid_size_y
                ).astype(self.real_t)
                # reversing because meshgrid generates in order Y and X
                self.position_field = np.flipud(
                    np.array(np.meshgrid(y, x, indexing="ij"))
                )
            case 3:
                grid_size_z, grid_size_y, grid_size_x = self.grid_size
                self.y_range = self.x_range * grid_size_y / grid_size_x
                self.z_range = self.x_range * grid_size_z / grid_size_x
                y = np.linspace(
                    eul_grid_shift, self.y_range - eul_grid_shift, grid_size_y
                ).astype(self.real_t)
                z = np.linspace(
                    eul_grid_shift, self.z_range - eul_grid_shift, grid_size_z
                ).astype(self.real_t)
                # reversing because meshgrid generates in order Z, Y and X
                self.position_field = np.flipud(
                    np.array(np.meshgrid(z, y, x, indexing="ij"))
                )
        log = logging.getLogger()
        log.warning(
            "==============================================="
            f"\n{self.grid_dim}D flow domain initialized with:"
            f"\nX axis from 0.0 to {self.x_range}"
        )
        match self.grid_dim:
            case 2:
                log.warning(f"Y axis from 0.0 to {self.y_range}")
            case 3:
                log.warning(
                    f"Y axis from 0.0 to {self.y_range}"
                    f"\nZ axis from 0.0 to {self.z_range}"
                )
        log.warning(
            "Please initialize bodies within these bounds!"
            "\n==============================================="
        )

    @abstractmethod
    def _init_fields(self) -> None:
        """Initialize the necessary field arrays"""
        ...

    @abstractmethod
    def _compile_kernels(self) -> None:
        """Compile necessary kernels based on flow type"""
        ...

    @abstractmethod
    def _flow_time_step(self, dt: float, **kwargs):
        """Flow time step, stepping through the algorithm"""
        ...

    @abstractmethod
    def _finalise_flow_time_step(self) -> None:
        """Finalise the flow time step

        This function will set the member function 'flow_time_step'
        based on the type and the flags inside the simulator.
        """
        ...

    @abstractmethod
    def compute_stable_timestep(self) -> float:
        """Compute upper limit for stable time-stepping."""
        ...

    def _update_simulator_time(self, dt: float) -> None:
        """Updates simulator time."""
        self.time += dt

    def time_step(self, dt: float, **kwargs) -> None:
        """Final simulator time step"""
        self._flow_time_step(dt=dt, **kwargs)
        self._update_simulator_time(dt=dt)
