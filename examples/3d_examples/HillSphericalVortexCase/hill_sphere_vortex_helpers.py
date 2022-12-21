import numpy as np
import sopht.utils as spu


class HillSphereVortex:
    """Class for Hill's spherical vortex properties.

    Computes vorticity, velocities using cylindrical coordinates,
    with axis of vortex aligned along Z

    References:
    https://en.wikipedia.org/wiki/Vortex_ring

    Branlard, E. (2017). Spherical Geometry Models: Flow About a Sphere
    and Hill’s Vortex. In Wind Turbine Aerodynamics and Vorticity-Based
    Methods (pp. 407-417). Springer, Cham.
    """

    def __init__(
        self,
        free_stream_velocity: float,
        vortex_radius: float,
        vortex_origin: tuple[float, float, float],
    ) -> None:
        self.free_stream_velocity = free_stream_velocity
        self.vortex_radius = vortex_radius
        self.vortex_origin = vortex_origin
        self.grid_dim = 3

    def compute_local_coordinates(
        self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_origin, y_origin, z_origin = self.vortex_origin
        local_x_grid = x_grid - x_origin
        local_y_grid = y_grid - y_origin
        local_z_grid = z_grid - z_origin
        cylinder_r_grid = np.sqrt(np.square(local_x_grid) + np.square(local_y_grid))
        sphere_r_grid = np.sqrt(
            np.square(local_x_grid) + np.square(local_y_grid) + np.square(local_z_grid)
        )
        return local_x_grid, local_y_grid, local_z_grid, cylinder_r_grid, sphere_r_grid

    def get_inside_vortex_mask(self, sphere_r_grid: np.ndarray) -> np.ndarray:
        inside_vortex = sphere_r_grid < self.vortex_radius
        return inside_vortex

    def get_vorticity(
        self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
    ) -> np.ndarray:
        (
            local_x_grid,
            local_y_grid,
            local_z_grid,
            cylinder_r_grid,
            sphere_r_grid,
        ) = self.compute_local_coordinates(x_grid, y_grid, z_grid)
        inside_vortex = self.get_inside_vortex_mask(sphere_r_grid)
        vorticity_mag = (
            inside_vortex
            * (15.0 / 2.0)
            * self.free_stream_velocity
            * cylinder_r_grid
            / self.vortex_radius**2
        )
        grid_size = x_grid.shape
        vorticity = np.zeros((self.grid_dim, *grid_size))
        vorticity[spu.VectorField.x_axis_idx()] = (
            -vorticity_mag * local_y_grid / cylinder_r_grid
        )
        vorticity[spu.VectorField.y_axis_idx()] = (
            vorticity_mag * local_x_grid / cylinder_r_grid
        )
        return vorticity

    def get_velocity(
        self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
    ) -> np.ndarray:
        (
            local_x_grid,
            local_y_grid,
            local_z_grid,
            cylinder_r_grid,
            sphere_r_grid,
        ) = self.compute_local_coordinates(x_grid, y_grid, z_grid)
        inside_vortex = self.get_inside_vortex_mask(sphere_r_grid)
        grid_size = x_grid.shape
        velocity = np.zeros((self.grid_dim, *grid_size))
        radial_velocity = (
            1.5
            * self.free_stream_velocity
            * local_z_grid
            * cylinder_r_grid
            / self.vortex_radius**2
        ) * (
            inside_vortex
            + (1 - inside_vortex) * (self.vortex_radius / sphere_r_grid) ** 5
        )
        velocity[spu.VectorField.x_axis_idx()] = (
            radial_velocity * local_x_grid / cylinder_r_grid
        )
        velocity[spu.VectorField.y_axis_idx()] = (
            radial_velocity * local_y_grid / cylinder_r_grid
        )
        velocity[spu.VectorField.z_axis_idx()] = inside_vortex * (
            -1.5
            * self.free_stream_velocity
            * (2 * np.square(cylinder_r_grid) + np.square(local_z_grid))
            / self.vortex_radius**2
            + 2.5 * self.free_stream_velocity
        ) - (1 - inside_vortex) * (
            (self.vortex_radius / sphere_r_grid) ** 5
            * (np.square(cylinder_r_grid) - 2 * np.square(local_z_grid))
            / 2
            / self.vortex_radius**2
        )
        return velocity

    def get_kinetic_energy(self) -> float:
        kinetic_energy = (
            10.0
            * np.pi
            / 7.0
            * self.free_stream_velocity**2
            * self.vortex_radius**3
        )
        return kinetic_energy

    def get_vortex_stretching(
        self, x_grid: np.ndarray, y_grid: np.ndarray, z_grid: np.ndarray
    ) -> np.ndarray:
        (
            local_x_grid,
            local_y_grid,
            local_z_grid,
            cylinder_r_grid,
            sphere_r_grid,
        ) = self.compute_local_coordinates(x_grid, y_grid, z_grid)
        inside_vortex = self.get_inside_vortex_mask(sphere_r_grid)
        vortex_stretching_mag = (
            inside_vortex
            * (45.0 / 4.0)
            * self.free_stream_velocity**2
            * cylinder_r_grid
            * local_z_grid
            / self.vortex_radius**4
        )
        grid_size = x_grid.shape
        vortex_stretching = np.zeros((self.grid_dim, *grid_size))
        vortex_stretching[spu.VectorField.x_axis_idx()] = (
            -vortex_stretching_mag * local_y_grid / cylinder_r_grid
        )
        vortex_stretching[spu.VectorField.y_axis_idx()] = (
            vortex_stretching_mag * local_x_grid / cylinder_r_grid
        )
        return vortex_stretching
