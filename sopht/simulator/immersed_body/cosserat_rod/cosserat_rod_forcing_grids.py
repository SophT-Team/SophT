from elastica._linalg import _batch_cross, _batch_matvec, _batch_matrix_transpose
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace
import elastica as ea
import numpy as np
from sopht.simulator.immersed_body import ImmersedBodyForcingGrid
import math
from numba import njit
from typing import List


class CosseratRodNodalForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim: int, cosserat_rod: ea.CosseratRod) -> None:
        num_lag_nodes = cosserat_rod.n_elems + 1
        super().__init__(grid_dim, num_lag_nodes)
        self.cosserat_rod = cosserat_rod
        self.moment_arm = np.zeros((3, cosserat_rod.n_elems))

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = self.cosserat_rod.position_collection[
            : self.grid_dim
        ]

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = self.cosserat_rod.velocity_collection[
            : self.grid_dim
        ]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim] = -lag_grid_forcing_field

        # torque from grid forcing
        self.moment_arm[...] = (
            self.cosserat_rod.position_collection[..., 1:]
            - self.cosserat_rod.position_collection[..., :-1]
        ) / 2.0
        body_flow_torques[...] = _batch_cross(
            self.moment_arm,
            (body_flow_forces[..., 1:] - body_flow_forces[..., :-1]) / 2.0,
        )
        # end element corrections
        body_flow_torques[..., -1] += (
            np.cross(
                self.moment_arm[..., -1],
                body_flow_forces[..., -1],
            )
            / 2.0
        )
        body_flow_torques[..., 0] -= (
            np.cross(
                self.moment_arm[..., 0],
                body_flow_forces[..., 0],
            )
            / 2.0
        )
        # convert global to local frame
        body_flow_torques[...] = _batch_matvec(
            self.cosserat_rod.director_collection,
            body_flow_torques,
        )

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        return np.amax(self.cosserat_rod.lengths)


class CosseratRodElementCentricForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim: int, cosserat_rod: ea.CosseratRod) -> None:
        num_lag_nodes = cosserat_rod.n_elems
        super().__init__(grid_dim, num_lag_nodes)
        self.cosserat_rod = cosserat_rod

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = (
            self.cosserat_rod.position_collection[: self.grid_dim, 1:]
            + self.cosserat_rod.position_collection[: self.grid_dim, :-1]
        ) / 2.0

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = node_to_element_velocity(
            self.cosserat_rod.mass, self.cosserat_rod.velocity_collection
        )[: self.grid_dim]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        body_flow_forces[...] = 0.0
        body_flow_forces[: self.grid_dim, 1:] -= 0.5 * lag_grid_forcing_field
        body_flow_forces[: self.grid_dim, :-1] -= 0.5 * lag_grid_forcing_field

        # torque from grid forcing (don't modify since set = 0 at initialisation)
        # because no torques acting on element centers

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        return np.amax(self.cosserat_rod.lengths)


# Forcing grid implementation for tapered rod
class CosseratRodEdgeForcingGrid(ImmersedBodyForcingGrid):
    """
        Class for forcing grid at Cosserat rod element centers and edges.

    Notes
    -----
        For tapered rods (varying cross-sectional area) and for thicker rods
        (high cross-section area to length ratio) this class has to be used.

    """

    def __init__(self, grid_dim: int, cosserat_rod: ea.CosseratRod) -> None:
        if grid_dim != 2:
            raise ValueError(
                "Invalid grid dimensions. Cosserat rod edge forcing grid is only "
                "defined for grid_dim=2"
            )
        self.cosserat_rod = cosserat_rod
        # 1 for element center 2 for edges
        num_lag_nodes = cosserat_rod.n_elems + 2 * cosserat_rod.n_elems
        super().__init__(grid_dim, num_lag_nodes)

        self.z_vector = np.repeat(
            np.array([0, 0, 1.0]).reshape(3, 1), self.cosserat_rod.n_elems, axis=-1
        )

        self.moment_arm = np.zeros((3, cosserat_rod.n_elems))

        self.start_idx_elems = 0
        self.end_idx_elems = self.start_idx_elems + cosserat_rod.n_elems
        self.start_idx_left_edge_nodes = self.end_idx_elems
        self.end_idx_left_edge_nodes = (
            self.start_idx_left_edge_nodes + cosserat_rod.n_elems
        )
        self.start_idx_right_edge_nodes = self.end_idx_left_edge_nodes
        self.end_idx_right_edge_nodes = (
            self.start_idx_right_edge_nodes + cosserat_rod.n_elems
        )

        self.element_forces_left_edge_nodes = np.zeros((3, cosserat_rod.n_elems))
        self.element_forces_right_edge_nodes = np.zeros((3, cosserat_rod.n_elems))

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""

        rod_element_position = 0.5 * (
            self.cosserat_rod.position_collection[..., 1:]
            + self.cosserat_rod.position_collection[..., :-1]
        )

        self.position_field[
            :, self.start_idx_elems : self.end_idx_elems
        ] = rod_element_position[: self.grid_dim]

        # Rod normal is used to compute the edge points. Rod normal is not necessarily be same as the d1.
        # Here we also assume rod will always be in XY plane.
        rod_normal_direction = _batch_cross(self.z_vector, self.cosserat_rod.tangents)

        # rd1
        self.moment_arm[:] = rod_normal_direction * self.cosserat_rod.radius

        # x_elem + rd1
        self.position_field[
            :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        ] = (rod_element_position + self.moment_arm)[: self.grid_dim]

        # x_elem - rd1
        # self.moment_arm_edge_right[:] = -self.moment_arm_edge_left
        self.position_field[
            :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        ] = (rod_element_position - self.moment_arm)[: self.grid_dim]

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""

        # Element velocity
        element_velocity = node_to_element_velocity(
            self.cosserat_rod.mass, self.cosserat_rod.velocity_collection
        )
        # Element angular velocity
        omega_collection = _batch_matvec(
            _batch_matrix_transpose(self.cosserat_rod.director_collection),
            self.cosserat_rod.omega_collection,
        )

        self.velocity_field[
            :, self.start_idx_elems : self.end_idx_elems
        ] = element_velocity[: self.grid_dim]

        # v_elem + omega X rd1
        self.velocity_field[
            :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        ] = (element_velocity + _batch_cross(omega_collection, self.moment_arm))[
            : self.grid_dim
        ]

        # v_elem - omega X rd1
        self.velocity_field[
            :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        ] = (element_velocity + _batch_cross(omega_collection, -self.moment_arm))[
            : self.grid_dim
        ]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        body_flow_forces[...] = 0.0
        body_flow_torques[...] = 0.0

        # negative sign due to Newtons third law
        body_flow_forces[: self.grid_dim, 1:] -= (
            0.5 * lag_grid_forcing_field[:, self.start_idx_elems : self.end_idx_elems]
        )
        body_flow_forces[: self.grid_dim, :-1] -= (
            0.5 * lag_grid_forcing_field[:, self.start_idx_elems : self.end_idx_elems]
        )

        # Lagrangian nodes on left edge.
        self.element_forces_left_edge_nodes[: self.grid_dim] = -lag_grid_forcing_field[
            :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        ]
        # torque from grid forcing
        body_flow_torques[...] += _batch_cross(
            self.moment_arm, self.element_forces_left_edge_nodes
        )

        # Lagrangian nodes on right edge.
        self.element_forces_right_edge_nodes[: self.grid_dim] = -lag_grid_forcing_field[
            :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        ]
        # torque from grid forcing
        body_flow_torques[...] += _batch_cross(
            -self.moment_arm, self.element_forces_right_edge_nodes
        )

        # Convert forces on elements to nodes
        total_element_forces = (
            self.element_forces_left_edge_nodes + self.element_forces_right_edge_nodes
        )
        elements_to_nodes_inplace(total_element_forces, body_flow_forces)

        # convert global to local frame
        body_flow_torques[...] = _batch_matvec(
            self.cosserat_rod.director_collection,
            body_flow_torques,
        )

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        return np.amax(self.cosserat_rod.lengths)


# Forcing grid implementation for tapered rod
class CosseratRodSurfaceForcingGrid(ImmersedBodyForcingGrid):
    """
        Class for forcing grid at Cosserat rod element surface points.

    Notes
    -----
        For the 3D simulations of Cosserat rods this grid can be used.
        Depending on the maximum radius, grid points between different element can vary.

    """

    def __init__(
        self,
        grid_dim: int,
        cosserat_rod: ea.CosseratRod,
        surface_grid_density_for_largest_element: int,
        with_cap: bool = False,
    ) -> None:
        if grid_dim != 3:
            raise ValueError(
                "Invalid grid dimensions. Cosserat rod surface forcing grid is only "
                "defined for grid_dim=3"
            )
        self.cosserat_rod = cosserat_rod
        self.n_elems = cosserat_rod.n_elems

        # Surface grid density at the arm maximum radius
        self.surface_grid_density_for_largest_element = (
            surface_grid_density_for_largest_element
        )

        # Enable capping of rod ends
        self.with_cap = with_cap

        # Surface grid points scaled between different element based on the largest radius.
        self.surface_grid_points = np.rint(
            self.cosserat_rod.radius[:]
            / np.max(self.cosserat_rod.radius[:])
            * self.surface_grid_density_for_largest_element
        ).astype(int)
        # If there are less than 1 point then set it equal to 1 since we will place it on the element center.
        self.surface_grid_points[np.where(self.surface_grid_points < 3)[0]] = 1

        # store grid point radius ratio for each forcing point
        self.grid_point_radius_ratio = np.ones((self.surface_grid_points.sum()))
        # Generate rotation angle of each grid point
        self.surface_point_rotation_angle_list = []
        for i in range(self.n_elems):
            if self.surface_grid_points[i] > 1:
                self.surface_point_rotation_angle_list.append(
                    np.linspace(
                        0, 2 * np.pi, self.surface_grid_points[i], endpoint=False
                    )
                )
            else:
                # If there is only one point, then that point is on the element center so pass empty array.
                self.surface_point_rotation_angle_list.append(np.array([]))

        # Modify surface grid quantities to account for caps
        if self.with_cap:
            self._update_surface_grid_point_for_caps()

        # finalize number of lag nodes
        num_lag_nodes = self.surface_grid_points.sum()
        super().__init__(grid_dim, num_lag_nodes)

        # Since lag grid points are on the surface, for each node we need to compute moment arm.
        self.moment_arm = np.zeros_like(self.position_field)

        # Depending on the rod taper number of surface points can vary between elements, so we need start_idx and
        # end_idx to slice grid points according to element they belong.
        self.start_idx = np.zeros((self.n_elems), dtype=int)
        self.end_idx = np.zeros((self.n_elems), dtype=int)

        start_idx_temp = 0
        end_idx_temp = 0
        for i in range(self.n_elems):
            self.start_idx[i] = start_idx_temp
            start_idx_temp += self.surface_grid_points[i]
            end_idx_temp += self.surface_grid_points[i]
            self.end_idx[i] = end_idx_temp

        self.local_frame_surface_points = np.zeros_like(self.position_field)

        # Compute surface(grid) point positions on local frame for each element. If there is one grid point then
        # that point is on the element center so just pass 0.0
        for i, surface_point_rotation_angle in enumerate(
            self.surface_point_rotation_angle_list
        ):
            if surface_point_rotation_angle.size == 0:
                # This is true if there is only one point for element, and it is on element center.
                self.local_frame_surface_points[
                    :, self.start_idx[i] : self.end_idx[i]
                ] = 0.0
            else:
                self.local_frame_surface_points[
                    0, self.start_idx[i] : self.end_idx[i]
                ] = np.cos(surface_point_rotation_angle)
                self.local_frame_surface_points[
                    1, self.start_idx[i] : self.end_idx[i]
                ] = np.sin(surface_point_rotation_angle)

        # some caching stuff
        self.rod_director_collection_transpose = np.zeros_like(
            self.cosserat_rod.director_collection
        )
        self.rod_element_position = np.zeros((self.grid_dim, self.n_elems))
        self.rod_element_velocity = np.zeros_like(self.rod_element_position)
        self.rod_element_global_frame_omega = np.zeros_like(self.rod_element_position)

        self.grid_point_director_transpose = np.zeros((3, 3, self.num_lag_nodes))
        self.grid_point_radius = np.zeros((self.num_lag_nodes))
        self.grid_point_omega = np.zeros_like(self.position_field)
        self.lag_grid_torque_field = np.zeros_like(self.position_field)

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""

        self.rod_element_position[...] = 0.5 * (
            self.cosserat_rod.position_collection[..., 1:]
            + self.cosserat_rod.position_collection[..., :-1]
        )

        # Cache rod director collection transpose since it will be used to compute velocity field.
        self.rod_director_collection_transpose[...] = _batch_matrix_transpose(
            self.cosserat_rod.director_collection
        )

        # Broadcast rod properties to the grid points.
        for i in range(self.n_elems):
            self.grid_point_director_transpose[
                :, :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_director_collection_transpose[:, :, i : i + 1]
            self.grid_point_radius[self.start_idx[i] : self.end_idx[i]] = (
                self.cosserat_rod.radius[i]
                * self.grid_point_radius_ratio[self.start_idx[i] : self.end_idx[i]]
            )
            self.position_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_position[:, i : i + 1]

        # Compute the moment arm or distance from the element center for each grid point.
        self.moment_arm[:] = self.grid_point_radius * _batch_matvec(
            self.grid_point_director_transpose, self.local_frame_surface_points
        )

        # Surface positions are moment_arm + element center position
        self.position_field += self.moment_arm

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""

        # Element velocity
        self.rod_element_velocity[...] = node_to_element_velocity(
            self.cosserat_rod.mass, self.cosserat_rod.velocity_collection
        )
        # Element angular velocity
        self.rod_element_global_frame_omega[...] = _batch_matvec(
            self.rod_director_collection_transpose,
            self.cosserat_rod.omega_collection,
        )

        # Broadcast rod properties to the grid points.
        for i in range(self.n_elems):
            self.grid_point_omega[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_global_frame_omega[:, i : i + 1]
            self.velocity_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_velocity[:, i : i + 1]

        # v_elem + omega X moment_arm
        self.velocity_field += _batch_cross(self.grid_point_omega, self.moment_arm)

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        body_flow_forces[...] = 0.0
        body_flow_torques[...] = 0.0

        # negative sign due to Newtons third law
        for i in range(self.n_elems):
            body_forces_on_elems = np.sum(
                lag_grid_forcing_field[:, self.start_idx[i] : self.end_idx[i]], axis=1
            )
            body_flow_forces[:, i] -= 0.5 * body_forces_on_elems
            body_flow_forces[:, i + 1] -= 0.5 * body_forces_on_elems

        # negative sign due to Newtons third law
        # torque generated by all lagrangian points are
        self.lag_grid_torque_field[...] = _batch_cross(
            self.moment_arm, -lag_grid_forcing_field
        )

        # Update body torques
        # convert global to local frame
        for i in range(self.n_elems):
            body_flow_torques[:, i] = self.cosserat_rod.director_collection[
                :, :, i
            ] @ np.sum(
                self.lag_grid_torque_field[:, self.start_idx[i] : self.end_idx[i]],
                axis=1,
            )

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        grid_angular_spacing = 2 * np.pi / self.surface_grid_density_for_largest_element
        return np.amax(
            [self.cosserat_rod.lengths, self.cosserat_rod.radius * grid_angular_spacing]
        )

    def _update_surface_grid_point_for_caps(self):
        """
        Update surface grid point count, grid point radius ratio and grid point angle
        on rod ends to account for caps.
        """
        for rod_end_idx in [0, -1]:
            rod_end_radius = self.cosserat_rod.radius[rod_end_idx]
            # (1) First update surface grid point count for end elements
            if self.surface_grid_points[rod_end_idx] > 1:
                # we can still fit points within the surface grid circle
                # use max(arc-length of angular discretization, rod element length)
                # as reference length for radial discretization
                grid_angular_spacing = (
                    2.0 * np.pi / self.surface_grid_points[rod_end_idx]
                )
                end_elem_surface_grid_radial_spacing = (
                    rod_end_radius * grid_angular_spacing
                )
                # note: we include at least one point on the element for the cap
                end_elem_grid_radial_density = max(
                    int(rod_end_radius // end_elem_surface_grid_radial_spacing), 1
                )
            else:
                # the rod surface grid is already the minimum anyway (single point)
                # no point adding more grid points for end surface
                end_elem_grid_radial_density = 0

            # linearly scale the number of grid points towards outer most surface
            end_elem_surface_grid_points = np.linspace(
                1,
                self.surface_grid_points[rod_end_idx],
                end_elem_grid_radial_density,
                endpoint=False,
            ).astype(int)

            # update surface grid point count at the end elements
            self.surface_grid_points[rod_end_idx] += end_elem_surface_grid_points.sum()

            # After updating the surface grid point count to account for cap surface,
            # we make modifications to grid point radius ratio and grid point angles.
            # (2) Update radius ratio to account for end element cap surface grid points
            idx = 0 if rod_end_idx == 0 else self.grid_point_radius_ratio.shape[0]
            # modify grid point radius ratio array to account for additional grid point
            # for cap surface
            self.grid_point_radius_ratio = np.insert(
                self.grid_point_radius_ratio,
                idx,
                np.ones(end_elem_surface_grid_points.sum()),
            )
            end_elem_grid_points_radius_ratio = (
                np.linspace(
                    0, rod_end_radius, end_elem_grid_radial_density, endpoint=False
                )
                / rod_end_radius
            )
            # start index for grid points within the surface grid point of end element
            # we store the inner grid points at the end portion of the array window
            # corresponding to the element
            start_idx = (
                self.surface_grid_points.cumsum()[rod_end_idx]
                - end_elem_surface_grid_points.sum()
            )
            # loop over each concentric circle and update the radius ratio
            for idx, num_grid_points in enumerate(end_elem_surface_grid_points):
                end_idx = start_idx + num_grid_points
                self.grid_point_radius_ratio[
                    start_idx:end_idx
                ] = end_elem_grid_points_radius_ratio[idx]
                start_idx = end_idx

            # (3) Update grid point rotation angle on rod ends
            if self.surface_grid_points[rod_end_idx] > 1:
                # If there are more than one point on the surface then compute the angle of these points.
                # Surface points are on the local frame
                surface_point_angles_list = []
                # first, include outer most surface point angles
                surface_point_angles_list.extend(
                    np.linspace(
                        0,
                        2 * np.pi,
                        self.surface_grid_points[rod_end_idx]
                        - end_elem_surface_grid_points.sum(),
                        endpoint=False,
                    ).tolist()
                )
                # then append those corresponding to the inner points
                for num_grid_points in end_elem_surface_grid_points:
                    # compute point angles for each concentric circles
                    surface_point_angles_list.extend(
                        np.linspace(
                            0, 2 * np.pi, num_grid_points, endpoint=False
                        ).tolist()
                    )
                surface_point_angles = np.array(surface_point_angles_list)
                self.surface_point_rotation_angle_list[
                    rod_end_idx
                ] = surface_point_angles


# Forcing grid implementation for octopus arm
class OctopusArmSurfaceForcingGrid(ImmersedBodyForcingGrid):
    """
        Class for forcing grid at Octopus arm surface points.

    Notes
    -----
        For the 3D simulations of Octopus arm this grid can be used.
        Depending on the maximum radius, grid points between different element can vary.

    """

    def __init__(
        self,
        memory_block_list: list,
        grid_dim: int,
        rod_list: List[ea.CosseratRod],
        rod_numbers: list,
        surface_grid_density_for_largest_element: int,
        with_cap: bool = False,
    ) -> None:
        if grid_dim != 3:
            raise ValueError(
                "Invalid grid dimensions. Octopus arm surface forcing grid is only "
                "defined for grid_dim=3"
            )
        self.rod_list = rod_list
        self.memory_block = memory_block_list
        self.total_number_of_ring_rods = rod_numbers[0]
        self.total_number_of_straight_rods = rod_numbers[1]
        self.total_number_of_helical_rods = rod_numbers[2]
        self.n_elem_straight_rods = self.rod_list[
            self.total_number_of_ring_rods
        ].n_elems
        self.n_elem_helical_rods = self.rod_list[-1].n_elems
        self.num_elem_ratio = int(
            (self.n_elem_helical_rods + 1) / self.n_elem_straight_rods
        )

        # Surface grid density at the arm maximum radius
        self.surface_grid_density_for_largest_element = (
            surface_grid_density_for_largest_element
        )

        # Enable capping of rod ends
        self.with_cap = with_cap

        # TODO: make a array of initial radius here to be used in next stages
        main_rod = self.rod_list[self.total_number_of_ring_rods]
        self.cosserat_rod = main_rod
        self.initial_arm_radius = np.zeros_like(main_rod.radius)

        self.initial_arm_radius = np.zeros(self.n_elem_straight_rods)
        for i in range(self.n_elem_straight_rods):
            if self.total_number_of_straight_rods == 0:
                self.initial_arm_radius[i] = main_rod.radius[i]
            else:
                self.initial_arm_radius[i] = (
                    main_rod.radius[i]
                    + 2 * self.rod_list[self.total_number_of_ring_rods + 1].radius[i]
                )

            helical_radius = self.rod_list[-1].radius[self.num_elem_ratio * i]
            self.initial_arm_radius[i] += 2 * helical_radius

        # Surface grid points scaled between different element based on the largest radius.
        self.surface_grid_points = np.rint(
            self.initial_arm_radius[:]
            / np.max(self.initial_arm_radius[:])
            * surface_grid_density_for_largest_element
        ).astype(int)

        # If there are less than 1 point then set it equal to 1 since we will place it on the element center.
        self.surface_grid_points[np.where(self.surface_grid_points < 3)[0]] = 1

        # store grid point radius ratio for each forcing point
        self.grid_point_radius_ratio = np.ones((self.surface_grid_points.sum()))

        # Generate rotation angle of each grid point
        self.surface_point_rotation_angle_list = []
        for i in range(self.n_elem_straight_rods):
            if self.surface_grid_points[i] > 1:
                self.surface_point_rotation_angle_list.append(
                    np.linspace(
                        0, 2 * np.pi, self.surface_grid_points[i], endpoint=False
                    )
                )
            else:
                # If there is only one point, then that point is on the element center so pass empty array.
                self.surface_point_rotation_angle_list.append(np.array([]))

        # Modify surface grid quantities to account for caps
        if self.with_cap:
            self._update_surface_grid_point_for_caps()

        # finalize number of lag nodes
        num_lag_nodes = self.surface_grid_points.sum()
        super().__init__(grid_dim, num_lag_nodes)

        # Depending on the rod taper number of surface points can vary between elements, so we need start_idx and
        # end_idx to slice grid points according to element they belong.
        self.start_idx = np.zeros(self.n_elem_straight_rods, dtype=int)
        self.end_idx = np.zeros(self.n_elem_straight_rods, dtype=int)

        start_idx_temp = 0
        end_idx_temp = 0
        for i in range(self.n_elem_straight_rods):
            self.start_idx[i] = start_idx_temp
            start_idx_temp += self.surface_grid_points[i]
            end_idx_temp += self.surface_grid_points[i]
            self.end_idx[i] = end_idx_temp

        for i, surface_point_rotation_angle in enumerate(
            self.surface_point_rotation_angle_list
        ):
            if len(surface_point_rotation_angle) == 0:
                continue
            else:
                surface_point_rotation_angle = np.vstack(
                    (
                        surface_point_rotation_angle,
                        self.grid_point_radius_ratio[
                            self.start_idx[i] : self.end_idx[i]
                        ],
                    )
                )
                self.surface_point_rotation_angle_list[i] = surface_point_rotation_angle

        self.grid_constants, self.grid_indices_all = self.get_grid_indices()

        self.moment_arm = np.zeros_like(self.grid_indices_all, dtype=float)
        self.grid_radius = np.zeros((self.grid_indices_all.shape[1] - 1))

        self.memory_block_position = self.memory_block[0][0].position_collection
        self.memory_block_radius = self.memory_block[0][0].radius
        self.tangents = self.cosserat_rod.tangents

        self.moment_arm = self.find_grid_points(
            memory_block_position=self.memory_block_position,
            memory_block_radius=self.memory_block_radius,
            grid_indices_all=self.grid_indices_all,
            grid_constants=self.grid_constants,
            tangents=self.tangents,
            initial_arm_radius=self.moment_arm,
            ratio=self.num_elem_ratio,
            total_num_of_straight_rods=self.total_number_of_straight_rods,
            straight_rod_n_elems=self.n_elem_straight_rods,
            helical_rod_n_elems=self.n_elem_helical_rods,
        )

        # some caching stuff
        self.rod_director_collection_transpose = np.zeros_like(
            self.cosserat_rod.director_collection
        )
        self.rod_element_position = np.zeros((self.grid_dim, self.n_elem_straight_rods))
        self.rod_element_velocity = np.zeros_like(self.rod_element_position)
        self.rod_element_global_frame_omega = np.zeros_like(self.rod_element_position)

        self.grid_point_director_transpose = np.zeros((3, 3, self.num_lag_nodes))
        self.grid_point_radius = np.zeros((self.num_lag_nodes))
        self.grid_point_omega = np.zeros_like(self.position_field)
        self.lag_grid_torque_field = np.zeros_like(self.position_field)

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self) -> None:
        """Computes location of forcing grid for the Cosserat rod"""

        self.rod_element_position[...] = 0.5 * (
            self.cosserat_rod.position_collection[..., 1:]
            + self.cosserat_rod.position_collection[..., :-1]
        )

        # Cache rod director collection transpose since it will be used to compute velocity field.
        self.rod_director_collection_transpose[...] = _batch_matrix_transpose(
            self.cosserat_rod.director_collection
        )

        # Broadcast rod properties to the grid points.
        for i in range(self.n_elem_straight_rods):
            self.position_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_position[:, i : i + 1]

        # Surface positions are moment_arm + element center position
        self.position_field += self.moment_arm

    def compute_lag_grid_velocity_field(self) -> None:
        """Computes velocity of forcing grid points for the Cosserat rod"""

        # Element velocity
        self.rod_element_velocity[...] = node_to_element_velocity(
            self.cosserat_rod.mass, self.cosserat_rod.velocity_collection
        )
        # Element angular velocity
        self.rod_element_global_frame_omega[...] = _batch_matvec(
            self.rod_director_collection_transpose,
            self.cosserat_rod.omega_collection,
        )

        # Broadcast rod properties to the grid points.
        for i in range(self.n_elem_straight_rods):
            self.grid_point_omega[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_global_frame_omega[:, i : i + 1]
            self.velocity_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_velocity[:, i : i + 1]

        self.velocity_field += _batch_cross(self.grid_point_omega, self.moment_arm)

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces: np.ndarray,
        body_flow_torques: np.ndarray,
        lag_grid_forcing_field: np.ndarray,
    ) -> None:
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        body_flow_forces[...] = 0.0
        body_flow_torques[...] = 0.0

        # negative sign due to Newtons third law
        for i in range(self.n_elem_straight_rods):
            body_forces_on_elems = np.sum(
                lag_grid_forcing_field[:, self.start_idx[i] : self.end_idx[i]], axis=1
            )
            body_flow_forces[:, i] -= 0.5 * body_forces_on_elems
            body_flow_forces[:, i + 1] -= 0.5 * body_forces_on_elems

        # negative sign due to Newtons third law
        # torque generated by all lagrangian points are
        self.lag_grid_torque_field[...] = _batch_cross(
            self.moment_arm, -lag_grid_forcing_field
        )

        # Update body torques
        # convert global to local frame
        for i in range(self.n_elem_straight_rods):
            body_flow_torques[:, i] = self.cosserat_rod.director_collection[
                :, :, i
            ] @ np.sum(
                self.lag_grid_torque_field[:, self.start_idx[i] : self.end_idx[i]],
                axis=1,
            )

    def get_maximum_lagrangian_grid_spacing(self) -> float:
        """Get the maximum Lagrangian grid spacing"""
        grid_angular_spacing = 2 * np.pi / self.surface_grid_density_for_largest_element
        return np.amax(
            [
                self.cosserat_rod.rest_lengths,
                np.amax(np.linalg.norm(self.moment_arm, axis=0))
                * np.ones(self.cosserat_rod.n_elems)
                * grid_angular_spacing,
            ]
        )

    def _update_surface_grid_point_for_caps(self):
        """
        Update surface grid point count, grid point radius ratio and grid point angle
        on rod ends to account for caps.
        """
        for rod_end_idx in [0, -1]:
            rod_end_radius = self.initial_arm_radius[rod_end_idx]
            # (1) First update surface grid point count for end elements
            if self.surface_grid_points[rod_end_idx] > 1:
                # we can still fit points within the surface grid circle
                # use max(arc-length of angular discretization, rod element length)
                # as reference length for radial discretization
                grid_angular_spacing = (
                    2.0 * np.pi / self.surface_grid_points[rod_end_idx]
                )
                end_elem_surface_grid_radial_spacing = (
                    rod_end_radius * grid_angular_spacing
                )
                # note: we include at least one point on the element for the cap
                end_elem_grid_radial_density = max(
                    int(rod_end_radius // end_elem_surface_grid_radial_spacing), 1
                )
            else:
                # the rod surface grid is already the minimum anyway (single point)
                # no point adding more grid points for end surface
                end_elem_grid_radial_density = 0

            # linearly scale the number of grid points towards outer most surface
            end_elem_surface_grid_points = np.linspace(
                1,
                self.surface_grid_points[rod_end_idx],
                end_elem_grid_radial_density,
                endpoint=False,
            ).astype(int)

            # update surface grid point count at the end elements
            self.surface_grid_points[rod_end_idx] += end_elem_surface_grid_points.sum()

            # After updating the surface grid point count to account for cap surface,
            # we make modifications to grid point radius ratio and grid point angles.
            # (2) Update radius ratio to account for end element cap surface grid points
            idx = 0 if rod_end_idx == 0 else self.grid_point_radius_ratio.shape[0]
            # modify grid point radius ratio array to account for additional grid point
            # for cap surface
            self.grid_point_radius_ratio = np.insert(
                self.grid_point_radius_ratio,
                idx,
                np.ones(end_elem_surface_grid_points.sum()),
            )
            end_elem_grid_points_radius_ratio = (
                np.linspace(
                    0, rod_end_radius, end_elem_grid_radial_density, endpoint=False
                )
                / rod_end_radius
            )
            # start index for grid points within the surface grid point of end element
            # we store the inner grid points at the end portion of the array window
            # corresponding to the element
            start_idx = (
                self.surface_grid_points.cumsum()[rod_end_idx]
                - end_elem_surface_grid_points.sum()
            )
            # loop over each concentric circle and update the radius ratio
            for idx, num_grid_points in enumerate(end_elem_surface_grid_points):
                end_idx = start_idx + num_grid_points
                self.grid_point_radius_ratio[
                    start_idx:end_idx
                ] = end_elem_grid_points_radius_ratio[idx]
                start_idx = end_idx

            # (3) Update grid point rotation angle on rod ends
            if self.surface_grid_points[rod_end_idx] > 1:
                # If there are more than one point on the surface then compute the angle of these points.
                # Surface points are on the local frame
                surface_point_angles_list = []
                # first, include outer most surface point angles
                surface_point_angles_list.extend(
                    np.linspace(
                        0,
                        2 * np.pi,
                        self.surface_grid_points[rod_end_idx]
                        - end_elem_surface_grid_points.sum(),
                        endpoint=False,
                    ).tolist()
                )
                # then append those corresponding to the inner points
                for num_grid_points in end_elem_surface_grid_points:
                    # compute point angles for each concentric circles
                    surface_point_angles_list.extend(
                        np.linspace(
                            0, 2 * np.pi, num_grid_points, endpoint=False
                        ).tolist()
                    )
                surface_point_angles = np.array(surface_point_angles_list)
                self.surface_point_rotation_angle_list[
                    rod_end_idx
                ] = surface_point_angles

    def get_grid_indices(self):
        total_number_of_helical_rods = (
            len(self.rod_list)
            - self.total_number_of_ring_rods
            - self.total_number_of_straight_rods
            - 1
        )
        num_lag_nodes = self.surface_grid_points.sum()

        grid_indices_all = np.zeros((3, num_lag_nodes))
        grid_constants = np.zeros((3, num_lag_nodes))
        helical_angles = np.zeros((total_number_of_helical_rods))

        for i, surface_point_rotation_angle in enumerate(
            self.surface_point_rotation_angle_list
        ):
            if len(surface_point_rotation_angle) == 0:
                grid_constants[:, self.start_idx[i] : self.end_idx[i]] = np.zeros(
                    (3, 1)
                )
                grid_indices_all[:, self.start_idx[i] : self.end_idx[i]] = np.ones(
                    (3, 1)
                )
                grid_indices_all[2, self.start_idx[i]] = i
                grid_indices_all = grid_indices_all.astype(int)
            else:
                sorting_angles = np.argsort(
                    surface_point_rotation_angle[0], kind="stable"
                )
                self.surface_point_rotation_angle_list[
                    i
                ] = surface_point_rotation_angle[:, sorting_angles]
                for k in range(1, total_number_of_helical_rods + 1):
                    helical_rod_position = self.rod_list[-k].position_collection[
                        :, self.num_elem_ratio * i
                    ]
                    helical_angles[-k] = (
                        math.atan2(helical_rod_position[2], helical_rod_position[0])
                        % (2 * np.pi)
                        % (2 * np.pi)
                    )

                sorted_indices = np.argsort(helical_angles, kind="stable")
                helical_angles = helical_angles[sorted_indices]
                helical_angles_stack = np.vstack(
                    (
                        np.ones_like(helical_angles),
                        helical_angles,
                        np.zeros_like(helical_angles),
                        sorted_indices,
                    )
                )
                surface_point_rotation_angle = np.vstack(
                    (
                        np.zeros_like(surface_point_rotation_angle[0]),
                        surface_point_rotation_angle,
                        -1 * np.ones_like(surface_point_rotation_angle[0]),
                    )
                )

                all_angles = np.hstack(
                    (helical_angles_stack, surface_point_rotation_angle)
                )
                sorted_indices = np.argsort(all_angles[1], kind="stable")
                all_angles = all_angles[:, sorted_indices]

                grid_indices = np.where(all_angles[0] == 0)[0]
                rod_indices = np.where(all_angles[0] == 1)[0]

                before_index = np.zeros_like(grid_indices)
                after_index = np.zeros_like(grid_indices)

                for idx, grid_idx in enumerate(grid_indices):
                    before_indices = rod_indices[rod_indices < grid_idx]
                    after_indices = rod_indices[rod_indices > grid_idx]
                    if before_indices.size == 0:
                        before_index[idx] = after_indices[-1]
                    else:
                        before_index[idx] = before_indices[-1]
                    if after_indices.size == 0:
                        after_index[idx] = before_indices[0]
                    else:
                        after_index[idx] = after_indices[0]

                angle_diff = (
                    all_angles[1, grid_indices] - all_angles[1, before_index]
                ) / (all_angles[1, after_index] - all_angles[1, before_index])
                grid_angles = all_angles[1, grid_indices] - all_angles[1, before_index]
                first_one_idx = np.argmax(all_angles[0, :] == 1)
                last_one_idx = (
                    all_angles[0, :].size - np.argmax(all_angles[0, :][::-1] == 1) - 1
                )
                zeros_at_first_idx = np.where(all_angles[0, :][:first_one_idx] == 0)[0]
                zeros_at_last_idx = (
                    np.where(all_angles[0, :][last_one_idx + 1 :] == 0)[0]
                    + last_one_idx
                    + 1
                )
                all_angles[1, rod_indices[-1]] += -2 * np.pi

                for idx in zeros_at_first_idx:
                    angle_diff[idx] = (
                        all_angles[1, idx] - all_angles[1, before_index[idx]]
                    ) / (
                        all_angles[1, after_index[idx]]
                        - all_angles[1, before_index[idx]]
                    )
                    grid_angles[idx] = (
                        all_angles[1, idx] - all_angles[1, before_index[idx]]
                    )
                for idx in zeros_at_last_idx:
                    all_angles[1, idx] += -2 * np.pi
                    angle_diff[idx - rod_indices.size] = (
                        all_angles[1, idx]
                        - all_angles[1, before_index[idx - rod_indices.size]]
                    ) / (
                        all_angles[1, after_index[idx - rod_indices.size]]
                        - all_angles[1, before_index[idx - rod_indices.size]]
                    )
                    grid_angles[idx - rod_indices.size] = (
                        all_angles[1, idx]
                        - all_angles[1, before_index[idx - rod_indices.size]]
                    )

                grid_constants[:, self.start_idx[i] : self.end_idx[i]] = np.vstack(
                    (angle_diff, grid_angles, all_angles[2, grid_indices])
                )
                grid_indices_all[:, self.start_idx[i] : self.end_idx[i]] = np.vstack(
                    (
                        all_angles[3, before_index],
                        all_angles[3, after_index],
                        i * np.ones(after_index.shape[0]),
                    )
                )
                grid_indices_all = grid_indices_all.astype(int)

        return grid_constants, grid_indices_all

    @staticmethod
    @njit(cache=True)
    def find_grid_points(
        memory_block_position: np.ndarray,
        memory_block_radius: np.ndarray,
        grid_indices_all: np.ndarray,
        grid_constants: np.ndarray,
        tangents: np.ndarray,
        initial_arm_radius: np.ndarray,
        ratio: int,
        total_num_of_straight_rods: int,
        straight_rod_n_elems: int,
        helical_rod_n_elems: int,
    ):

        first_idx_helical = (total_num_of_straight_rods + 1) * (
            straight_rod_n_elems + 2
        )

        arm_radius = initial_arm_radius
        for i in range(grid_indices_all.shape[1]):
            r_2 = memory_block_position[
                :,
                first_idx_helical
                + grid_indices_all[1, i] * (helical_rod_n_elems + 2)
                + ratio * grid_indices_all[2, i],
            ]
            r_2_in_plane = (
                r_2
                - 0.5
                * (
                    memory_block_position[:, 1 : straight_rod_n_elems + 1]
                    + memory_block_position[:, :straight_rod_n_elems]
                )[:, grid_indices_all[2, i]]
            )
            r_2_norm = (
                np.linalg.norm(r_2_in_plane)
                + memory_block_radius[
                    0,
                    first_idx_helical
                    + grid_indices_all[1, i] * (helical_rod_n_elems + 2)
                    + ratio * grid_indices_all[2, i],
                ]
            )

            r_1 = memory_block_position[
                :,
                first_idx_helical
                + grid_indices_all[0, i] * (helical_rod_n_elems + 2)
                + ratio * grid_indices_all[2, i],
            ]
            r_1_in_plane = (
                r_1
                - 0.5
                * (
                    memory_block_position[:, 1 : straight_rod_n_elems + 1]
                    + memory_block_position[:, :straight_rod_n_elems]
                )[:, grid_indices_all[2, i]]
            )
            r_1_norm = (
                np.linalg.norm(r_1_in_plane)
                + memory_block_radius[
                    0,
                    first_idx_helical
                    + grid_indices_all[0, i] * (helical_rod_n_elems + 2)
                    + ratio * grid_indices_all[2, i],
                ]
            )

            r_norm = (
                (r_2_norm - r_1_norm) * grid_constants[0, i] + r_1_norm
            ) * grid_constants[2, i]

            # Normalize the axis of rotation
            axis = tangents[:, grid_indices_all[2, i]] / np.linalg.norm(
                tangents[:, grid_indices_all[2, i]]
            )

            # Create the rotation matrix using Rodrigues' rotation formula
            cos_theta = np.cos(-grid_constants[1, i])
            sin_theta = np.sin(-grid_constants[1, i])
            cross = np.cross(axis, r_1_in_plane)
            dot = np.dot(axis, r_1_in_plane)

            rotated_r_1_in_plane = (
                r_1_in_plane * cos_theta
                + cross * sin_theta
                + axis * dot * (1 - cos_theta)
            )

            arm_radius[:, i] = (
                r_norm * rotated_r_1_in_plane / np.linalg.norm(rotated_r_1_in_plane)
            )

            if i == grid_indices_all.shape[1] - 1:
                continue

        return arm_radius
