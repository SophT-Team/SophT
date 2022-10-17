from elastica._linalg import _batch_cross, _batch_matvec, _batch_matrix_transpose
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace
import elastica as ea
import numpy as np
from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class CosseratRodNodalForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim: int, cosserat_rod: type(ea.CosseratRod)):
        num_lag_nodes = cosserat_rod.n_elems + 1
        super().__init__(grid_dim, num_lag_nodes)
        self.cosserat_rod = cosserat_rod
        self.moment_arm = np.zeros((3, cosserat_rod.n_elems))

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = self.cosserat_rod.position_collection[
            : self.grid_dim
        ]

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = self.cosserat_rod.velocity_collection[
            : self.grid_dim
        ]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
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

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        return np.amax(self.cosserat_rod.lengths)


class CosseratRodElementCentricForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim: int, cosserat_rod: type(ea.CosseratRod)):
        num_lag_nodes = cosserat_rod.n_elems
        super().__init__(grid_dim, num_lag_nodes)
        self.cosserat_rod = cosserat_rod

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""
        self.position_field[...] = (
            self.cosserat_rod.position_collection[: self.grid_dim, 1:]
            + self.cosserat_rod.position_collection[: self.grid_dim, :-1]
        ) / 2.0

    def compute_lag_grid_velocity_field(self):
        """Computes velocity of forcing grid points for the Cosserat rod"""
        self.velocity_field[...] = node_to_element_velocity(
            self.cosserat_rod.mass, self.cosserat_rod.velocity_collection
        )[: self.grid_dim]

    def transfer_forcing_from_grid_to_body(
        self,
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
        """Transfer forcing from lagrangian forcing grid to the cosserat rod"""
        # negative sign due to Newtons third law
        body_flow_forces[...] = 0.0
        body_flow_forces[: self.grid_dim, 1:] -= 0.5 * lag_grid_forcing_field
        body_flow_forces[: self.grid_dim, :-1] -= 0.5 * lag_grid_forcing_field

        # torque from grid forcing (don't modify since set = 0 at initialisation)
        # because no torques acting on element centers

    def get_maximum_lagrangian_grid_spacing(self):
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

    def __init__(self, grid_dim: int, cosserat_rod: type(ea.CosseratRod)):
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

    def compute_lag_grid_position_field(self):
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

    def compute_lag_grid_velocity_field(self):
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
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
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

    def get_maximum_lagrangian_grid_spacing(self):
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
        cosserat_rod: type(ea.CosseratRod),
        surface_grid_density_for_largest_element: int,
    ):
        if grid_dim != 3:
            raise ValueError(
                "Invalid grid dimensions. Cosserat rod surface forcing grid is only "
                "defined for grid_dim=3"
            )
        self.cosserat_rod = cosserat_rod

        # Surface grid density at the arm maximum radius
        self.surface_grid_density_for_largest_element = (
            surface_grid_density_for_largest_element
        )

        # Surface grid points scaled between different element based on the largest radius.
        self.surface_grid_points = np.rint(
            self.cosserat_rod.radius[:]
            / np.max(self.cosserat_rod.radius[:])
            * self.surface_grid_density_for_largest_element
        ).astype(int)
        # If there are less than 1 point then set it equal to 1 since we will place it on the element center.
        self.surface_grid_points[np.where(self.surface_grid_points < 3)[0]] = 1
        num_lag_nodes = self.surface_grid_points.sum()
        super().__init__(grid_dim, num_lag_nodes)
        self.n_elems = cosserat_rod.n_elems

        self.surface_point_rotation_angle_list = []
        for i in range(self.n_elems):
            if self.surface_grid_points[i] > 1:
                # If there are more than one point on the surface then compute the angle of these points.
                # Surface points are on the local frame
                self.surface_point_rotation_angle_list.append(
                    np.linspace(
                        0, 2 * np.pi, self.surface_grid_points[i], endpoint=False
                    )
                )
            else:
                # If there is only one point, then that point is on the element center so pass empty array.
                self.surface_point_rotation_angle_list.append(np.array([]))

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

    def compute_lag_grid_position_field(self):
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
            self.grid_point_radius[
                self.start_idx[i] : self.end_idx[i]
            ] = self.cosserat_rod.radius[i]
            self.position_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.rod_element_position[:, i : i + 1]

        # Compute the moment arm or distance from the element center for each grid point.
        self.moment_arm[:] = self.grid_point_radius * _batch_matvec(
            self.grid_point_director_transpose, self.local_frame_surface_points
        )

        # Surface positions are moment_arm + element center position
        self.position_field += self.moment_arm

    def compute_lag_grid_velocity_field(self):
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
        body_flow_forces,
        body_flow_torques,
        lag_grid_forcing_field,
    ):
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

    def get_maximum_lagrangian_grid_spacing(self):
        """Get the maximum Lagrangian grid spacing"""
        grid_angular_spacing = 2 * np.pi / self.surface_grid_density_for_largest_element
        return np.amax(
            [self.cosserat_rod.lengths, self.cosserat_rod.radius * grid_angular_spacing]
        )
