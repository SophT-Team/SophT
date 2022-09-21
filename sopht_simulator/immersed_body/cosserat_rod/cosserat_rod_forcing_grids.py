__all__ = [
    "CosseratRodNodalForcingGrid",
    "CosseratRodElementCentricForcingGrid",
    "CosseratRodEdgeForcingGrid",
]
from elastica._linalg import _batch_cross, _batch_matvec, _batch_matrix_transpose
from elastica.rod.cosserat_rod import CosseratRod
from elastica.interaction import node_to_element_velocity, elements_to_nodes_inplace
import numpy as np

from sopht_simulator.immersed_body import ImmersedBodyForcingGrid


class CosseratRodNodalForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod nodes"""

    def __init__(self, grid_dim, cosserat_rod: type(CosseratRod)):
        self.num_lag_nodes = cosserat_rod.n_elems + 1
        self.cosserat_rod = cosserat_rod
        super().__init__(grid_dim)
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

    def get_minimum_lagrangian_grid_spacing(self):
        """Get the minimum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        return np.amin(self.cosserat_rod.rest_lengths)


class CosseratRodElementCentricForcingGrid(ImmersedBodyForcingGrid):
    """Class for forcing grid at Cosserat rod element centers"""

    def __init__(self, grid_dim, cosserat_rod: type(CosseratRod)):
        self.num_lag_nodes = cosserat_rod.n_elems
        self.cosserat_rod = cosserat_rod
        super().__init__(grid_dim)

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
        self.velocity_field[...] = (
            self.cosserat_rod.velocity_collection[: self.grid_dim, 1:]
            + self.cosserat_rod.velocity_collection[: self.grid_dim, :-1]
        ) / 2.0

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

    def get_minimum_lagrangian_grid_spacing(self):
        """Get the minimum Lagrangian grid spacing"""
        # estimated distance between consecutive elements
        return np.amin(self.cosserat_rod.rest_lengths)


# Forcing grid implementation for tapered rod
class CosseratRodEdgeForcingGrid(ImmersedBodyForcingGrid):
    """
        Class for forcing grid at Cosserat rod element centers and edges.

    Notes
    -----
        For tapered rods (varying cross-sectional area) and for thicker rods
        (high cross-section area to length ratio) this class has to be used.

    """

    def __init__(self, grid_dim, cosserat_rod: type(CosseratRod)):
        self.cosserat_rod = cosserat_rod
        # 1 for element center 2 for edges
        self.num_lag_nodes = cosserat_rod.n_elems + 2 * cosserat_rod.n_elems
        super().__init__(grid_dim)

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

    def get_minimum_lagrangian_grid_spacing(self):
        """Get the minimum Lagrangian grid spacing"""
        return np.amin([self.cosserat_rod.radius, self.cosserat_rod.rest_lengths])


from elastica._rotations import _get_rotation_matrix

# Forcing grid implementation for tapered rod
class CosseratRodSurfaceForcingGrid(ImmersedBodyForcingGrid):
    """
        Class for forcing grid at Cosserat rod element surface points.

    Notes
    -----
        For the 3D simulations of Cosserat rods this grid can be used.

    """

    def __init__(self, grid_dim, cosserat_rod: type(CosseratRod), grid_density: int):
        self.cosserat_rod = cosserat_rod
        # 1 for element center 2 for edges
        # Number of lagrangian nodes around the surface of element is controlled by grid_density
        self.num_lag_nodes = grid_density * cosserat_rod.n_elems
        super().__init__(grid_dim)

        # self.z_vector = np.repeat(
        #     np.array([0, 0, 1.0]).reshape(3, 1), self.cosserat_rod.n_elems, axis=-1
        # )

        # Number of lagrangian grid points per element
        self.grid_density = grid_density
        # Compute the rotational angle for each surface point.
        # Here surface point rotation corresponds to how much d1(normal) vector of element have to be
        # rotated to get a vector pointing from element center to lagrangian grid point.
        # TODO: maybe a better naming ?
        self.surface_point_rotation = np.linspace(
            0, 2 * np.pi, grid_density, endpoint=False
        )

        # Since lag grid points are on the surface, for each node we need to compute moment arm.
        self.moment_arm = np.zeros((3, self.num_lag_nodes))

        self.n_elems = cosserat_rod.n_elems

        # Here we are thinking for each surface point, we have n_elem of them.
        self.start_idx = np.zeros((grid_density), dtype=np.int)
        self.end_idx = np.zeros((grid_density), dtype=np.int)
        self.start_idx[:] = cosserat_rod.n_elems * np.arange(0, self.grid_density)
        self.end_idx[:] = cosserat_rod.n_elems * np.arange(1, self.grid_density + 1)

        # self.element_forces_left_edge_nodes = np.zeros((3, cosserat_rod.n_elems))
        # self.element_forces_right_edge_nodes = np.zeros((3, cosserat_rod.n_elems))

        # We need this temp array just only to be compatible with moment arm during torque calculation.
        # Since if the grid dim is 2 then lagrangian forces has size 2,num_lag_nodes which is not consistent with
        # moment arm
        self.surface_forces = np.zeros((3, self.num_lag_nodes))

        # to ensure position/velocity are consistent during initialisation
        self.compute_lag_grid_position_field()
        self.compute_lag_grid_velocity_field()

    def compute_lag_grid_position_field(self):
        """Computes location of forcing grid for the Cosserat rod"""

        rod_element_position = 0.5 * (
            self.cosserat_rod.position_collection[..., 1:]
            + self.cosserat_rod.position_collection[..., :-1]
        )
        rod_normal_direction = self.cosserat_rod.director_collection[0, :, :]
        rod_tangent_direction = self.cosserat_rod.director_collection[2, :, :]
        for i in range(self.grid_density):
            self.moment_arm[
                :, self.start_idx[i] : self.end_idx[i]
            ] = self.cosserat_rod.radius * _batch_matvec(
                _get_rotation_matrix(
                    self.surface_point_rotation[i], rod_tangent_direction
                ),
                rod_normal_direction,
            )

        # Surface positions are moment_arm + element center position
        self.position_field[:] = self.moment_arm[: self.grid_dim]
        for i in range(self.grid_density):
            self.position_field[
                :, self.start_idx[i] : self.end_idx[i]
            ] += rod_element_position[: self.grid_dim]

        # self.position_field[
        #     :, self.start_idx : self.end_idx
        # ] = rod_element_position[: self.grid_dim]
        #
        # # Rod normal is used to compute the edge points. Rod normal is not necessarily be same as the d1.
        # # Here we also assume rod will always be in XY plane.
        # rod_normal_direction = _batch_cross(self.z_vector, self.cosserat_rod.tangents)
        #
        # # rd1
        # self.moment_arm[:] = rod_normal_direction * self.cosserat_rod.radius
        #
        # # x_elem + rd1
        # self.position_field[
        #     :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        # ] = (rod_element_position + self.moment_arm)[: self.grid_dim]
        #
        # # x_elem - rd1
        # # self.moment_arm_edge_right[:] = -self.moment_arm_edge_left
        # self.position_field[
        #     :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        # ] = (rod_element_position - self.moment_arm)[: self.grid_dim]

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

        # v_elem + omega X moment_arm
        for i in range(self.grid_density):
            self.velocity_field[:, self.start_idx[i] : self.end_idx[i]] = (
                element_velocity
                + _batch_cross(
                    omega_collection,
                    self.moment_arm[:, self.start_idx[i] : self.end_idx[i]],
                )
            )[: self.grid_dim]

        # self.velocity_field[
        #     :, self.start_idx_elems : self.end_idx_elems
        # ] = element_velocity[: self.grid_dim]
        #
        # # v_elem + omega X rd1
        # self.velocity_field[
        #     :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        # ] = (element_velocity + _batch_cross(omega_collection, self.moment_arm))[
        #     : self.grid_dim
        # ]
        #
        # # v_elem - omega X rd1
        # self.velocity_field[
        #     :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        # ] = (element_velocity + _batch_cross(omega_collection, -self.moment_arm))[
        #     : self.grid_dim
        # ]

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
        for i in range(self.grid_density):
            body_flow_forces[: self.grid_dim, 1:] -= (
                0.5 * lag_grid_forcing_field[:, self.start_idx[i], self.end_idx[i]]
            )
            body_flow_forces[: self.grid_dim, :-1] -= (
                0.5 * lag_grid_forcing_field[:, self.start_idx[i], self.end_idx[i]]
            )

        # # negative sign due to Newtons third law
        # body_flow_forces[: self.grid_dim, 1:] -= (
        #     0.5 * lag_grid_forcing_field[:, self.start_idx_elems : self.end_idx_elems]
        # )
        # body_flow_forces[: self.grid_dim, :-1] -= (
        #     0.5 * lag_grid_forcing_field[:, self.start_idx_elems : self.end_idx_elems]
        # )
        self.surface_forces[: self.grid_dim] -= lag_grid_forcing_field[:]

        # torque generated by all lagrangian points are
        lag_grid_torque_field = _batch_cross(self.moment_arm, self.surface_forces)

        for i in range(self.grid_density):
            body_flow_torques[:] += lag_grid_torque_field[
                :, self.start_idx[i] : self.end_idx[:]
            ]

        # # Lagrangian nodes on left edge.
        # self.element_forces_left_edge_nodes[: self.grid_dim] = -lag_grid_forcing_field[
        #     :, self.start_idx_left_edge_nodes : self.end_idx_left_edge_nodes
        # ]
        # # torque from grid forcing
        # body_flow_torques[...] += _batch_cross(
        #     self.moment_arm, self.element_forces_left_edge_nodes
        # )
        #
        # # Lagrangian nodes on right edge.
        # self.element_forces_right_edge_nodes[: self.grid_dim] = -lag_grid_forcing_field[
        #     :, self.start_idx_right_edge_nodes : self.end_idx_right_edge_nodes
        # ]
        # # torque from grid forcing
        # body_flow_torques[...] += _batch_cross(
        #     -self.moment_arm, self.element_forces_right_edge_nodes
        # )
        #
        # # Convert forces on elements to nodes
        # total_element_forces = (
        #     self.element_forces_left_edge_nodes + self.element_forces_right_edge_nodes
        # )
        # elements_to_nodes_inplace(total_element_forces, body_flow_forces)

        # convert global to local frame
        body_flow_torques[...] = _batch_matvec(
            self.cosserat_rod.director_collection,
            body_flow_torques,
        )

    def get_minimum_lagrangian_grid_spacing(self):
        """Get the minimum Lagrangian grid spacing"""
        return np.amin([self.cosserat_rod.radius, self.cosserat_rod.rest_lengths])
