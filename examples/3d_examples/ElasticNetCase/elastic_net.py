import elastica as ea
import numpy as np
import sopht.utils as spu
from elastica.experimental.connection_contact_joint.parallel_connection import (
    SurfaceJointSideBySide,
    get_connection_vector_straight_straight_rod,
)
from scipy.interpolate import interp1d


def compute_non_dimensional_rod_positions(
    n_elem: int,
    num_rods_along_perp_axis: int,
    rod_positions_along_perp_axis: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    This function is used to compute the node positions of rods such that element centers of rods are
    at same position as the perpendicular rod element positions.

    Parameters
    ----------
    n_elem : int
        Current rod number of elements.
    num_rods_along_perp_axis : int
        Number of rods that are perpendicular to the current rod position is computed.
    rod_positions_along_perp_axis : numpy.ndarray
        1D numpy.ndarray contains type 'float'
        Start positions of perpendicular rods in the perpendicular axis.

    Returns
    -------

    """
    non_dimensional_length = np.linspace(0, 1, num_rods_along_perp_axis)
    non_dimensional_position_func_for_rod_along_axis = interp1d(
        non_dimensional_length, rod_positions_along_perp_axis
    )
    element_positions = non_dimensional_position_func_for_rod_along_axis(
        np.linspace(0, 1, n_elem)
    )
    positions = np.zeros((n_elem + 1))
    element_length = element_positions[1] - element_positions[0]
    positions[:-1] = element_positions - element_length / 2
    positions[-1] = element_positions[-1] + element_length / 2

    base_length = (positions[1:] - positions[:-1]).sum()

    return positions, base_length


class ElasticNetSimulator:
    def __init__(
        self,
        final_time: float = 2.0,
        rod_density: float = 1e3,
        youngs_modulus: float = 1e5,
        num_rods_along_x: int = 4,
        num_rods_along_y: int = 4,
        gap_between_rods: float = 0.2,
        gap_radius_ratio: float = 10,
        num_rod_elements_per_gap=8,
        elastic_net_origin: np.ndarray = np.array([0.0, 0.0, 0.0]),
        plot_result: bool = True,
    ) -> None:
        class BaseSimulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Connections,
        ):
            ...

        # Set number of elements based on the number of rods
        num_gaps_along_x = num_rods_along_y - 1
        num_gaps_along_y = num_rods_along_x - 1
        # below 1 is needed for correctly connecting rods
        # TODO @armantekinalp please clarify
        n_elem_rods_along_x = 1 + num_gaps_along_y * num_rod_elements_per_gap
        n_elem_rods_along_y = 1 + num_gaps_along_x * num_rod_elements_per_gap

        self.plot_result = plot_result
        self.net_simulator = BaseSimulator()
        # setting up test params
        n_rods = num_rods_along_x + num_rods_along_y
        # Compute base radius of rods
        base_radius = gap_between_rods / gap_radius_ratio
        self.spacing_between_rods = gap_between_rods + 2 * base_radius

        grid_dim = 3
        rod_dim = grid_dim
        x_axis_idx = spu.VectorField.x_axis_idx()
        y_axis_idx = spu.VectorField.y_axis_idx()
        z_axis_idx = spu.VectorField.z_axis_idx()
        start_collection = np.zeros((n_rods, grid_dim))

        # Compute rod start positions, first rods along y then rods along x.
        # We offset rods by 2*base_radius from each other to be able to use surface connections.
        for i in range(num_rods_along_y):
            start_collection[i, y_axis_idx] = i * self.spacing_between_rods
            start_collection[i, z_axis_idx] = base_radius

        for i in range(num_rods_along_x):
            start_collection[i + num_rods_along_y, x_axis_idx] = (
                i * self.spacing_between_rods
            )
            start_collection[i + num_rods_along_y, z_axis_idx] = -base_radius

        # For plotting elastic net
        self.elastic_net_length_x = (
            self.spacing_between_rods * (num_rods_along_x - 1) + 2 * base_radius
        )
        self.elastic_net_length_y = (
            self.spacing_between_rods * (num_rods_along_y - 1) + 2 * base_radius
        )

        # shift the carpet to the provided origin
        self.elastic_net_origin = elastic_net_origin
        start_collection += self.elastic_net_origin.reshape(-1, rod_dim)

        normal = np.array([0.0, 0.0, 1.0])
        poisson_ratio = 0.5
        shear_modulus = youngs_modulus / (1.0 + poisson_ratio)  # Pa
        self.rod_list = []

        # First place the rods that are along the y-axis.
        for i in range(num_rods_along_y):
            start = start_collection[i]
            direction = np.array([1.0, 0.0, 0.0])

            (
                non_dimensional_positions,
                self.base_length_rod_along_y,
            ) = compute_non_dimensional_rod_positions(
                n_elem_rods_along_y,
                num_rods_along_x,
                start_collection[num_rods_along_y:, x_axis_idx],
            )
            # non_dimensional_positions is a 1D vector multiply with direction to convert position_collection.
            positions = direction.reshape(rod_dim, 1) * non_dimensional_positions
            # Position start at correct x but not y, z position. Update the position vector.
            positions += (start - np.dot(start, direction) * direction).reshape(
                rod_dim, 1
            )

            rod = ea.CosseratRod.straight_rod(
                n_elem_rods_along_y,
                positions[..., 0],
                direction,
                normal,
                self.base_length_rod_along_y,
                base_radius,
                rod_density,
                0.0,
                youngs_modulus,
                shear_modulus=shear_modulus,
                position=positions,
            )
            self.net_simulator.append(rod)
            self.rod_list.append(rod)

        # Place the rods that are along the x-axis
        for i in range(num_rods_along_x):
            start = start_collection[i + num_rods_along_y]
            direction = np.array([0.0, 1.0, 0.0])

            (
                non_dimensional_positions,
                self.base_length_rod_along_x,
            ) = compute_non_dimensional_rod_positions(
                n_elem_rods_along_x,
                num_rods_along_y,
                start_collection[:num_rods_along_y, y_axis_idx],
            )
            # non_dimensional_positions is a 1D vector multiply with direction to convert position_collection.
            positions = direction.reshape(rod_dim, 1) * non_dimensional_positions
            # Position start at correct x but not y, z position. Update the position vector.
            positions += (start - np.dot(start, direction) * direction).reshape(
                rod_dim, 1
            )

            rod = ea.CosseratRod.straight_rod(
                n_elem_rods_along_x,
                positions[..., 0],
                direction,
                normal,
                self.base_length_rod_along_x,
                base_radius,
                rod_density,
                0.0,
                youngs_modulus,
                shear_modulus=shear_modulus,
                position=positions,
            )
            self.net_simulator.append(rod)
            self.rod_list.append(rod)

        # Add boundary conditions, one end of rod is clamped
        for i in range(n_rods):
            self.net_simulator.constrain(self.rod_list[i]).using(
                ea.GeneralConstraint,
                constrained_position_idx=(0, -1),
                constrained_director_idx=(0, -1),
            )

        # Add connections
        n_connection = 0
        for rod_one_idx, rod_one in enumerate(self.rod_list):
            for rod_two_idx, rod_two in enumerate(self.rod_list[rod_one_idx + 1 :]):
                for rod_one_elem_idx in range(rod_one.n_elems):
                    for rod_two_elem_idx in range(rod_two.n_elems):
                        (
                            rod_one_direction_vec_in_material_frame,
                            rod_two_direction_vec_in_material_frame,
                            offset_btw_rods,
                        ) = get_connection_vector_straight_straight_rod(
                            rod_one,
                            rod_two,
                            rod_one_idx=(rod_one_elem_idx, rod_one_elem_idx + 1),
                            rod_two_idx=(rod_two_elem_idx, rod_two_elem_idx + 1),
                        )

                        if np.abs(offset_btw_rods) > 1e-13:
                            continue
                        n_connection += 1

                        self.net_simulator.connect(
                            first_rod=rod_one,
                            second_rod=rod_two,
                            first_connect_idx=rod_one_elem_idx,
                            second_connect_idx=rod_two_elem_idx,
                        ).using(
                            SurfaceJointSideBySide,
                            k=1e5,
                            nu=0.1,
                            k_repulsive=1e6,
                            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                                ..., 0
                            ],
                            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                                ..., 0
                            ],
                            offset_btw_rods=offset_btw_rods[..., 0],
                        )

        assert (
            n_connection == num_rods_along_y * num_rods_along_x
        ), "Not all rods are connected, change number of elements of rods along y or along x"

        # add damping
        damping_constant = 0.2
        for i in range(num_rods_along_y):
            dl = self.base_length_rod_along_y / n_elem_rods_along_y
            self.dt = 0.05 * dl
            self.net_simulator.dampen(self.rod_list[i]).using(
                ea.AnalyticalLinearDamper,
                damping_constant=damping_constant,
                time_step=self.dt,
            )

        for i in range(num_rods_along_x):
            dl = self.base_length_rod_along_x / n_elem_rods_along_x
            self.dt = 0.05 * dl
            self.net_simulator.dampen(self.rod_list[i + num_rods_along_y]).using(
                ea.AnalyticalLinearDamper,
                damping_constant=damping_constant,
                time_step=self.dt,
            )

        ################################
        # Gravity remove later
        # Add gravitational forces
        # gravitational_acc = -9.80665
        # for rod in self.rod_list:
        #     self.net_simulator.add_forcing_to(rod).using(
        #         ea.GravityForces, acc_gravity=np.array([0.0, 0., gravitational_acc])
        #     )

        self.final_time = final_time
        self.total_steps = int(self.final_time / self.dt)

        if plot_result:
            self.rendering_fps = 30
            self.rod_post_processing_list: list[dict] = []
            self.add_callback()

        self.timestepper = ea.PositionVerlet()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.timestepper, self.net_simulator
        )

    def finalize(self) -> None:
        self.net_simulator.finalize()

    def add_callback(self) -> None:
        # Add callbacks
        class BeamCallBack(ea.CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict):
                ea.CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system: ea.CosseratRod, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["velocity"].append(
                        system.velocity_collection.copy()
                    )
                    self.callback_params["tangents"].append(system.tangents.copy())

        # Add call back for plotting time history of the rod
        for idx, rod in enumerate(self.rod_list):
            self.rod_post_processing_list.append(ea.defaultdict(list))
            self.net_simulator.collect_diagnostics(rod).using(
                BeamCallBack,
                step_skip=int(1.0 / (self.rendering_fps * self.dt)),
                callback_params=self.rod_post_processing_list[idx],
            )

    def time_step(self, time: float, time_step: float) -> float:
        """Time step the simulator"""
        time = self.do_step(
            self.timestepper,
            self.stages_and_updates,
            self.net_simulator,
            time,
            time_step,
        )
        return time

    def run(
        self,
    ) -> None:
        ea.integrate(
            self.timestepper, self.net_simulator, self.final_time, self.total_steps
        )

        if self.plot_result:
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()
            # Plot the magnetic rod time history
            spu.plot_video_of_rod_surface(
                self.rod_post_processing_list,
                fps=self.rendering_fps,
                step=10,
                x_limits=(
                    self.elastic_net_origin[x_axis_idx]
                    - 1.0 * self.elastic_net_length_x,
                    self.elastic_net_origin[x_axis_idx]
                    + 1.0 * self.elastic_net_length_x,
                ),
                y_limits=(
                    self.elastic_net_origin[y_axis_idx]
                    - 1.0 * self.elastic_net_length_y,
                    self.elastic_net_origin[y_axis_idx]
                    + 1.0 * self.elastic_net_length_y,
                ),
                z_limits=(
                    self.elastic_net_origin[z_axis_idx]
                    - 1.0 * self.elastic_net_length_x,
                    self.elastic_net_origin[z_axis_idx]
                    + 1.0 * self.elastic_net_length_x,
                ),
                vis3D=True,
            )


if __name__ == "__main__":
    elastic_net_sim = ElasticNetSimulator()
    elastic_net_sim.finalize()
    elastic_net_sim.run()
