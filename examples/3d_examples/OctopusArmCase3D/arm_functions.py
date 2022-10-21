import elastica as ea
import numpy as np
from numba import njit
from elastica._linalg import _batch_norm


# Sigmoid activation functions
class SigmoidActivationLongitudinalMuscles:
    def __init__(
        self,
        beta,
        tau,
        start_time,
        end_time,
        start_non_dim_length,
        end_non_dim_length,
        n_elems,
        activation_level_max=1.0,
        activation_level_end=0.0,
        activation_lower_threshold=2e-3,
    ):
        self.beta = beta
        self.tau = tau/n_elems
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = int(np.rint(start_non_dim_length*n_elems))
        self.end_idx = int(np.rint(end_non_dim_length*n_elems))

        self.activation_level_max = activation_level_max
        self.activation_level_end = activation_level_end
        self.activation_lower_threshold = activation_lower_threshold

    def apply_activation(self, system, activation, time: np.float64 = 0.0):
        n_elems = self.end_idx - self.start_idx
        index = np.arange(0, n_elems, dtype=np.int64)
        fiber_activation = np.zeros((n_elems))
        activation *= 0

        time = round(time, 5)
        if time > self.start_time - 4 * self.tau / self.beta:
            fiber_activation = (
                self.activation_level_max
                * 0.5
                * (
                    1
                    + np.tanh(
                        self.beta * ((time - self.start_time) / self.tau - index + 0)
                    )
                )
            ) + (
                -(self.activation_level_max - self.activation_level_end)
                * (
                    0.5
                    * (
                        1
                        + np.tanh(
                            self.beta * ((time - self.end_time) / self.tau - index + 0)
                        )
                    )
                )
            )
        active_index = np.where(fiber_activation > self.activation_lower_threshold)[0]
        activation[self.start_idx + active_index] = fiber_activation[active_index]


class LocalActivation:
    def __init__(
        self,
        ramp_interval,
        ramp_up_time,
        ramp_down_time,
        start_idx,
        end_idx,
        activation_level=1.0,
    ):
        self.ramp = ramp_interval
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.activation_level = activation_level
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def apply_activation(self, system, activation, time: np.float64 = 0.0):

        time = round(time, 5)
        factor = 0.0
        if (time - self.ramp_up_time) <= 0:
            factor = 0.0
        elif (time - self.ramp_up_time) > 0 and (time - self.ramp_up_time) <= self.ramp:
            factor = (
                1 + np.sin(np.pi * (time - self.ramp_up_time) / self.ramp - np.pi / 2)
            ) / 2
        elif (time - self.ramp_up_time) > 0 and (time - self.ramp_down_time) < 0:
            factor = 1.0

        elif (time - self.ramp_down_time) > 0 and (
            time - self.ramp_down_time
        ) / self.ramp < 1.0:
            factor = (
                1
                - (
                    1
                    + np.sin(
                        np.pi * (time - self.ramp_down_time) / self.ramp - np.pi / 2
                    )
                )
                / 2
            )

        fiber_activation = self.activation_level * factor
        if fiber_activation > 0.0:
            activation[self.start_idx : self.end_idx] = fiber_activation

    # Drag force


from elastica._linalg import _batch_dot
from elastica.interaction import elements_to_nodes_inplace


class DragForceOnStraightRods(ea.NoForces):
    def __init__(self, cd_perpendicular, cd_tangent, rho_water, start_time=0.0):
        self.cd_perpendicular = cd_perpendicular
        self.cd_tangent = cd_tangent
        self.rho_water = rho_water
        self.start_time = start_time

    def apply_forces(self, system, time: np.float64 = 0.0):
        if time > self.start_time:
            self._apply_forces(
                self.cd_perpendicular,
                self.cd_tangent,
                self.rho_water,
                system.radius,
                system.lengths,
                system.tangents,
                system.velocity_collection,
                system.external_forces,
            )

    @staticmethod
    @njit(cache=True)
    def _apply_forces(
        cd_perpendicular,
        cd_tangent,
        rho_water,
        radius,
        lengths,
        tangents,
        velocity_collection,
        external_forces,
    ):
        projected_area = 2 * radius * lengths
        surface_area = np.pi * projected_area

        element_velocity = 0.5 * (
            velocity_collection[:, 1:] + velocity_collection[:, :-1]
        )

        tangent_velocity = _batch_dot(element_velocity, tangents) * tangents
        perpendicular_velocity = element_velocity - tangent_velocity

        tangent_velocity_mag = _batch_norm(tangent_velocity)
        perpendicular_velocity_mag = _batch_norm(perpendicular_velocity)

        forces_in_tangent_dir = (
            0.5
            * rho_water
            * surface_area
            * cd_tangent
            * tangent_velocity_mag
            * tangent_velocity
        )
        forces_in_perpendicular_dir = (
            0.5
            * rho_water
            * projected_area
            * cd_perpendicular
            * perpendicular_velocity_mag
            * perpendicular_velocity
        )

        elements_to_nodes_inplace(-forces_in_tangent_dir, external_forces)
        elements_to_nodes_inplace(-forces_in_perpendicular_dir, external_forces)


# Call back functions
class StraightRodCallBack(ea.CallBackBaseClass):
    """
    Call back function for two arm octopus
    """

    def __init__(self, step_skip: int, callback_params: dict):
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.callback_params["time"].append(time)
            self.callback_params["step"].append(current_step)
            self.callback_params["position"].append(system.position_collection.copy())
            self.callback_params["velocity"].append(system.velocity_collection.copy())
            self.callback_params["radius"].append(system.radius.copy())
            self.callback_params["com"].append(system.compute_position_center_of_mass())
            self.callback_params["external_forces"].append(
                system.external_forces.copy()
            )
            self.callback_params["internal_forces"].append(
                system.internal_forces.copy()
            )
            self.callback_params["tangents"].append(system.tangents.copy())
            self.callback_params["internal_stress"].append(
                system.internal_stress.copy()
            )
            self.callback_params["dilatation"].append(system.dilatation.copy())
            if current_step == 0:
                self.callback_params["lengths"].append(system.rest_lengths.copy())
            else:
                self.callback_params["lengths"].append(system.lengths.copy())

            # self.callback_params["activation"].append(system.fiber_activation.copy())
            self.callback_params["kappa"].append(system.kappa.copy())

            self.callback_params["directors"].append(system.director_collection.copy())

            self.callback_params["sigma"].append(system.sigma.copy())

            return


class CylinderCallBack(ea.CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        self.every = step_skip
        self.callback_params = callback_params

    def save_params(self, system, time):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius)
        self.callback_params["height"].append(system.length)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())
