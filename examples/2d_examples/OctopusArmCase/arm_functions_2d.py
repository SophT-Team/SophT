import elastica as ea
import numpy as np


# Sigmoid activation functions
class SigmoidActivationLongitudinalMuscles:
    def __init__(
        self,
        beta: float,
        tau: float,
        start_time: float,
        end_time: float,
        start_idx: int,
        end_idx: int,
        activation_level_max: float = 1.0,
        activation_level_end: float = 0.0,
        activation_lower_threshold: float = 2e-3,
    ) -> None:
        self.beta = beta
        self.tau = tau
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.activation_level_max = activation_level_max
        self.activation_level_end = activation_level_end
        self.activation_lower_threshold = activation_lower_threshold

    def apply_activation(
        self, system, activation: np.ndarray, time: float = 0.0
    ) -> None:
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
        ramp_interval: float,
        ramp_up_time: float,
        ramp_down_time: float,
        start_idx: int,
        end_idx: int,
        activation_level: float = 1.0,
    ) -> None:
        self.ramp = ramp_interval
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.activation_level = activation_level
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def apply_activation(
        self, system, activation: np.ndarray, time: float = 0.0
    ) -> None:

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


# Call back functions
class StraightRodCallBack(ea.CallBackBaseClass):
    """
    Call back function for two arm octopus
    """

    def __init__(self, step_skip: int, callback_params: dict) -> None:
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(
        self, system: ea.CosseratRod, time: float, current_step: int
    ) -> None:
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
    def __init__(self, step_skip: int, callback_params: dict) -> None:
        ea.CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def save_params(self, system: ea.Cylinder, time: float) -> None:
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius)
        self.callback_params["height"].append(system.length)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())
