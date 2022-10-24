import numpy as np


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
        self.tau = tau / n_elems
        self.start_time = start_time
        self.end_time = end_time
        self.start_idx = int(np.rint(start_non_dim_length * n_elems))
        self.end_idx = int(np.rint(end_non_dim_length * n_elems))

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
