import numpy as np
from matplotlib import pyplot as plt


class OscillationActivation:
    def __init__(
        self,
        wave_number: float,
        frequency: float,
        phase_shift: float,
        start_time: float,
        end_time: float,
        start_non_dim_length: float,
        end_non_dim_length: float,
        n_elems: int,
        a: float,
        b: float,
        activation_level_max: float = 1.0,
        activation_lower_threshold: float = 2e-14,
    ):

        self.start_time = start_time
        self.end_time = end_time
        self.wave_number = wave_number
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.start_non_dim_length = start_non_dim_length
        self.end_non_dim_length = end_non_dim_length
        self.start_idx = int(np.rint(start_non_dim_length * n_elems))
        self.end_idx = int(np.rint(end_non_dim_length * n_elems))
        self.activation_level_max = activation_level_max
        self.activation_lower_threshold = activation_lower_threshold
        self.a = a
        self.b = b

        self.non_dimensional_length = np.linspace(0, 1, n_elems)

    def apply_activation(self, system, activation, time: float = 0.0):

        n_elems = self.end_idx - self.start_idx
        fiber_activation = np.zeros((n_elems))
        activation *= 0

        time = round(time, 5)

        if time > self.start_time and time < self.end_time:

            sigmoid = 1 / (1 + np.exp(-self.a * (self.frequency * time - self.b)))

            fiber_activation = (
                self.activation_level_max
                * ((np.cos(np.pi * self.non_dimensional_length) + 1) / 2) ** 4
                * np.sin(
                    2 * np.pi * self.wave_number * self.non_dimensional_length
                    - 2 * np.pi * self.frequency * time / 2
                    + 2 * np.pi * self.phase_shift
                )
                ** 2
                * sigmoid
            )

        active_index = np.where(fiber_activation > self.activation_lower_threshold)[0]
        activation[self.start_idx + active_index] = fiber_activation[active_index]


if __name__ == "__main__":

    A = 20
    omega_r = 10
    beta = 3

    t1 = A / (18 * omega_r)
    omega_p = beta * omega_r
    T_r = 61 * A / (30 * omega_r)
    T_p = -3 / (5 * beta) * (beta**2 + 60) * t1
    # For sculling
    T = T_r + T_p

    f_r = 1 / T
    f_p = 1 / T
    X_r = 0
    X_p = 0

    wave_number = 0.05

    n_elem = 50
    T = 4
    time = np.linspace(0, 10 * T, 1000)
    activation = np.zeros((time.shape[0], n_elem))
    start_non_dim_length = 0
    end_non_dim_length = 1

    oscillation_activation = OscillationActivation(
        wave_number=wave_number,
        frequency=1 / T,  # f_p,
        phase_shift=0,  # X_p,
        start_time=0.0,
        end_time=10000,
        start_non_dim_length=start_non_dim_length,
        end_non_dim_length=end_non_dim_length,
        n_elems=n_elem,
        activation_level_max=0.2,
        a=10,
        b=0.5,
    )

    system = None

    for i in range(time.shape[0]):
        oscillation_activation.apply_activation(system, activation[i, :], time[i])

    non_dimensional_length = oscillation_activation.non_dimensional_length.copy()

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(time, activation[:, 0])
    axs[0].set_xlabel("non dim length", fontsize=20)
    axs[0].set_ylabel("activation", fontsize=20)

    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig("oscillation_activation_vs_time.png")
    plt.close(plt.gcf())
