import numpy as np
from matplotlib import pyplot as plt


def compute_and_plot_fish_velocity() -> None:
    """
    This function computes and plots the velocity of fish.
    """

    velocity_file_name = "fish_velocity_vs_time.csv"
    position_file_name = "fish_com_position_vs_time.csv"
    period = 1
    velocity_com = np.loadtxt(velocity_file_name, delimiter=",")
    position_com = np.loadtxt(position_file_name, delimiter=",")

    # read corresponding data
    time = position_com[:, 0]
    position = position_com[:, 1:]
    velocity = velocity_com[:, 1:4]
    speed = velocity_com[:, 4]

    # Compute projected velocity
    # Compute rod velocity in rod direction. We need to compute that because,
    # after fish starts to move it chooses an arbitrary direction, which does not
    # have to be initial tangent direction of the rod. Thus, we need to project the
    # fish velocity with respect to its new tangent and roll direction, after that
    # we will get the correct forward and lateral speed. After this projection
    # lateral velocity of the snake has to be oscillating between + and - values with
    # zero mean.

    # Number of steps in one period.
    time_per_period = time / period
    period_step = int(1.0 / (time_per_period[-1] - time_per_period[-2]))
    number_of_period = int(time_per_period[-1])

    # Center of mass position averaged in one period
    center_of_mass_averaged_over_one_period = np.zeros((number_of_period - 2, 3))
    for i in range(1, number_of_period - 1):
        # position of center of mass averaged over one period
        center_of_mass_averaged_over_one_period[i - 1] = np.mean(
            position[(i + 1) * period_step : (i + 2) * period_step]
            - position[(i + 0) * period_step : (i + 1) * period_step],
            axis=0,
        )
    # Average the rod directions over multiple periods and get the direction of the rod.
    direction_of_rod = np.mean(center_of_mass_averaged_over_one_period, axis=0)
    direction_of_rod /= np.linalg.norm(direction_of_rod, ord=2)

    # Compute the projected rod velocity in the direction of the rod
    velocity_mag_in_direction_of_rod = np.einsum("ji,i->j", velocity, direction_of_rod)
    velocity_in_direction_of_rod = np.einsum(
        "j,i->ji", velocity_mag_in_direction_of_rod, direction_of_rod
    )

    # Get the lateral or roll velocity of the rod after subtracting its projected
    # velocity in the direction of rod
    velocity_in_rod_roll_dir = velocity - velocity_in_direction_of_rod

    velocity_mag_in_roll_dir = np.einsum("ji->j", velocity_in_rod_roll_dir)

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(time, position[:, 0], label="x pos")
    axs[0].plot(time, position[:, 1], label="y pos")
    axs[0].plot(time, position[:, 2], label="z pos")
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("fish_com_pos.png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(time, velocity[:, 0], label="x vel")
    axs[0].plot(time, velocity[:, 1], label="y vel")
    axs[0].plot(time, velocity[:, 2], label="z vel")
    axs[0].plot(time, speed, label="speed")
    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("fish_com_vel.png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(time, velocity_mag_in_direction_of_rod, label="axial vel")
    axs[0].plot(time, velocity_mag_in_roll_dir, label="lateral vel")
    plt.tight_layout()
    plt.minorticks_on()
    plt.grid(which="major", color="darkgrey", linestyle="-")
    plt.grid(which="minor", color="lightgrey", linestyle="--")
    fig.align_ylabels()
    fig.legend(prop={"size": 20})
    fig.savefig("fish_com_vel_projected.png")
    plt.close(plt.gcf())

    np.savetxt(
        "velocity_mag_parallel.csv",
        np.vstack((time, velocity_mag_in_direction_of_rod)).T,
        delimiter=",",
    )
    np.savetxt(
        "velocity_mag_perpendicular.csv",
        np.vstack((time, velocity_mag_in_roll_dir)).T,
        delimiter=",",
    )
