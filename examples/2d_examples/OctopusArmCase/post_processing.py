import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm
from matplotlib.patches import Circle
from typing import Dict, Sequence
from collections import defaultdict


def plot_video_muscle_forces(
    force_postprocessing_dict: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
):
    import matplotlib.animation as manimation

    time = np.array(force_postprocessing_dict["time"])
    force_mag = np.array(force_postprocessing_dict["force_mag"])
    force = np.array(force_postprocessing_dict["force"])
    element_position = np.array(force_postprocessing_dict["element_position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.rcParams.update({"font.size": 20})
    plt.subplot(111)
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(1, time.shape[0], int(step))):
            # ax1 = plt.subplot(2, 2, 1)
            # ax2 = plt.subplot(222, frameon=False)
            # x = activation[time][2]
            force_x = force[time][0]
            force_y = force[time][1]
            position = element_position[time]
            fig.clf()

            plt.subplot(1, 1, 1)
            plt.hlines(force_mag[time], 0.0, position[-1], label="force mag")
            plt.plot(np.hstack((0.0, position)), force_x, "-", label="force x")
            plt.plot(np.hstack((0.0, position)), force_y, "-", label="force y")
            # plt.xlim([0 - margin, 2.5 + margin])

            fig.legend(prop={"size": 16})
            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_video_with_surface(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    folder_name = kwargs.get("folder_name", "")

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # Generate target sphere data
    sphere_flag = False
    if kwargs.__contains__("sphere_history"):
        sphere_flag = True
        sphere_history = kwargs.get("sphere_history")
        n_visualized_spheres = len(sphere_history)  # should be one for now
        sphere_history_unpacker = lambda sph_idx, t_idx: (
            sphere_history[sph_idx]["position"][t_idx],
            sphere_history[sph_idx]["radius"][t_idx],
        )
        # color mapping
        sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    if kwargs.get("vis3D", True):
        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = ax.scatter(
                    sphere_position[0],
                    sphere_position[1],
                    sphere_position[2],
                    s=np.pi * (scaling_factor * sphere_radius) ** 2,
                )
                # sphere_radius,
                # color=sphere_cmap(sphere_idx),)
                ax.add_artist(sphere_artists[sphere_idx])

        # ax.set_aspect("equal")
        video_name_3D = folder_name + "3D_" + video_name

        with writer.saving(fig, video_name_3D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_scatters[rod_idx]._offsets3d = (
                            inst_position[0],
                            inst_position[1],
                            inst_position[2],
                        )

                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx]._offsets3d = (
                                sphere_position[0],
                                sphere_position[1],
                                sphere_position[2],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis2D", True):
        max_axis_length = max(difference(xlim), difference(ylim))
        # The scaling factor from physical space to matplotlib space
        scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[0], inst_position[1], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[0], sphere_position[1]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_xy_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_lines[rod_idx].set_xdata(inst_position[0])
                        rod_lines[rod_idx].set_ydata(inst_position[1])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[0])
                        rod_com_lines[rod_idx].set_ydata(com[1])

                        rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[0],
                                sphere_position[1],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

        # Plot zy
        max_axis_length = max(difference(zlim), difference(ylim))
        # The scaling factor from physical space to matplotlib space
        scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*zlim)
        ax.set_ylim(*ylim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[2], inst_position[1], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[2], inst_com[1], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[2],
                inst_position[1],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[2], sphere_position[1]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_zy_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_lines[rod_idx].set_xdata(inst_position[2])
                        rod_lines[rod_idx].set_ydata(inst_position[1])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[2])
                        rod_com_lines[rod_idx].set_ydata(com[1])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[2], inst_position[1])).T
                        )
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[2],
                                sphere_position[1],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

        # Plot xz
        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*zlim)

        # The scaling factor from physical space to matplotlib space
        max_axis_length = max(difference(zlim), difference(xlim))
        scaling_factor = (2 * 0.1) / (max_axis_length)  # Octopus head dimension
        scaling_factor *= 2.6e3  # Along one-axis

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[0], inst_position[2], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[2], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (sphere_position[0], sphere_position[2]),
                    sphere_radius,
                    color=sphere_cmap(sphere_idx),
                )
                ax.add_artist(sphere_artists[sphere_idx])

        ax.set_aspect("equal")
        video_name_2D = folder_name + "2D_xz_" + video_name

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        if not inst_position.shape[1] == inst_radius.shape[0]:
                            inst_position = 0.5 * (
                                inst_position[..., 1:] + inst_position[..., :-1]
                            )

                        rod_lines[rod_idx].set_xdata(inst_position[0])
                        rod_lines[rod_idx].set_ydata(inst_position[2])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[0])
                        rod_com_lines[rod_idx].set_ydata(com[2])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[0], inst_position[2])).T
                        )
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[0],
                                sphere_position[2],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())


def plot_video_activation_muscle(
    pressure_profile_recorder: dict,
    video_name="pressure_force.mp4",
    margin=0.2,
    fps=20,
    step=1,
    dpi=100,
    **kwargs,
):
    import matplotlib.animation as manimation

    time = np.array(pressure_profile_recorder["time"])
    pressure_mag = np.array(pressure_profile_recorder["pressure_mag"])
    external_force = np.array(pressure_profile_recorder["external_forces"])
    element_position = np.array(pressure_profile_recorder["element_position"])
    control_points = np.array(pressure_profile_recorder["control_points"])

    max_pressure = np.max(pressure_mag)
    min_pressure = np.min(pressure_mag)
    max_element_pos = np.max(element_position)
    min_element_pos = np.min(element_position)
    max_force = np.max(external_force)
    min_force = np.min(external_force)

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    plt.rcParams.update({"font.size": 22})
    axs = []
    axs.append(plt.subplot2grid((2, 1), (0, 0)))
    axs.append(plt.subplot2grid((2, 1), (1, 0)))

    # pressure_lines = axs[0].plot(
    #     element_position[0], pressure_mag[0, :], "-", linewidth=3,
    # )[0]
    control_points_lines = axs[0].plot(
        control_points[0][0], control_points[0][1], "*", markersize=20
    )[0]
    axs[0].set_xlim(min_element_pos * -0.1 - 0.2, max_element_pos * 1.10)
    axs[0].set_ylim(min_pressure * 1.15, max_pressure * 1.15)
    axs[0].set_ylabel("pressure mag", fontsize=20)

    force_lines = [None for _ in range(3)]
    for i in range(3):
        force_lines[i] = axs[1].plot(
            np.hstack((0.0, element_position[0])),
            external_force[0][i],
            "-",
            linewidth=3,
        )[0]

    axs[1].set_xlim(min_element_pos * -0.1 - 0.2, max_element_pos * 1.10)
    axs[1].set_ylim(min_force * 1.15 - 5, max_force * 1.15 + 5)
    axs[1].set_ylabel("pressure force", fontsize=20)
    axs[1].set_xlabel("position", fontsize=20)

    plt.tight_layout()
    fig.align_ylabels()

    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(0, time.shape[0], int(step))):

            # pressure_lines.set_xdata(element_position[time])
            # pressure_lines.set_ydata(pressure_mag[time, :])
            control_points_lines.set_xdata(control_points[time][0])
            control_points_lines.set_ydata(control_points[time][1])

            for i in range(3):
                force_lines[i].set_xdata(np.hstack((0.0, element_position[time])))
                force_lines[i].set_ydata(external_force[time][i])

            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_initial_and_final_positions(circular_rod_position, rod_initial_positions):
    filename = "circular_rod_position.png"
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((2, 1), (0, 0)))
    axs.append(plt.subplot2grid((2, 1), (1, 0)))
    axs[0].plot(
        circular_rod_position[0],
        circular_rod_position[2],
        linewidth=3,
    )
    axs[0].plot(
        circular_rod_position[..., -1][0],
        circular_rod_position[..., -1][2],
        "*",
        markersize=20,
    )
    axs[0].plot(
        circular_rod_position[..., 10][0],
        circular_rod_position[..., 10][2],
        "*",
        markersize=20,
    )
    axs[0].plot(
        circular_rod_position[..., 20][0],
        circular_rod_position[..., 20][2],
        "*",
        markersize=20,
    )
    axs[0].plot(
        circular_rod_position[..., 30][0],
        circular_rod_position[..., 30][2],
        "*",
        markersize=20,
    )
    axs[0].set_ylabel("circular rod", fontsize=20)

    axs[1].plot(rod_initial_positions[0], rod_initial_positions[2], linewidth=3)
    axs[1].plot(
        rod_initial_positions[..., -1][0],
        rod_initial_positions[..., -1][2],
        "*",
        markersize=20,
    )
    axs[1].plot(
        rod_initial_positions[..., 10][0],
        rod_initial_positions[..., 10][2],
        "*",
        markersize=20,
    )
    axs[1].plot(
        rod_initial_positions[..., 20][0],
        rod_initial_positions[..., 20][2],
        "*",
        markersize=20,
    )
    axs[1].plot(
        rod_initial_positions[..., 30][0],
        rod_initial_positions[..., 30][2],
        "*",
        markersize=20,
    )
    axs[1].set_ylabel("initial configuration", fontsize=20)

    # axs[1].set_xlabel("element", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    # fig.legend(prop={"size": 20})
    # fig.savefig(filename)
    plt.show()
    plt.close(plt.gcf())


def plot_video_internal_force(
    rod_history: dict,
    video_name="internal_force.mp4",
    margin=0.2,
    fps=20,
    step=1,
    dpi=100,
    **kwargs,
):
    import matplotlib.animation as manimation

    time = np.array(rod_history["time"])
    internal_stress_force = np.array(rod_history["internal_stress"])
    external_force = np.array(rod_history["external_forces"])
    # position = np.array(rod_history["element_position"])
    element_position = np.array(rod_history["element_position"])
    strain = np.array(rod_history["strain"])

    max_pressure = np.max(internal_stress_force)
    min_pressure = np.min(internal_stress_force)
    max_element_pos = np.max(element_position)
    min_element_pos = np.min(element_position)
    max_force = np.max(external_force)
    min_force = np.min(external_force)
    max_strain = np.max(strain[:, 2])
    min_strain = np.min(strain[:, 2])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(2, figsize=(10, 10), frameon=True, dpi=dpi)
    plt.rcParams.update({"font.size": 22})
    axs = []
    axs.append(plt.subplot2grid((3, 1), (0, 0)))
    axs.append(plt.subplot2grid((3, 1), (1, 0)))
    axs.append(plt.subplot2grid((3, 1), (2, 0)))

    pressure_lines = axs[0].plot(
        element_position[0],
        internal_stress_force[0, 2],
        "-",
        linewidth=3,
    )[0]
    axs[0].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    axs[0].set_ylim(min_pressure * 1.05, max_pressure * 1.05)
    axs[0].set_ylabel("internal stress mag", fontsize=20)

    force_lines = [None for _ in range(3)]
    for i in range(3):
        force_lines[i] = axs[1].plot(
            element_position[0],
            external_force[0][i],
            "-",
            linewidth=3,
        )[0]

    axs[1].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    axs[1].set_ylim(min_force * 1.01 - 5, max_force * 1.01 + 5)
    axs[1].set_ylabel("external force", fontsize=20)
    axs[1].set_xlabel("position", fontsize=20)

    strain_lines = axs[2].plot(
        element_position[0],
        strain[0, 2],
        "-",
        linewidth=3,
    )[0]
    axs[2].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    axs[2].set_ylim(min_strain * 1.01, max_strain * 1.01)
    axs[2].set_ylabel("strain", fontsize=20)
    axs[2].set_xlabel("position", fontsize=20)

    plt.tight_layout()
    fig.align_ylabels()

    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(1, time.shape[0], int(step))):

            pressure_lines.set_xdata(element_position[time])
            pressure_lines.set_ydata(internal_stress_force[time, 2])

            for i in range(3):
                force_lines[i].set_xdata(element_position[time])
                force_lines[i].set_ydata(external_force[time][i])

            strain_lines.set_xdata(element_position[time])
            strain_lines.set_ydata(strain[time, 2])

            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_tentacle_length_vs_time(
    rod_history, club_length, tentacle_length_exp, tentacle_length_sim
):
    filename = "tentacle_length_vs_time"
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    position = np.array(rod_history["position"])
    time = np.array(rod_history["time"])

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(
        time * 1e3, position[:, 1, -1] + club_length, linewidth=3, label="PyElastica"
    )
    axs[0].plot(
        tentacle_length_sim[:, 0],
        tentacle_length_sim[:, 1],
        "--",
        linewidth=3,
        label="Kier Sim",
    )
    axs[0].plot(
        tentacle_length_exp[:, 0],
        tentacle_length_exp[:, 1],
        "--",
        linewidth=3,
        label="Kier Exp",
    )
    axs[0].set_ylabel("tentacle length [mm]", fontsize=20)
    axs[0].set_xlabel("time [ms]")

    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(loc="upper center", prop={"size": 20})
    fig.savefig(filename + ".png")
    fig.savefig(filename + ".eps")
    fig.savefig(filename + ".svg")
    plt.show()
    plt.close(plt.gcf())


def plot_tentacle_velocity_vs_time(
    rod_history, tentacle_velocity_exp, tentacle_velocity_sim
):
    filename = "tentacle_velocity_vs_time"
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    velocity = np.array(rod_history["velocity"]) * 1e-3  # Convert from mm to m
    time = np.array(rod_history["time"]) * 1e3  # Convert from s to ms

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(time, velocity[:, 1, -1], linewidth=3, label="PyElastica")
    axs[0].plot(
        tentacle_velocity_sim[:, 0],
        tentacle_velocity_sim[:, 1],
        "--",
        linewidth=3,
        label="Kier Sim",
    )
    axs[0].plot(
        tentacle_velocity_exp[:, 0],
        tentacle_velocity_exp[:, 1],
        "--",
        linewidth=3,
        label="Kier Exp",
    )
    axs[0].set_ylabel("tentacle extension velocity [m/s]", fontsize=20)
    axs[0].set_xlabel("time [ms]")
    # axs[0].grid(True)

    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(loc="upper left", prop={"size": 20})
    fig.savefig(filename + ".png")
    fig.savefig(filename + ".eps")
    fig.savefig(filename + ".svg")
    plt.show()
    plt.close(plt.gcf())


def plot_volume_vs_time(rod_history: dict, filename):

    time = np.array(rod_history["time"])
    volume_total = np.array(rod_history["volume_total"])
    elongation = np.array(rod_history["elongation"])
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((3, 1), (0, 0)))
    axs.append(plt.subplot2grid((3, 1), (1, 0)))
    axs.append(plt.subplot2grid((3, 1), (2, 0)))
    axs[0].plot(
        time,
        volume_total,
        linewidth=3,
    )
    axs[0].set_ylabel("volume", fontsize=20)
    axs[1].plot(
        time,
        elongation,
        linewidth=3,
    )
    axs[1].set_ylabel("elongation", fontsize=20)
    axs[2].plot(
        time,
        elongation - 1,
        linewidth=3,
    )
    axs[2].set_ylabel("strain", fontsize=20)

    # axs[1].set_xlabel("element", fontsize=20)
    plt.tight_layout()
    fig.align_ylabels()
    # fig.legend(prop={"size": 20})
    fig.savefig(filename)
    # plt.show()
    plt.close(plt.gcf())


def plot_volumetric_pressure_diagnostics(
    volume_pressure_profile: dict,
    video_name="volume_pressure_diagnostics.mp4",
    file_name="volume_vs_time.png",
    margin=0.2,
    fps=20,
    step=1,
    dpi=100,
    **kwargs,
):
    import matplotlib.animation as manimation

    time = np.array(volume_pressure_profile["time"])
    target_strain = np.array(volume_pressure_profile["target_strain"])
    current_strain = np.array(volume_pressure_profile["current_strain"])
    element_position = np.array(volume_pressure_profile["element_position"])
    pressure_force_on_elements = np.array(
        volume_pressure_profile["pressure_force_on_elements"]
    )
    strain_from_ring_rods = np.array(volume_pressure_profile["strain_from_ring_rods"])

    max_pressure = np.max(pressure_force_on_elements)
    min_pressure = np.min(pressure_force_on_elements)
    max_element_pos = np.max(element_position)
    min_element_pos = np.min(element_position)
    max_strain = np.max(current_strain[:])
    min_strain = np.min(current_strain[:])
    max_target_strain = np.max(target_strain[:])
    min_target_strain = np.max(target_strain[:])

    print("plot video volumetric pressure diagnostics")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure(2, figsize=(10, 10), frameon=True, dpi=dpi)
    plt.rcParams.update({"font.size": 22})
    axs = []
    axs.append(plt.subplot2grid((2, 1), (0, 0)))
    axs.append(plt.subplot2grid((2, 1), (1, 0)))
    # axs.append(plt.subplot2grid((3, 1), (2, 0)))

    pressure_lines = axs[0].plot(
        element_position[0],
        pressure_force_on_elements[0],
        "-",
        linewidth=3,
    )[0]
    axs[0].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    axs[0].set_ylim(min_pressure * 1.05, max_pressure * 1.05)
    axs[0].set_ylabel("pressure force", fontsize=20)

    # force_lines = [None for _ in range(3)]
    # for i in range(3):
    #     force_lines[i] = axs[1].plot(
    #         element_position[0], external_force[0][i], "-", linewidth=3,
    #     )[0]
    #
    # axs[1].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    # axs[1].set_ylim(min_force * 1.01 - 5, max_force * 1.01 + 5)
    # axs[1].set_ylabel("external force", fontsize=20)
    # axs[1].set_xlabel("position", fontsize=20)
    #
    target_strain_lines = axs[1].plot(
        element_position[0], target_strain[0], "-", linewidth=3, label="target"
    )[0]
    current_strain_lines = axs[1].plot(
        element_position[0], current_strain[0], "--", linewidth=3, label="current"
    )[0]
    strain_from_ring_rods_lines = axs[1].plot(
        element_position[0],
        strain_from_ring_rods[0],
        "-.",
        linewidth=3,
        label="ring rods",
    )[0]
    axis_limit_strain = max(
        np.abs(min_strain),
        np.abs(max_strain),
        np.abs(min_target_strain),
        np.abs(max_target_strain),
    )
    axs[1].set_xlim(min_element_pos * -0.1, max_element_pos * 1.01)
    axs[1].set_ylim(
        np.sign(min_strain) * axis_limit_strain * 1.01,
        np.sign(max_strain) * axis_limit_strain * 1.01,
    )
    axs[1].set_ylabel("strain", fontsize=20)
    axs[1].set_xlabel("position", fontsize=20)

    plt.tight_layout()
    fig.align_ylabels()
    fig.legend(prop={"size": 20})

    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(0, time.shape[0], int(step))):

            pressure_lines.set_xdata(element_position[time])
            pressure_lines.set_ydata(pressure_force_on_elements[time])

            # for i in range(3):
            #     force_lines[i].set_xdata(element_position[time])
            #     force_lines[i].set_ydata(external_force[time][i])

            target_strain_lines.set_xdata(element_position[time])
            target_strain_lines.set_ydata(target_strain[time])

            current_strain_lines.set_xdata(element_position[time])
            current_strain_lines.set_ydata(current_strain[time])

            strain_from_ring_rods_lines.set_xdata(element_position[time])
            strain_from_ring_rods_lines.set_ydata(strain_from_ring_rods[time])

            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())

    time = np.array(volume_pressure_profile["time"])
    volume = np.array(volume_pressure_profile["volume"])
    if volume.size != 0:
        fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)
        axs = []
        axs.append(plt.subplot2grid((1, 1), (0, 0)))

        axs[0].plot(
            time,
            volume,
            linewidth=3,
        )
        axs[0].set_ylabel("volume", fontsize=20)

        axs[0].set_xlabel("time", fontsize=20)

        # axs[1].set_xlabel("element", fontsize=20)
        plt.tight_layout()
        fig.align_ylabels()
        # fig.legend(prop={"size": 20})
        fig.savefig(file_name)
        # plt.show()
        plt.close(plt.gcf())
