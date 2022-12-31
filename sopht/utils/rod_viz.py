import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from matplotlib.patches import Circle
import matplotlib.animation as animation
from typing import Sequence
from sopht.utils.field import VectorField


def plot_video_of_rod_surface(  # noqa C901
    rods_history: Sequence[dict],
    video_name="video.mp4",
    fps: int = 60,
    step: int = 1,
    **kwargs,
) -> None:
    plt.rcParams.update({"font.size": 22})
    folder_name = kwargs.get("folder_name", "")
    # 2d case <always 2d case for now>
    # simulation time
    sim_time = np.array(rods_history[0]["time"])
    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now

    # Rod info
    def rod_history_unpacker(rod_idx: int, t_idx: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            rods_history[rod_idx]["position"][t_idx],
            rods_history[rod_idx]["radius"][t_idx],
        )

    # Rod center of mass
    def com_history_unpacker(rod_idx: int) -> np.ndarray:
        return rods_history[rod_idx]["com"][time_idx]

    # Generate target sphere data
    sphere_flag = False
    if kwargs.__contains__("sphere_history"):
        sphere_flag = True
        sphere_history = kwargs["sphere_history"]
        n_visualized_spheres = len(sphere_history)  # should be one for now

        def sphere_history_unpacker(
            sph_idx: int, t_idx: int
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
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

    def difference(x: tuple[float, float]) -> float:
        return x[1] - x[0]

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
        ax.set_box_aspect((difference(xlim), difference(ylim), difference(zlim)))
        time_idx = 0
        rod_scatters = []

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            # for reference see
            # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot
            # -axes-scatter-markersize-by-x-scale/48174228#48174228
            scaling_factor = (
                ax.get_window_extent().width / max_axis_length * 72.0 / fig.dpi
            )
            rod_scatters.append(
                ax.scatter(
                    inst_position[VectorField.x_axis_idx()],
                    inst_position[VectorField.y_axis_idx()],
                    inst_position[VectorField.z_axis_idx()],
                    # for circle s = 4/pi*area = 4 * r^2
                    s=4 * (scaling_factor * inst_radius) ** 2,
                )
            )

        if sphere_flag:
            sphere_artists = []
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                scaling_factor = (
                    ax.get_window_extent().width / max_axis_length * 72.0 / fig.dpi
                )
                sphere_artists.append(
                    ax.scatter(
                        sphere_position[VectorField.x_axis_idx()],
                        sphere_position[VectorField.y_axis_idx()],
                        sphere_position[VectorField.z_axis_idx()],
                        # for circle s = 4/pi*area = 4 * r^2
                        s=4 * (scaling_factor * sphere_radius) ** 2,
                    )
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
                            inst_position[VectorField.x_axis_idx()],
                            inst_position[VectorField.y_axis_idx()],
                            inst_position[VectorField.z_axis_idx()],
                        )

                        scaling_factor = (
                            ax.get_window_extent().width
                            / max_axis_length
                            * 72.0
                            / fig.dpi
                        )
                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx]._offsets3d = (
                                sphere_position[VectorField.x_axis_idx()],
                                sphere_position[VectorField.y_axis_idx()],
                                sphere_position[VectorField.z_axis_idx()],
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
        rod_lines = []
        rod_com_lines = []
        rod_scatters = []

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            if not inst_position.shape[1] == inst_radius.shape[0]:
                inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines.append(
                ax.plot(
                    inst_position[VectorField.x_axis_idx()],
                    inst_position[VectorField.y_axis_idx()],
                    "r",
                    lw=0.5,
                )[0]
            )
            inst_com = com_history_unpacker(rod_idx)
            rod_com_lines.append(
                ax.plot(
                    inst_com[VectorField.x_axis_idx()],
                    inst_com[VectorField.y_axis_idx()],
                    "k--",
                    lw=2.0,
                )[0]
            )

            scaling_factor = (
                ax.get_window_extent().width / max_axis_length * 72.0 / fig.dpi
            )
            rod_scatters.append(
                ax.scatter(
                    inst_position[VectorField.x_axis_idx()],
                    inst_position[VectorField.y_axis_idx()],
                    s=4 * (scaling_factor * inst_radius) ** 2,
                )
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (
                        sphere_position[VectorField.x_axis_idx()],
                        sphere_position[VectorField.y_axis_idx()],
                    ),
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

                        rod_lines[rod_idx].set_xdata(
                            inst_position[VectorField.x_axis_idx()]
                        )
                        rod_lines[rod_idx].set_ydata(
                            inst_position[VectorField.y_axis_idx()]
                        )

                        com = com_history_unpacker(rod_idx)
                        rod_com_lines[rod_idx].set_xdata(com[VectorField.x_axis_idx()])
                        rod_com_lines[rod_idx].set_ydata(com[VectorField.y_axis_idx()])
                        rod_scatters[rod_idx].set_offsets(
                            np.vstack(
                                (
                                    inst_position[VectorField.x_axis_idx()],
                                    inst_position[VectorField.y_axis_idx()],
                                )
                            ).T
                        )
                        scaling_factor = (
                            ax.get_window_extent().width
                            / max_axis_length
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[VectorField.x_axis_idx()],
                                sphere_position[VectorField.y_axis_idx()],
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
                inst_position[VectorField.z_axis_idx()],
                inst_position[VectorField.y_axis_idx()],
                "r",
                lw=0.5,
            )[0]
            inst_com = com_history_unpacker(rod_idx)
            rod_com_lines[rod_idx] = ax.plot(
                inst_com[VectorField.z_axis_idx()],
                inst_com[VectorField.y_axis_idx()],
                "k--",
                lw=2.0,
            )[0]

            scaling_factor = (
                ax.get_window_extent().width / max_axis_length * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[VectorField.z_axis_idx()],
                inst_position[VectorField.y_axis_idx()],
                s=4 * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (
                        sphere_position[VectorField.z_axis_idx()],
                        sphere_position[VectorField.y_axis_idx()],
                    ),
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

                        rod_lines[rod_idx].set_xdata(
                            inst_position[VectorField.z_axis_idx()]
                        )
                        rod_lines[rod_idx].set_ydata(
                            inst_position[VectorField.y_axis_idx()]
                        )

                        com = com_history_unpacker(rod_idx)
                        rod_com_lines[rod_idx].set_xdata(com[VectorField.z_axis_idx()])
                        rod_com_lines[rod_idx].set_ydata(com[VectorField.y_axis_idx()])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack(
                                (
                                    inst_position[VectorField.z_axis_idx()],
                                    inst_position[VectorField.y_axis_idx()],
                                )
                            ).T
                        )
                        scaling_factor = (
                            ax.get_window_extent().width
                            / max_axis_length
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[VectorField.z_axis_idx()],
                                sphere_position[VectorField.y_axis_idx()],
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
        scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
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
                inst_position[VectorField.x_axis_idx()],
                inst_position[VectorField.z_axis_idx()],
                "r",
                lw=0.5,
            )[0]
            inst_com = com_history_unpacker(rod_idx)
            rod_com_lines[rod_idx] = ax.plot(
                inst_com[VectorField.x_axis_idx()],
                inst_com[VectorField.z_axis_idx()],
                "k--",
                lw=2.0,
            )[0]

            scaling_factor = (
                ax.get_window_extent().width / max_axis_length * 72.0 / fig.dpi
            )
            rod_scatters[rod_idx] = ax.scatter(
                inst_position[VectorField.x_axis_idx()],
                inst_position[VectorField.z_axis_idx()],
                s=4 * (scaling_factor * inst_radius) ** 2,
            )

        if sphere_flag:
            sphere_artists = [None for _ in range(n_visualized_spheres)]
            for sphere_idx in range(n_visualized_spheres):
                sphere_position, sphere_radius = sphere_history_unpacker(
                    sphere_idx, time_idx
                )
                sphere_artists[sphere_idx] = Circle(
                    (
                        sphere_position[VectorField.x_axis_idx()],
                        sphere_position[VectorField.z_axis_idx()],
                    ),
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

                        rod_lines[rod_idx].set_xdata(
                            inst_position[VectorField.x_axis_idx()]
                        )
                        rod_lines[rod_idx].set_ydata(
                            inst_position[VectorField.z_axis_idx()]
                        )

                        com = com_history_unpacker(rod_idx)
                        rod_com_lines[rod_idx].set_xdata(com[VectorField.x_axis_idx()])
                        rod_com_lines[rod_idx].set_ydata(com[VectorField.z_axis_idx()])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack(
                                (
                                    inst_position[VectorField.x_axis_idx()],
                                    inst_position[VectorField.z_axis_idx()],
                                )
                            ).T
                        )
                        scaling_factor = (
                            ax.get_window_extent().width
                            / max_axis_length
                            * 72.0
                            / fig.dpi
                        )
                        rod_scatters[rod_idx].set_sizes(
                            4 * (scaling_factor * inst_radius) ** 2
                        )

                    if sphere_flag:
                        for sphere_idx in range(n_visualized_spheres):
                            sphere_position, _ = sphere_history_unpacker(
                                sphere_idx, time_idx
                            )
                            sphere_artists[sphere_idx].center = (
                                sphere_position[VectorField.x_axis_idx()],
                                sphere_position[VectorField.z_axis_idx()],
                            )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
