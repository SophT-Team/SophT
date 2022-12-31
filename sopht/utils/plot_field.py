import matplotlib.pyplot as plt


def create_figure_and_axes(
    fig_aspect_ratio: float | str = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Creates figure and axes for plotting contour fields"""

    plt.style.use("seaborn")
    fig = plt.figure(frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    if fig_aspect_ratio == "default":
        pass
    else:
        ax.set_aspect(aspect=fig_aspect_ratio)
    return fig, ax


def save_and_clear_fig(
    fig: plt.Figure, ax: plt.Axes, cbar: plt.Figure.colorbar = None, file_name: str = ""
) -> None:
    """Save figure and clear for next iteration"""
    fig.savefig(
        file_name,
        bbox_inches="tight",
        pad_inches=0,
    )
    ax.cla()
    if cbar is not None:
        cbar.remove()
