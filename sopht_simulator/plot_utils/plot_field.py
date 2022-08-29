import matplotlib.pyplot as plt


def create_figure_and_axes():
    """Creates figure and axes for plotting contour fields"""
    plt.style.use("seaborn")
    fig = plt.figure(frameon=True, dpi=150)
    ax = fig.add_subplot(111)
    ax.set_aspect(aspect=1)
    return fig, ax


def save_and_clear_fig(fig, ax, cbar=None, file_name=""):
    """Save figure and clear for next iteration"""
    fig.savefig(
        file_name,
        bbox_inches="tight",
        pad_inches=0,
    )
    ax.cla()
    if cbar is not None:
        cbar.remove()
