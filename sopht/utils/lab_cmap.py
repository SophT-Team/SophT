import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import cm


def get_lab_cmap(res: int = 256) -> ListedColormap:
    """Returns Custom map resembling Orange-Blue scheme"""
    top = cm.get_cmap("Oranges", res)
    bottom = cm.get_cmap("Blues", res)

    newcolors = np.vstack(
        (top(np.linspace(0.75, 0, res)), bottom(np.linspace(0, 0.75, res)))
    )
    lab_cmap = ListedColormap(newcolors, name="OrangeBlue").reversed()
    return lab_cmap
