import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import cm

# Custom map resembling Orange-Blue scheme
res = 256
top = cm.get_cmap("Oranges", 256)
bottom = cm.get_cmap("Blues", 256)

newcolors = np.vstack(
    (top(np.linspace(0.75, 0, 256)), bottom(np.linspace(0, 0.75, 256)))
)
lab_cmap = ListedColormap(newcolors, name="OrangeBlue").reversed()
