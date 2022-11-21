from sopht.utils.IO import IO
from elastica.rod.cosserat_rod import CosseratRod
import numpy as np


class CosseratRodIO(IO):
    """
    Derived IO class for Cosserat rod IO.
    """

    def __init__(self, cosserat_rod: type(CosseratRod), dim, real_dtype=np.float64):
        super().__init__(dim, real_dtype)
        self.cosserat_rod = cosserat_rod

        # Initialize rod element position
        self.rod_element_position = np.zeros((3, cosserat_rod.n_elems))
        self._update_rod_element_position

        # Add the element position to IO
        self.add_as_lagrangian_fields_for_io(
            lagrangian_grid=self.rod_element_position,
            lagrangian_grid_name="rod",
            scalar_3d=self.cosserat_rod.radius,
            lagrangian_grid_connect=True,
        )

    def save(self, h5_file_name, time=0.0):
        self._update_rod_element_position()
        self._save(h5_file_name=h5_file_name, time=time)

    def _update_rod_element_position(self):
        self.rod_element_position[...] = 0.5 * (
            self.cosserat_rod.position_collection[..., 1:]
            + self.cosserat_rod.position_collection[..., :-1]
        )


def make_dir_and_transfer_h5_data(dir_name: str, clean_dir: bool = True):
    """Makes a new directory and transfers h5 flow data files to the directory"""
    import os

    if clean_dir:
        os.system(f"rm -rf {dir_name}")
    os.system(f"mkdir {dir_name}")
    os.system(f"mv *.xmf *.h5 {dir_name}")
