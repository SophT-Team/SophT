from typing import Type
import os
import elastica as ea
from sopht.utils.io import IO


def restart_simulation(
    restart_simulator: Type[ea.BaseSystemCollection],
    io: IO,
    rod_io: IO,
    forcing_io: IO,
    restart_dir: str,
) -> float:
    # find latest saved data
    iter_num = []
    for filename in os.listdir():
        if "sopht" in filename and "h5" in filename:
            iter_num.append(int(filename.split("_")[-1].split(".")[0]))

    if len(iter_num) == 0:
        raise FileNotFoundError("There is no file to load in the directory.")

    latest = max(iter_num)
    # load sopht data
    curr_time = io.load(h5_file_name=f"sopht_{latest:04d}.h5")
    rod_io.load(h5_file_name=f"rod_{latest:04d}.h5")
    forcing_io.load(h5_file_name=f"forcing_grid_{latest:04d}.h5")
    rod_time = ea.load_state(restart_simulator, restart_dir, True)

    if curr_time != rod_time:
        raise ValueError(
            "Simulation time of the flow is not matched with the Elastica, check your inputs!"
        )
    print(f"sopht_{latest:04d}.h5 has been loaded")

    return curr_time
