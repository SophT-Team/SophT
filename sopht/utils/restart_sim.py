import logging
from pathlib import Path

import elastica as ea

from sopht.utils.io import IO

logger = logging.getLogger(__name__)


def restart_simulation(
    restart_simulator: ea.BaseSystemCollection,
    io: IO,
    rod_io: IO,
    forcing_io: IO,
    restart_dir: str,
) -> float:
    # find latest saved data
    iter_num = [int(filename.stem.split("_")[-1]) for filename in Path.cwd().glob("sopht_*.h5")]

    if len(iter_num) == 0:
        msg = "There is no file to load in the directory."
        raise FileNotFoundError(msg)

    latest = max(iter_num)
    # load sopht data
    curr_time = io.load(h5_file_name=f"sopht_{latest:04d}.h5")
    rod_io.load(h5_file_name=f"rod_{latest:04d}.h5")
    forcing_io.load(h5_file_name=f"forcing_grid_{latest:04d}.h5")
    rod_time = ea.load_state(restart_simulator, restart_dir, True)

    if curr_time != rod_time:
        msg = "Simulation time of the flow is not matched with the Elastica, check your inputs!"
        raise ValueError(msg)
    logger.info("sopht_%04d.h5 has been loaded", latest)

    return curr_time
