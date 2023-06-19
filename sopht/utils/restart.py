from typing import Type
import os
import elastica as ea


def restart_simulation(
    flow_sim: Type,
    restart_example_simulator: Type,
    io: Type,
    rod_io: Type,
    forcing_io: Type,
    restart_dir: str,
) -> None:
    # find latest saved data
    iter_num = []
    for filename in os.listdir():
        if "sopht" in filename and "h5" in filename:
            iter_num.append(int(filename.split("_")[-1].split(".")[0]))

    if len(iter_num) == 0:
        print("There is no file to load in the directory")
        return

    latest = max(iter_num)
    # load sopht data
    flow_sim.time = io.load(h5_file_name=f"sopht_{latest:04d}.h5")
    _ = rod_io.load(h5_file_name=f"rod_{latest:04d}.h5")
    _ = forcing_io.load(h5_file_name=f"forcing_grid_{latest:04d}.h5")
    rod_time = ea.load_state(restart_example_simulator, restart_dir, True)

    assert flow_sim.time == rod_time, 'Simulation time of the flow is not matched with the Elastica, check your inputs!'
    print(f"sopht_{latest:04d}.h5 has been loaded")