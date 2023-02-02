import numpy as np
import matplotlib.pyplot as plt
import h5py


if __name__ == "__main__":
    num_rods_along_x = 16
    num_rods_along_y = 8
    womersley = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
    MEratio = [3.0]
    wavelength_x_factors = [0.5, 1.0, 2.0, 4.0]

    domain_size = (7.5, 13.5, 13.5)
    if num_rods_along_x == 16:
        domain_size = (7.5, 13.5, 25.5)

    fig, ax = plt.subplots(nrows=4, ncols=6)
    data_path = "data/data_256"
    fig.set_size_inches(26, 16)
    # fig, ax = plt.subplots()

    for i, mn in enumerate(womersley):
        for j, MBAL2_EI in enumerate(MEratio):
            for k, wlx in enumerate(wavelength_x_factors):
                filename = (
                    f"{data_path}/cilia_{num_rods_along_x}_{num_rods_along_y}_womersley_"
                    + str(mn).replace(".", "pt")
                    + "_MEratio_"
                    + str(MBAL2_EI).replace(".", "pt")
                    + "_wavelength_x_"
                    + str(wlx).replace(".", "pt")
                    + ".h5"
                )
                data = {}
                with h5py.File(filename, "r") as f:
                    avg_field_data = f["Eulerian"]["Vector"]
                    for key in avg_field_data.keys():
                        data[key] = avg_field_data[key][0]
                velocity_x = data["avg_velocity_0"]
                velocity_y = data["avg_velocity_1"]
                velocity_z = data["avg_velocity_2"]
                (grid_size_z, grid_size_y, grid_size_x) = velocity_x.shape
                z = np.linspace(0, domain_size[0], grid_size_z)
                y = np.linspace(0, domain_size[1], grid_size_y)
                x = np.linspace(0, domain_size[2], grid_size_x)

                level = 68
                avg_velocity_x = np.mean(
                    velocity_x[:, int(grid_size_y * 0.25) : int(grid_size_y * 0.75), :],
                    axis=1,
                )
                avg_velocity_z = np.mean(
                    velocity_z[:, int(grid_size_y * 0.25) : int(grid_size_y * 0.75), :],
                    axis=1,
                )
                ax[k][i].set_title(f"Womersley={mn}, wavelength={wlx}")
                ax[k][i].streamplot(x, z, avg_velocity_x, avg_velocity_z, density=1.5)
                # ax[k][i].streamplot(x, z, velocity_x[:, level, :], velocity_z[:, level, :], density=1.5)
                ax[k][i].axhline(domain_size[0] / 5, color="red")
                # ax.plot(z, velocity_x[:, level, int(-domain_size[2] / 2)], label=f"mn={mn}, m/e={MBAL2_EI}")
    # ax.legend()
    plt.show()
    print(velocity_x.shape)
    # fig.savefig("streamline_spritesheet.png", dpi=150)
