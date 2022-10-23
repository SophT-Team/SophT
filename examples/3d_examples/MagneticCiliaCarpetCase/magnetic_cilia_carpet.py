import elastica as ea
import magneto_pyelastica as mea
import numpy as np
from post_processing import plot_video_with_surface


class MagneticCiliaCarpetSimulator:
    def __init__(
        self,
        rod_base_length=1.5,
        n_elem_per_rod=25,
        num_cycles=2.0,
        num_rods_along_x=8,
        num_rods_along_y=4,
        carpet_base_centroid=np.array([0.0, 0.0, 0.0]),
        plot_result=True,
    ):
        class MagneticBeamSimulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
        ):
            ...

        self.plot_result = plot_result
        self.magnetic_beam_sim = MagneticBeamSimulator()
        # setting up test params
        n_rods = num_rods_along_x * num_rods_along_y
        self.rod_base_length = rod_base_length
        self.spacing_between_rods = self.rod_base_length  # following Gu2020
        n_elem = n_elem_per_rod
        grid_dim = 3
        x_axis = 0
        y_axis = 1
        start_collection = np.zeros((n_rods, grid_dim))
        for i in range(n_rods):
            start_collection[i, x_axis] = (
                i % num_rods_along_x
            ) * self.spacing_between_rods
            start_collection[i, y_axis] = (
                i // num_rods_along_x
            ) * self.spacing_between_rods
        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])
        aspect_ratio = 5.0
        base_radius = rod_base_length / aspect_ratio / 2.0
        base_area = np.pi * base_radius**2
        volume = base_area * self.rod_base_length
        moment_of_inertia = np.pi / 4 * base_radius**4
        density = 2.39e3  # kg/m3
        E = 1.85e5  # Pa
        shear_modulus = 6.16e4  # Pa

        # Parameters are from Gu2020
        angular_frequency = np.deg2rad(
            10.0
        )  # angular frequency of the rotating magnetic field
        self.velocity_scale = self.rod_base_length * angular_frequency
        magnetic_field_strength = 80e-3  # 80mT
        # MBAL2_EI is a non-dimensional number from Wang 2019
        MBAL2_EI = (
            3.82e-5
            * magnetic_field_strength
            * 4e-3
            / (1.85e5 * np.pi / 4 * 0.4e-3**4)
        )  # Magnetization magnitude * B * Length/(EI)
        magnetization_density = (
            MBAL2_EI
            * E
            * moment_of_inertia
            / (volume * magnetic_field_strength * self.rod_base_length)
        )
        self.carpet_length_x = self.spacing_between_rods * (num_rods_along_x - 1)
        self.carpet_length_y = self.spacing_between_rods * (num_rods_along_y - 1)
        spatial_magnetisation_wavelength = self.carpet_length_x
        spatial_magnetisation_phase_diff = np.pi
        magnetization_angle_x = spatial_magnetisation_phase_diff + (
            2 * np.pi * start_collection[..., x_axis] / spatial_magnetisation_wavelength
        )
        magnetization_angle_y = spatial_magnetisation_phase_diff + (
            2 * np.pi * start_collection[..., y_axis] / spatial_magnetisation_wavelength
        )
        self.magnetic_rod_list = []
        magnetization_direction_list = []

        # shift the carpet to the provided centroid
        self.carpet_base_centroid = carpet_base_centroid
        current_carpet_start_centroid = np.mean(start_collection, axis=0)
        start_collection += (
            self.carpet_base_centroid.reshape(-1, grid_dim)
            - current_carpet_start_centroid
        )

        for i in range(n_rods):
            magnetization_direction = (
                np.array(
                    [
                        np.sin(magnetization_angle_x[i]),
                        np.sin(magnetization_angle_y[i]),
                        np.cos(magnetization_angle_x[i])
                        + np.cos(magnetization_angle_y[i]),
                    ]
                ).reshape(3, 1)
                * np.ones(n_elem)
                / np.sqrt(
                    2
                    + 2
                    * np.cos(magnetization_angle_x[i])
                    * np.cos(magnetization_angle_y[i])
                    + 1e-12
                )
            )
            magnetic_rod = ea.CosseratRod.straight_rod(
                n_elem,
                start_collection[i],
                direction,
                normal,
                self.rod_base_length,
                base_radius,
                density,
                0.0,
                E,
                shear_modulus=shear_modulus,
            )
            self.magnetic_beam_sim.append(magnetic_rod)
            self.magnetic_rod_list.append(magnetic_rod)
            magnetization_direction_list.append(magnetization_direction.copy())

        # Add boundary conditions, one end of rod is clamped
        for i in range(n_rods):
            self.magnetic_beam_sim.constrain(self.magnetic_rod_list[i]).using(
                ea.OneEndFixedBC,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
            )

        # Set the constant magnetic field object
        magnetic_field_object = mea.SingleModeOscillatingMagneticField(
            magnetic_field_amplitude=magnetic_field_strength * np.array([1, 1e-2, 1]),
            magnetic_field_angular_frequency=np.array(
                [angular_frequency, 0, angular_frequency]
            ),
            magnetic_field_phase_difference=np.array([0, np.pi / 2, np.pi / 2]),
            ramp_interval=0.01,
            start_time=0.0,
            end_time=5e3,
        )

        # Apply magnetic forces
        for magnetization_direction, magnetic_rod in zip(
            magnetization_direction_list, self.magnetic_rod_list
        ):
            self.magnetic_beam_sim.add_forcing_to(magnetic_rod).using(
                mea.MagneticForces,
                external_magnetic_field=magnetic_field_object,
                magnetization_density=magnetization_density,
                magnetization_direction=magnetization_direction,
                rod_volume=magnetic_rod.volume,
                rod_director_collection=magnetic_rod.director_collection,
            )

        # add damping
        dl = self.rod_base_length / n_elem
        self.dt = 0.1 * dl
        damping_constant = 0.5
        for i in range(n_rods):
            self.magnetic_beam_sim.dampen(self.magnetic_rod_list[i]).using(
                ea.AnalyticalLinearDamper,
                damping_constant=damping_constant,
                time_step=self.dt,
            )

        self.final_time = num_cycles * 2 * np.pi / angular_frequency
        self.total_steps = int(self.final_time / self.dt)

        if plot_result:
            self.rendering_fps = 30
            self.rod_post_processing_list = []
            self.add_callback()

        self.timestepper = ea.PositionVerlet()

    def finalize(self):
        self.magnetic_beam_sim.finalize()

    def add_callback(self):
        # Add callbacks
        class MagneticBeamCallBack(ea.CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict):
                ea.CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system: ea.CosseratRod, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["velocity"].append(
                        system.velocity_collection.copy()
                    )
                    self.callback_params["tangents"].append(system.tangents.copy())

        # Add call back for plotting time history of the rod
        for idx, rod in enumerate(self.magnetic_rod_list):
            self.rod_post_processing_list.append(ea.defaultdict(list))
            self.magnetic_beam_sim.collect_diagnostics(rod).using(
                MagneticBeamCallBack,
                step_skip=int(1.0 / (self.rendering_fps * self.dt)),
                callback_params=self.rod_post_processing_list[idx],
            )

    def run(
        self,
    ):
        ea.integrate(
            self.timestepper, self.magnetic_beam_sim, self.final_time, self.total_steps
        )

        if self.plot_result:
            # Plot the magnetic rod time history
            plot_video_with_surface(
                self.rod_post_processing_list,
                fps=self.rendering_fps,
                step=10,
                x_limits=(
                    self.carpet_base_centroid[0] - 0.6 * self.carpet_length_x,
                    self.carpet_base_centroid[0] + 0.6 * self.carpet_length_x,
                ),
                y_limits=(
                    self.carpet_base_centroid[1] - 0.6 * self.carpet_length_y,
                    self.carpet_base_centroid[1] + 0.6 * self.carpet_length_y,
                ),
                z_limits=(
                    self.carpet_base_centroid[2] - 0.1 * self.rod_base_length,
                    self.carpet_base_centroid[2] + 1.5 * self.rod_base_length,
                ),
                vis3D=True,
            )


if __name__ == "__main__":
    cilia_carpet_sim = MagneticCiliaCarpetSimulator()
    cilia_carpet_sim.finalize()
    cilia_carpet_sim.run()
