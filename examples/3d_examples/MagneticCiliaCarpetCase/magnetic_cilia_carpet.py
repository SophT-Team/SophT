import elastica as ea
import magneto_pyelastica as mea
import numpy as np
import sopht.utils as spu


class MagneticCiliaCarpetSimulator:
    """This class initializes a PyElastica simulator with a cilia carpet
    of uniform spacings on a rectangular grid. The rods are initialized
    in the [0, 0, 1] direction, and magnetized in spatially varying
    directions dependent pn on input wavelength. The carpet is subject to
    a rotating magnetic field using magneto-pyelastica, which forms a
    metachronal wave. For details on the setup, see Gu et al. (2020):
    https://www.nature.com/articles/s41467-020-16458-4#Bib1

    :param magnetic_bond_number: Non-dimensional parameter that describes
    the ratio between magnetic forces to elastic forces, defined as
    (m * B * A * L^2) / (E * I), defaults to 3.0
    :type magnetic_bond_number: float
    :param frequency_ratio: Ratio between magnetic field angular frequency
    and the first bending mode of the rod as a cantilever, defaults to 0.2
    :type frequency_ratio: float
    :param rod_base_length: Length of the rod, defaults to 1.5
    :type rod_base_length: float
    :param n_elem_per_rod: Number of element in each rod, defaults to 25
    :type n_elem_per_rod: int
    :param num_cycles: Total number of cycles the magnetic field rotates
    about the carpet for, defaults to 2.0
    :type num_cycles: float
    :param num_rods_along_x: Number of rods in the x direction
    :type: int
    :param num_rods_along_y: Number of rods in the y direction
    :type: int
    :param wavelength_x_factor: Ratio between magnetization wavelength
    and carpet length in the x direction, defaults to 1.0
    :type wavelength_x_factor: float
    :param wavelength_y_factor: Ratio between magnetization wavelength
    and carpet length in the y direction, defaults to 1.0
    :type wavelength_y_factor: float
    :param carpet_spacing_factor: Ratio between spacing between rods and
    rod_base_length, defaults to 1.0
    :type carpet_spacing_factor: float
    :param carpet_base_centroid: Centroid coordinates of carpet, defaults
    to np.array([0.0, 0.0, 0.0])
    :type carpet_base_centroid: np.ndarray
    :param magnetization_2d: `True` if the carpet is magnetized in both x
    and y direction, `False` if magnetized only in x direction. Defaults to
    `False`
    :type magnetization_2d: bool
    :param plot_result: `True` if video output is required, `False` if not.
    Defaults to `True`
    :type plot_result: bool
    """

    def __init__(
        self,
        magnetic_bond_number: float = 3.0,
        frequency_ratio: float = 0.2,
        rod_base_length: float = 1.5,
        n_elem_per_rod: int = 25,
        num_cycles: float = 2.0,
        num_rods_along_x: int = 8,
        num_rods_along_y: int = 4,
        wavelength_x_factor: float = 1.0,
        wavelength_y_factor: float = 1.0,
        carpet_spacing_factor: float = 1.0,
        carpet_base_centroid: np.ndarray = np.array([0.0, 0.0, 0.0]),
        magnetization_2d: bool = False,
        plot_result: bool = True,
    ) -> None:
        """Constructor method"""

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

        # ====================== Setting up test params ====================
        # Key parameters
        self.rod_base_length = rod_base_length
        magnetic_field_strength = 80e-3
        youngs_modulus = 1.85e5
        density = 2.39e3
        aspect_ratio = 5.0
        shear_modulus = 6.16e4
        spatial_magnetisation_phase_diff = np.pi  # Antiplectic

        # Derived parameters
        base_radius = self.rod_base_length / aspect_ratio / 2.0
        base_area = np.pi * base_radius**2
        volume = base_area * self.rod_base_length
        moment_of_inertia = np.pi / 4 * base_radius**4
        first_mode_frequency = (
            3.5160
            / rod_base_length**2
            * np.sqrt(  # rad / s
                youngs_modulus * moment_of_inertia / density / base_area
            )
        )

        self.angular_frequency = frequency_ratio * first_mode_frequency
        self.spacing_between_rods = self.rod_base_length * carpet_spacing_factor
        self.period = 2.0 * np.pi / np.abs(self.angular_frequency)
        self.carpet_length_x = self.spacing_between_rods * (num_rods_along_x - 1)
        self.carpet_length_y = self.spacing_between_rods * (num_rods_along_y - 1)

        # ==================== Setting up the carpet grid ===================
        n_rods = num_rods_along_x * num_rods_along_y
        n_elem = n_elem_per_rod
        grid_dim = 3
        x_axis_idx = spu.VectorField.x_axis_idx()
        y_axis_idx = spu.VectorField.y_axis_idx()
        start_collection = np.zeros((n_rods, grid_dim))
        for i in range(n_rods):
            start_collection[i, x_axis_idx] = (
                i % num_rods_along_x
            ) * self.spacing_between_rods
            start_collection[i, y_axis_idx] = (
                i // num_rods_along_x
            ) * self.spacing_between_rods

        # Shift the carpet to the provided centroid
        self.carpet_base_centroid = carpet_base_centroid
        current_carpet_start_centroid = np.mean(start_collection, axis=0)
        start_collection += (
            self.carpet_base_centroid.reshape(-1, grid_dim)
            - current_carpet_start_centroid
        )

        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([0.0, 1.0, 0.0])

        # Add magnetization to rods
        spatial_magnetisation_wavelength_x = self.carpet_length_x * wavelength_x_factor
        spatial_magnetisation_wavelength_y = self.carpet_length_y * wavelength_y_factor
        magnetization_density = (
            magnetic_bond_number
            * youngs_modulus
            * moment_of_inertia
            / (volume * magnetic_field_strength * self.rod_base_length)
        )
        magnetization_angle_x = spatial_magnetisation_phase_diff + (
            2
            * np.pi
            * start_collection[..., x_axis_idx]
            / spatial_magnetisation_wavelength_x
        )
        magnetization_angle_y = spatial_magnetisation_phase_diff + (
            2
            * np.pi
            * start_collection[..., y_axis_idx]
            / spatial_magnetisation_wavelength_y
        )
        self.magnetic_rod_list = []
        magnetization_direction_list = []

        for i in range(n_rods):
            if not magnetization_2d:
                magnetization_direction = np.array(
                    [
                        np.sin(magnetization_angle_x[i]),
                        0.0,
                        np.cos(magnetization_angle_x[i]),
                    ]
                ).reshape(3, 1) * np.ones(n_elem)
            else:
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
                youngs_modulus=youngs_modulus,
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
                [self.angular_frequency, 0, self.angular_frequency]
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

        self.final_time = num_cycles * self.period
        self.total_steps = int(self.final_time / self.dt)

        if plot_result:
            self.rendering_fps = 30
            self.rod_post_processing_list: list[dict] = []
            self.add_callback()

        self.timestepper = ea.PositionVerlet()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.timestepper, self.magnetic_beam_sim
        )

    def finalize(self) -> None:
        """Finalize the PyElastica simulator"""
        self.magnetic_beam_sim.finalize()

    def add_callback(self) -> None:
        """Add callbacks"""

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

    def time_step(self, time: float, time_step: float) -> float:
        """Time step the simulator"""
        time = self.do_step(
            self.timestepper,
            self.stages_and_updates,
            self.magnetic_beam_sim,
            time,
            time_step,
        )
        return time

    def run(
        self,
    ) -> None:
        """Run simulation"""
        ea.integrate(
            self.timestepper, self.magnetic_beam_sim, self.final_time, self.total_steps
        )

        if self.plot_result:
            x_axis_idx = spu.VectorField.x_axis_idx()
            y_axis_idx = spu.VectorField.y_axis_idx()
            z_axis_idx = spu.VectorField.z_axis_idx()
            # Plot the magnetic rod time history
            spu.plot_video_of_rod_surface(
                self.rod_post_processing_list,
                fps=self.rendering_fps,
                step=10,
                x_limits=(
                    self.carpet_base_centroid[x_axis_idx] - 0.6 * self.carpet_length_x,
                    self.carpet_base_centroid[x_axis_idx] + 0.6 * self.carpet_length_x,
                ),
                y_limits=(
                    self.carpet_base_centroid[y_axis_idx] - 0.6 * self.carpet_length_y,
                    self.carpet_base_centroid[y_axis_idx] + 0.6 * self.carpet_length_y,
                ),
                z_limits=(
                    self.carpet_base_centroid[z_axis_idx] - 0.1 * self.rod_base_length,
                    self.carpet_base_centroid[z_axis_idx] + 1.5 * self.rod_base_length,
                ),
                vis3D=True,
            )


if __name__ == "__main__":
    cilia_carpet_sim = MagneticCiliaCarpetSimulator()
    cilia_carpet_sim.finalize()
    cilia_carpet_sim.run()
