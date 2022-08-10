__all__ = ["Environment"]

import numpy as np
from elastica import *
from elastica.timestepper import extend_stepper_interface
from elastica._calculus import _isnan_check
from post_processing import plot_video_with_surface
from arm_functions import StraightRodCallBack, CylinderCallBack

# TODO: After HS releases the COMM remove below import statements and just import the package.
from actuations.muscles import (
    force_length_weight_guassian,
    force_length_weight_poly,
)
from actuations.muscles import (
    MuscleGroup,
    LongitudinalMuscle,
    ObliqueMuscle,
    TransverseMuscle,
    ApplyMuscleGroups,
)


class BaseSimulator(
    BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping
):
    pass


class ArmEnvironment:
    def __init__(
        self,
        final_time,
        time_step=1.0e-5,
        rendering_fps=30,
        COLLECT_DATA_FOR_POSTPROCESSING=True,
    ):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.rendering_fps = rendering_fps
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

    def get_systems(
        self,
    ):
        return [self.shearable_rod]

    def get_data(
        self,
    ):
        return [self.rod_parameters_dict]

    def set_arm(self):
        base_length, radius = self.set_rod()
        self.set_muscles(radius[0], self.shearable_rod)
        # self.set_drag_force(
        #     base_length, radius[0], radius[-1],
        #     self.shearable_rod, self.rod_parameters_dict
        # )

    def setup(self):
        self.set_arm()

    def set_rod(self):
        """Set up a rod"""
        n_elements = 50  # number of discretized elements of the arm
        base_length = 0.2  # total length of the arm
        radius_base = 0.012  # radius of the arm at the base
        radius_tip = 0.0012  # radius of the arm at the tip
        radius = np.linspace(radius_base, radius_tip, n_elements + 1)
        # radius = np.linspace(radius_base, radius_base, n_elements + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2
        damp_coefficient = 0.05
        density = 1050

        start = np.zeros((3,)) + np.array([0.5 * base_length, 0.5 * base_length, 0])
        direction = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, -1.0])

        self.shearable_rod = CosseratRod.straight_rod(
            n_elements=n_elements,
            start=start,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=radius_mean.copy(),
            density=density,
            nu=0.0,  # internal damping constant, deprecated in v0.3.0
            youngs_modulus=10_000,
            shear_modulus=10_000 / (2 * (1 + 0.5)),
            # poisson_ratio=0.5, Default is 0.5
            nu_for_torques=damp_coefficient * ((radius_mean / radius_base) ** 4) * 0.0,
        )
        self.simulator.append(self.shearable_rod)

        # Cylinder
        cylinder_start = start + direction * base_length * 0.7 + np.array([0, 0.1, 0])
        cylinder_direction = normal
        cylinder_normal = direction
        cylinder_radius = radius_base
        cylinder_density = density
        self.cylinder = Cylinder(
            cylinder_start,
            cylinder_direction,
            cylinder_normal,
            base_length,
            cylinder_radius,
            cylinder_density,
        )
        self.simulator.append(self.cylinder)

        self.rod_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            StraightRodCallBack,
            step_skip=self.step_skip,
            callback_params=self.rod_parameters_dict,
        )

        self.cylinder_parameters_dict = defaultdict(list)
        self.simulator.collect_diagnostics(self.cylinder).using(
            CylinderCallBack,
            step_skip=self.step_skip,
            callback_params=self.cylinder_parameters_dict,
        )

        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedRod, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        # Set exponential damper
        # Below damping tuned for time-step 2.5E-4
        damping_constant = (
            damp_coefficient / density / (np.pi * radius_base**2) / 15
        )  # For tapered rod /15 stable
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        # # # Filter damper
        # self.simulator.dampen(self.shearable_rod).using(LaplaceDissipationFilter, filter_order=4)

        return base_length, radius

    def set_muscles(self, base_radius, arm):
        """Add muscle actuation"""

        def add_muscle_actuation(radius_base, arm):

            muscle_groups = []

            LM_ratio_muscle_position = 0.0075 / radius_base
            OM_ratio_muscle_position = 0.01125 / radius_base

            AN_ratio_radius = 0.002 / radius_base
            TM_ratio_radius = 0.0045 / radius_base
            LM_ratio_radius = 0.003 / radius_base
            OM_ratio_radius = 0.00075 / radius_base

            OM_rotation_number = 6

            shearable_rod_area = np.pi * arm.radius**2
            TM_rest_muscle_area = shearable_rod_area * (
                TM_ratio_radius**2 - AN_ratio_radius**2
            )
            LM_rest_muscle_area = shearable_rod_area * (LM_ratio_radius**2)
            OM_rest_muscle_area = shearable_rod_area * (OM_ratio_radius**2)

            # stress is in unit [Pa]
            TM_max_muscle_stress = 15_000.0
            LM_max_muscle_stress = 50_000.0 * 2
            OM_max_muscle_stress = 50_000.0

            muscle_dict = dict(
                force_length_weight=force_length_weight_poly,
            )

            # Add a transverse muscle
            muscle_groups.append(
                MuscleGroup(
                    muscles=[
                        TransverseMuscle(
                            rest_muscle_area=TM_rest_muscle_area,
                            max_muscle_stress=TM_max_muscle_stress,
                            **muscle_dict,
                        )
                    ],
                    type_name="TM",
                )
            )

            # Add 4 longitudinal muscles
            for k in range(4):
                muscle_groups.append(
                    MuscleGroup(
                        muscles=[
                            LongitudinalMuscle(
                                muscle_init_angle=np.pi * 0.5 * k,
                                ratio_muscle_position=LM_ratio_muscle_position,
                                rest_muscle_area=LM_rest_muscle_area,
                                max_muscle_stress=LM_max_muscle_stress,
                                **muscle_dict,
                            )
                        ],
                        type_name="LM",
                    )
                )

            # Add a clockwise oblique muscle group (4 muscles)
            muscle_groups.append(
                MuscleGroup(
                    muscles=[
                        ObliqueMuscle(
                            muscle_init_angle=np.pi * 0.5 * m,
                            ratio_muscle_position=OM_ratio_muscle_position,
                            rotation_number=OM_rotation_number,
                            rest_muscle_area=OM_rest_muscle_area,
                            max_muscle_stress=OM_max_muscle_stress,
                            **muscle_dict,
                        )
                        for m in range(4)
                    ],
                    type_name="OM",
                )
            )

            # Add a counter-clockwise oblique muscle group (4 muscles)
            muscle_groups.append(
                MuscleGroup(
                    muscles=[
                        ObliqueMuscle(
                            muscle_init_angle=np.pi * 0.5 * m,
                            ratio_muscle_position=OM_ratio_muscle_position,
                            rotation_number=-OM_rotation_number,
                            rest_muscle_area=OM_rest_muscle_area,
                            max_muscle_stress=OM_max_muscle_stress,
                            **muscle_dict,
                        )
                        for m in range(4)
                    ],
                    type_name="OM",
                )
            )

            for muscle_group in muscle_groups:
                muscle_group.set_current_length_as_rest_length(arm)

            return muscle_groups

        self.muscle_groups = add_muscle_actuation(base_radius, arm)
        self.muscle_callback_params_list = [
            defaultdict(list) for _ in self.muscle_groups
        ]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscleGroups,
            muscle_groups=self.muscle_groups,
            step_skip=self.step_skip,
            callback_params_list=self.muscle_callback_params_list,
        )

    def reset(self):
        self.simulator = BaseSimulator()

        self.setup()

        """ Finalize the simulator and create time stepper """

    def finalize(self):
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        """ Return 
            (1) total time steps for the simulation step iterations
            (2) systems for controller design
        """
        return self.total_steps, self.get_systems()

    def step(self, time, muscle_activations):

        """Set muscle activations"""
        for muscle_group, activation in zip(self.muscle_groups, muscle_activations):
            muscle_group.apply_activation(activation)

        """ Run the simulation for one step """
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        """ Done is a boolean to reset the environment before episode is completed """
        done = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)

        if invalid_values_condition == True:
            print("NaN detected in the simulation !!!!!!!!")
            done = True

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done

    def save_data(self, filename="simulation", **kwargs):
        # Save data in numpy format
        import os

        current_path = os.getcwd()
        current_path = kwargs.get("folder_name", current_path)
        save_folder = os.path.join(current_path, "data")
        os.makedirs(save_folder, exist_ok=True)

        self.post_processing_dict_list = [self.rod_parameters_dict]

        time = np.array(self.post_processing_dict_list[0]["time"])
        number_of_straight_rods = 1
        if number_of_straight_rods > 0:
            n_elems_straight_rods = self.shearable_rod.n_elems
            straight_rods_position_history = np.zeros(
                (
                    number_of_straight_rods,
                    time.shape[0],
                    3,
                    n_elems_straight_rods + 1,
                )
            )
            straight_rods_radius_history = np.zeros(
                (number_of_straight_rods, time.shape[0], n_elems_straight_rods)
            )

            for i in range(number_of_straight_rods):
                straight_rods_position_history[i, :, :, :] = np.array(
                    self.post_processing_dict_list[i]["position"]
                )
                straight_rods_radius_history[i, :, :] = np.array(
                    self.post_processing_dict_list[i]["radius"]
                )

        else:
            straight_rods_position_history = None
            straight_rods_radius_history = None

        np.savez(
            os.path.join(save_folder, "octopus_arm_test.npz"),
            time=time,
            straight_rods_position_history=straight_rods_position_history,
            straight_rods_radius_history=straight_rods_radius_history,
        )

    def post_processing(self, filename_video, **kwargs):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            import os

            current_path = os.getcwd()
            current_path = kwargs.get("folder_name", current_path)

            plot_video_with_surface(
                [self.rod_parameters_dict],
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                **kwargs,
            )


class Environment(ArmEnvironment):
    def get_systems(self):
        return [self.shearable_rod, self.cylinder]

    def get_data(self):
        return [self.rod_parameters_dict, self.cylinder_parameters_dict]

    def setup(self):
        self.set_arm()
