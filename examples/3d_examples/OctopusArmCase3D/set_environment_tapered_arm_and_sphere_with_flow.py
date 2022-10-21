__all__ = ["Environment"]

from collections import defaultdict
import numpy as np
import elastica as ea
from elastica._calculus import _isnan_check
from arm_functions import StraightRodCallBack, CylinderCallBack

# TODO: After HS releases the COMM remove below import statements and just import the package.
from coomm.actuations.muscles import (
    force_length_weight_guassian,
    force_length_weight_poly,
)
from coomm.actuations.muscles import (
    MuscleGroup,
    LongitudinalMuscle,
    ObliqueMuscle,
    TransverseMuscle,
    ApplyMuscleGroups,
)


class BaseSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.CallBacks,
    ea.Damping,
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
        self.StatefulStepper = ea.PositionVerlet()

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

    def set_arm(self, E, rod):
        self.set_rod(E, rod)
        self.set_muscles(self.shearable_rod)
        # self.set_drag_force(
        #     base_length, radius[0], radius[-1],
        #     self.shearable_rod, self.rod_parameters_dict
        # )

    def setup(self, E, rod):
        self.set_arm(E, rod)

    def set_rod(self, E, rod):
        """Set up a rod"""

        self.E = E
        #
        # n_elements = 50  # number of discretized elements of the arm
        # base_length = 0.2  # total length of the arm
        # radius_base = 0.012  # radius of the arm at the base
        # radius_tip = 0.0012   # radius of the arm at the tip
        # radius = np.linspace(radius_base, radius_tip, n_elements + 1)
        # # radius = np.linspace(radius_base, radius_base, n_elements + 1)
        # radius_mean = (radius[:-1] + radius[1:]) / 2
        # damp_coefficient = 0.05
        # density = 1050
        # poisson_ratio = 0.5
        # youngs_modulus = self.E
        # shear_modulus = self.E / (2 * (1 + poisson_ratio))
        #
        # start = np.zeros((3,)) + np.array([0.5 * base_length, 0.5 * base_length, 0])
        # direction = np.array([1.0, 0.0, 0.0])
        # normal = np.array([0.0, 0.0, -1.0])
        #
        # self.shearable_rod = ea.CosseratRod.straight_rod(
        #     n_elements=n_elements,
        #     start=start,
        #     direction=direction,
        #     normal=normal,
        #     base_length=base_length,
        #     base_radius=radius_mean.copy(),
        #     density=density,
        #     nu=0.0,  # internal damping constant, deprecated in v0.3.0
        #     youngs_modulus=youngs_modulus,
        #     shear_modulus=shear_modulus,
        # )
        self.shearable_rod = rod
        self.simulator.append(self.shearable_rod)

        # Cylinder
        # cylinder_start = start + direction * base_length * 0.7 + np.array([0, 0.1, 0])
        # cylinder_direction = normal
        # cylinder_normal = direction
        # cylinder_radius = radius_base
        # cylinder_density = density
        # self.cylinder = ea.Cylinder(
        #     cylinder_start,
        #     cylinder_direction,
        #     cylinder_normal,
        #     base_length,
        #     cylinder_radius,
        #     cylinder_density,
        # )
        # self.simulator.append(self.cylinder)

        """ Set up boundary conditions """
        self.simulator.constrain(self.shearable_rod).using(
            ea.OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        # Set exponential damper
        # Below damping tuned for time-step 2.5E-4
        damp_coefficient = 0.5e-2  # 0.05
        density = 1
        radius_base = self.shearable_rod.radius[0]
        damping_constant = (
            damp_coefficient / density / (np.pi * radius_base**2) / 15
        )  # For tapered rod /15 stable
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        # # Filter damper
        # self.simulator.dampen(self.shearable_rod).using(ea.LaplaceDissipationFilter, filter_order=2)

        # return base_length, radius

    def set_muscles(self, arm):
        """Add muscle actuation"""

        def add_muscle_actuation(arm):
            # radius_base = arm.radius[0]

            radius_base = 0.012

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
            TM_max_muscle_stress = 1.5 * self.E  # 15_000.0
            LM_max_muscle_stress = 10 * self.E  # 50_000.0 * 2
            OM_max_muscle_stress = 5 * self.E  # 50_000.0

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

        self.muscle_groups = add_muscle_actuation(arm)
        self.muscle_callback_params_list = [
            defaultdict(list) for _ in self.muscle_groups
        ]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscleGroups,
            muscle_groups=self.muscle_groups,
            step_skip=self.step_skip,
            callback_params_list=self.muscle_callback_params_list,
        )

    def reset(self, E, rod):
        self.simulator = BaseSimulator()

        self.setup(E, rod)

        """ Finalize the simulator and create time stepper """

    def finalize(self):
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
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


class Environment(ArmEnvironment):
    def get_systems(self):
        return [self.shearable_rod]

    def setup(self, E, rod):
        self.set_arm(E, rod)
