from collections import defaultdict
import numpy as np
import elastica as ea
from elastica._calculus import _isnan_check

from coomm.actuations.muscles import (
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
        return self.rod_list  # [self.shearable_rod]

    def set_arm(self, E, density, rod, rod_id):
        self.set_rod(E, density, rod)
        self.set_muscles(rod, rod_id)

    def setup(self, E, density, rod, rod_id):
        self.set_arm(E, density, rod, rod_id)

    def set_rod(self, E, density, rod):
        """Set up a rod"""

        self.E = E
        # self.shearable_rod = rod
        self.simulator.append(rod)

        """ Set up boundary conditions """
        self.simulator.constrain(rod).using(
            ea.GeneralConstraint,
            # constrained_position_idx=(0,),
            constrained_director_idx=(0,),
            # translational_constraint_selector = np.array([True, True, True]),
            rotational_constraint_selector=np.array([True, True, True]),
        )

        # Set exponential damper
        # Below damping tuned for time-step 2.5E-4
        damp_coefficient = 0.5e-2  # 0.05
        radius_base = rod.radius[0]
        damping_constant = (
            damp_coefficient / density / (np.pi * radius_base**2) / 15 / 1e3 * 100
        )  # For tapered rod /15 stable
        self.simulator.dampen(rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

    def set_muscles(self, arm, arm_id):
        """Add muscle actuation"""

        def add_muscle_actuation(arm):
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

        self.rod_muscle_groups[arm_id] = add_muscle_actuation(arm)
        # self.muscle_groups = add_muscle_actuation(arm)
        self.muscle_callback_params_list[arm_id] = [
            defaultdict(list) for _ in self.rod_muscle_groups[arm_id]
        ]
        self.simulator.add_forcing_to(arm).using(
            ApplyMuscleGroups,
            muscle_groups=self.rod_muscle_groups[arm_id],  # self.muscle_groups,
            step_skip=self.step_skip,
            callback_params_list=self.muscle_callback_params_list[arm_id],
        )

    def reset(self, E, density, rod_list, arm_rod_list):
        self.simulator = BaseSimulator()

        self.rod_list = rod_list
        self.arm_rod_list = arm_rod_list
        self.rod_muscle_groups = []
        self.muscle_callback_params_list = []

        for i in range(len(rod_list)):
            self.rod_muscle_groups.append([])
            self.muscle_callback_params_list.append([])

        for rod_id, rod in enumerate(rod_list):
            self.setup(E, density, rod, rod_id)

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
        for rod_id in range(len(self.arm_rod_list)):
            for muscle_group, activation in zip(
                self.rod_muscle_groups[rod_id], muscle_activations[rod_id]
            ):
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
        for rod in self.rod_list:
            invalid_values_condition = _isnan_check(rod.position_collection)

            if invalid_values_condition == True:
                print("NaN detected in the simulation !!!!!!!!")
                done = True
                break

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done


class Environment(ArmEnvironment):
    def get_systems(self):
        return self.rod_list  # [self.shearable_rod]

    def setup(self, E, density, rod, rod_id):
        self.set_arm(E, density, rod, rod_id)
