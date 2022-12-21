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
        final_time: float,
        time_step: float = 1.0e-5,
        rendering_fps: int = 30,
        COLLECT_DATA_FOR_POSTPROCESSING: bool = True,
    ) -> None:
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
    ) -> list[ea.CosseratRod]:
        return [self.shearable_rod]

    def set_arm(self, E: float, rod: ea.CosseratRod) -> None:
        self.set_rod(E, rod)
        self.set_muscles(self.shearable_rod)

    def setup(self, E: float, rod: ea.CosseratRod) -> None:
        self.set_arm(E, rod)

    def set_rod(self, E: float, rod: ea.CosseratRod) -> None:
        """Set up a rod"""

        self.E = E
        self.shearable_rod = rod
        self.simulator.append(self.shearable_rod)

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

    def set_muscles(self, arm: ea.CosseratRod) -> None:
        """Add muscle actuation"""

        def add_muscle_actuation(arm: ea.CosseratRod) -> list[MuscleGroup]:
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
        self.muscle_callback_params_list: list = [
            defaultdict(list) for _ in self.muscle_groups
        ]
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ApplyMuscleGroups,
            muscle_groups=self.muscle_groups,
            step_skip=self.step_skip,
            callback_params_list=self.muscle_callback_params_list,
        )

    def reset(self, E: float, rod: ea.CosseratRod) -> None:
        self.simulator = BaseSimulator()

        self.setup(E, rod)

        """ Finalize the simulator and create time stepper """

    def finalize(self) -> tuple[int, list[ea.CosseratRod]]:
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        """ Return
            (1) total time steps for the simulation step iterations
            (2) systems for controller design
        """
        return self.total_steps, self.get_systems()

    def step(
        self, time: float, muscle_activations: list[np.ndarray]
    ) -> tuple[float, list[ea.CosseratRod], bool]:

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

        if invalid_values_condition:
            print("NaN detected in the simulation !!!!!!!!")
            done = True

        """ Return
            (1) current simulation time
            (2) current systems
            (3) a flag denotes whether the simulation runs correlectly
        """
        return time, self.get_systems(), done


class Environment(ArmEnvironment):
    def get_systems(self) -> list[ea.CosseratRod]:
        return [self.shearable_rod]

    def setup(self, E: float, rod: ea.CosseratRod) -> None:
        self.set_arm(E, rod)
