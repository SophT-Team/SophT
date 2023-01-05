import numpy as np
import elastica as ea
import sopht.utils as spu
from fish_geometry import update_rod_for_fish_geometry
from collections import defaultdict
from fish_muscle_forces import MuscleTorques


class ElasticFishSimulator:
    def __init__(
        self,
        n_elements: float = 50,
        final_time: float = 20,
        rod_density: float = 1e3,
        youngs_modulus: float = 1e5,
        base_length: float = 1.0,
        start: np.ndarray = np.array([0.0, 0.0, 0.0]),
        slenderness_ratio: float = 10,
        muscle_torque_coefficients=np.array([1.29, 0.52, 5.43, 4.28]),
        plot_result: bool = True,
    ) -> None:
        class BaseSimulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Connections,
        ):
            ...

        self.plot_result = plot_result
        self.simulator = BaseSimulator()

        grid_dim = 3
        rod_dim = grid_dim
        x_axis_idx = spu.VectorField.x_axis_idx()
        y_axis_idx = spu.VectorField.y_axis_idx()
        z_axis_idx = spu.VectorField.z_axis_idx()
        self.origin = start

        self.base_length = base_length
        base_diameter = self.base_length / slenderness_ratio
        base_radius = base_diameter / 2

        normal = np.array([0.0, 0.0, 1.0])
        poisson_ratio = 0.5
        shear_modulus = youngs_modulus / (1.0 + poisson_ratio)  # Pa
        direction = np.array([0.0, 0.0, 1.0])
        normal = np.array([1.0, 0.0, 0.0])

        self.shearable_rod = ea.CosseratRod.straight_rod(
            n_elements=n_elements,
            start=self.origin,
            direction=direction,
            normal=normal,
            base_length=self.base_length,
            base_radius=base_radius,
            density=rod_density,
            nu=0.0,  # internal damping constant, deprecated in v0.3.0
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )

        update_rod_for_fish_geometry(rod_density, youngs_modulus, self.shearable_rod)
        self.simulator.append(self.shearable_rod)

        # Muscle torques
        period = 1
        ramp_up_time = 0.5
        tau = 1.71
        wave_number = 2 * np.pi / base_length
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorques,
            base_length=self.base_length,
            coefficients=muscle_torque_coefficients,
            period=period,
            wave_number=wave_number,
            phase_shift=0,
            direction=normal,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=ramp_up_time,
        )

        damping_constant = 0.1
        self.dt = 0.05 * self.shearable_rod.rest_lengths[0]
        self.simulator.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.dt,
        )

        self.final_time = final_time
        self.total_steps = int(self.final_time / self.dt)

        if plot_result:
            self.rendering_fps = 30
            self.rod_post_processing_list: list[dict] = []
            self.add_callback()

        self.timestepper = ea.PositionVerlet()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.timestepper, self.simulator
        )

    def finalize(self) -> None:
        self.simulator.finalize()

    def add_callback(self) -> None:
        # Add callbacks
        class FishCallBack(ea.CallBackBaseClass):
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
        self.rod_post_processing_list.append(ea.defaultdict(list))
        self.simulator.collect_diagnostics(self.shearable_rod).using(
            FishCallBack,
            step_skip=int(1.0 / (self.rendering_fps * self.dt)),
            callback_params=self.rod_post_processing_list[0],
        )

    def time_step(self, time: float, time_step: float) -> float:
        """Time step the simulator"""
        time = self.do_step(
            self.timestepper,
            self.stages_and_updates,
            self.net_simulator,
            time,
            time_step,
        )
        return time

    def run(
        self,
    ) -> None:
        ea.integrate(
            self.timestepper, self.simulator, self.final_time, self.total_steps
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
                    self.origin[x_axis_idx] - 2.0 * self.base_length,
                    self.origin[x_axis_idx] + 2.0 * self.base_length,
                ),
                y_limits=(
                    self.origin[y_axis_idx] - 2.0 * self.base_length,
                    self.origin[y_axis_idx] + 2.0 * self.base_length,
                ),
                z_limits=(
                    self.origin[z_axis_idx] - 2.0 * self.base_length,
                    self.origin[z_axis_idx] + 2.0 * self.base_length,
                ),
                vis3D=True,
            )


if __name__ == "__main__":
    elastic_fish_sim = ElasticFishSimulator()
    elastic_fish_sim.finalize()
    elastic_fish_sim.run()

    a = 5
