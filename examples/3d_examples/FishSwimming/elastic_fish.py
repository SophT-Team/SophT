import numpy as np
import elastica as ea
import sopht.utils as spu
from fish_geometry import update_rod_for_fish_geometry, create_fish_geometry

# from fish_muscle_forces import MuscleTorques
from fish_curvature_actuation import FishCurvature
import os
from scipy.interpolate import CubicSpline


class ElasticFishSimulator:
    def __init__(
        self,
        n_elements: int = 100,
        final_time: float = 20,
        rod_density: float = 1e3 / 15,
        youngs_modulus: float = 15e5,  # 6e5,#4e5,
        base_length: float = 1.0,
        start: np.ndarray = np.array([0.0, 0.0, 0.0]),
        muscle_torque_coefficients=np.array([]),
        tau_coeff: float = 1.44,
        plot_result: bool = True,
        period: float = 1.0,
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
        self.origin = start
        self.base_length = base_length
        poisson_ratio = 0.5
        shear_modulus = youngs_modulus / (1.0 + poisson_ratio)  # Pa
        direction = np.array([1.0, 0.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        rest_lengths = base_length / n_elements * np.ones(n_elements)
        width, _ = create_fish_geometry(rest_lengths)  # use width as radius
        self.shearable_rod = ea.CosseratRod.straight_rod(
            n_elements=n_elements,
            start=self.origin,
            direction=direction,
            normal=normal,
            base_length=self.base_length,
            base_radius=width,
            density=rod_density,
            nu=0.0,  # internal damping constant, deprecated in v0.3.0
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )
        update_rod_for_fish_geometry(rod_density, youngs_modulus, self.shearable_rod)
        self.simulator.append(self.shearable_rod)

        # Muscle torques
        ramp_up_time = period
        wave_length = base_length / tau_coeff
        wave_number = 2 * np.pi / wave_length
        self.simulator.add_forcing_to(self.shearable_rod).using(
            FishCurvature,
            coefficients=muscle_torque_coefficients,
            period=period,
            wave_number=wave_number,
            phase_shift=0.0,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=ramp_up_time,
        )

        self.dt = 0.001 / 2 * self.shearable_rod.rest_lengths[0]
        # TODO: Dampen only when in space, otherwise let flow forces dampen the rod
        # damping_constant = 2.0/10/100
        # self.simulator.dampen(self.shearable_rod).using(
        #     ea.AnalyticalLinearDamper,
        #     damping_constant=damping_constant,
        #     time_step=self.dt,
        # )

        self.final_time = final_time
        self.total_steps = int(self.final_time / self.dt)

        if plot_result:
            self.rendering_fps = 120
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
                    self.callback_params["rest_lengths"].append(
                        system.rest_lengths.copy()
                    )
                    self.callback_params["curvature"].append(system.kappa.copy())
                    self.callback_params["torque"].append(
                        system.external_torques.copy()
                    )

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
            self.simulator,
            time,
            time_step,
        )
        return time

    def run(
        self,
    ) -> None:
        ea.integrate(
            self.timestepper,
            self.simulator,
            self.final_time,
            self.total_steps,
            progress_bar=False,
        )


if __name__ == "__main__":

    # if os.path.exists("optimized_coefficients.txt"):
    #     muscle_torque_coefficients = np.genfromtxt(
    #         "optimized_coefficients.txt", delimiter=","
    #     )
    # elif os.path.exists("outcmaes/xrecentbest.dat"):
    #     muscle_torque_coefficients = np.loadtxt("outcmaes/xrecentbest.dat", skiprows=1)[
    #         -1, 5:
    #     ]
    # else:
    #     muscle_torque_coefficients = np.array([1.51, 0.48, 5.74, 2.73, 1.44])

    muscle_torque_coefficients = np.zeros((2, 6))
    muscle_torque_coefficients[0, :] = np.array([0, 0.05, 0.33, 0.67, 0.95, 1])
    muscle_torque_coefficients[1, :] = np.array([0, 1.51, 0.48, 5.74, 2.73, 0.0])
    tau_coeff = 1.44

    period = 1.0
    final_time = 12 * period
    elastic_fish_sim = ElasticFishSimulator(
        muscle_torque_coefficients=muscle_torque_coefficients,
        tau_coeff=tau_coeff,
        final_time=final_time,
        period=period,
    )
    elastic_fish_sim.finalize()
    elastic_fish_sim.run()

    x_axis_idx = spu.VectorField.x_axis_idx()
    y_axis_idx = spu.VectorField.y_axis_idx()
    z_axis_idx = spu.VectorField.z_axis_idx()
    # Plot the magnetic rod time history
    spu.plot_video_of_rod_surface(
        elastic_fish_sim.rod_post_processing_list,
        fps=15,
        step=10,
        x_limits=(
            elastic_fish_sim.origin[x_axis_idx] - 0.5 * elastic_fish_sim.base_length,
            elastic_fish_sim.origin[x_axis_idx] + 1.5 * elastic_fish_sim.base_length,
        ),
        y_limits=(
            elastic_fish_sim.origin[y_axis_idx] - 0.2 * elastic_fish_sim.base_length,
            elastic_fish_sim.origin[y_axis_idx] + 0.2 * elastic_fish_sim.base_length,
        ),
        z_limits=(
            elastic_fish_sim.origin[z_axis_idx] - 0.5 * elastic_fish_sim.base_length,
            elastic_fish_sim.origin[z_axis_idx] + 0.5 * elastic_fish_sim.base_length,
        ),
        vis3D=True,
    )

    # Retrieve simulation results
    time_sim = np.array(elastic_fish_sim.rod_post_processing_list[0]["time"])
    nondim_time = time_sim / period
    # node_position_sim = np.array(
    #     elastic_fish_sim.rod_post_processing_list[0]["position"][:]
    # )
    # node_position_sim = node_position_sim - node_position_sim[:, :, 0][:, :, np.newaxis]

    # Get non-dimensional position along rod from simulation
    rest_lengths = np.array(
        elastic_fish_sim.rod_post_processing_list[0]["rest_lengths"][:]
    )
    s_node = np.zeros((rest_lengths.shape[0], rest_lengths.shape[1]))
    s_node[:, :] = np.cumsum(rest_lengths, axis=1)
    s_node /= s_node[:, -1:]
    s_node_inner = s_node[:, :-1]

    # Get curvatures from simulation
    curvatures = np.array(elastic_fish_sim.rod_post_processing_list[0]["curvature"][:])

    positions = np.array(elastic_fish_sim.rod_post_processing_list[0]["position"][:])

    # Compute error
    # compare only after ramp up, towards end of sim
    start = np.where(nondim_time >= final_time - 2 * period)[0][0]

    # Compute curvature solution
    # curv_spline = interp1d(control_points, curv_coeffs, kind="cubic")
    curv_spline = CubicSpline(
        muscle_torque_coefficients[0, :],
        muscle_torque_coefficients[1, :],
        bc_type="natural",
    )
    curvatures_amplitude = curv_spline(s_node_inner)
    curvatures_solution = curvatures_amplitude * np.sin(
        2.0 * np.pi * (nondim_time[:, np.newaxis] - tau_coeff * s_node_inner)
    )
    # curvature error
    error = np.linalg.norm(
        curvatures[start:, 1, :] - curvatures_solution[start:, :], axis=1
    )

    # plot curvature along rod for a few frames to see
    import matplotlib

    matplotlib.use("Agg")  # Must be before importing matplotlib.pyplot or pylab!
    from matplotlib import pyplot as plt

    # In Material Frame
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    skip_frames = 30
    axs[0].plot(
        curvatures[start::skip_frames, 1, :].T, "-", color="red", label="simulation"
    )
    axs[0].plot(
        curvatures_solution[start::skip_frames, :].T,
        "--",
        color="skyblue",
        label="solution",
    )
    plt.tight_layout()
    fig.align_ylabels()
    # fig.legend(prop={"size": 20})
    fig.savefig("curvature_envelope_comparison.png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    axs[0].plot(nondim_time[start:], error, "-", color="red")
    plt.tight_layout()
    fig.align_ylabels()
    fig.savefig("curvature_error.png")
    plt.close(plt.gcf())

    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(10, 10), frameon=True, dpi=150)

    axs = []
    axs.append(plt.subplot2grid((1, 1), (0, 0)))
    skip_frames = 20
    axs[0].plot(
        positions[start::skip_frames, 0, :].T, positions[start::skip_frames, 2, :].T
    )
    plt.tight_layout()
    fig.align_ylabels()
    # fig.legend(prop={"size": 20})
    fig.savefig("position_envelope.png")
    plt.close(plt.gcf())
