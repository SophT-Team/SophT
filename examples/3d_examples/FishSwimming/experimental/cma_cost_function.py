from elastic_fish import ElasticFishSimulator
import numpy as np
from scipy.interpolate import CubicSpline


def curvature_obj_fct(muscle_torque_coefficients):
    period = 1
    final_time = 12 * period

    elastic_fish_sim = ElasticFishSimulator(
        muscle_torque_coefficients=muscle_torque_coefficients,
        final_time=final_time,
        period=period,
    )
    elastic_fish_sim.finalize()
    elastic_fish_sim.run()

    # Retrieve simulation results
    time_sim = np.array(elastic_fish_sim.rod_post_processing_list[0]["time"])
    nondim_time = time_sim / period
    node_position_sim = np.array(
        elastic_fish_sim.rod_post_processing_list[0]["position"][:]
    )
    node_position_sim = node_position_sim - node_position_sim[:, :, 0][:, :, np.newaxis]

    # Get non-dimensional position along rod from simulation
    rest_lengths = np.array(
        elastic_fish_sim.rod_post_processing_list[0]["rest_lengths"][:]
    )
    s_node = np.zeros((rest_lengths.shape[0], rest_lengths.shape[1] + 1))
    s_node[:, 1:] = np.cumsum(rest_lengths, axis=1)
    s_node /= s_node[:, -1:]
    s_node_inner = s_node[:, 1:-1]
    # Get curvatures from simulation
    curvatures = np.array(elastic_fish_sim.rod_post_processing_list[0]["curvature"][:])

    # Compute error
    # compare only after ramp up, towards end of sim (say last two period)
    start = np.where(nondim_time >= final_time - 2 * period)[0][0]

    # Compute curvature solution (based on coefficients from Kern)
    control_points = np.array([0, 1.0 / 3, 2.0 / 3, 1])
    curv_coeffs = np.array([1.51, 0.48, 5.74, 2.73])
    tau_coeff = 1.44
    curv_spline = CubicSpline(control_points, curv_coeffs, bc_type="natural")
    curvatures_amplitude = curv_spline(s_node_inner)
    curvatures_solution = curvatures_amplitude * np.sin(
        2.0 * np.pi * (nondim_time[:, np.newaxis] - tau_coeff * s_node_inner)
    )
    # curvature error
    error = np.linalg.norm(curvatures[start:, 0, :] - curvatures_solution[start:, :])

    return error
