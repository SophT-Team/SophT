import numpy as np
import numpy.typing as npt
import elastica as ea
from elastica._rotations import _get_rotation_matrix
from oscillation_activation_functions import OscillationActivation
from set_environment_tapered_arm import OctopusEnvironment


def assemble_octopus(
    n_elems: int,
    start: npt.NDArray[np.float64],
    direction: npt.NDArray[np.float64],
    binormal: npt.NDArray[np.float64],
    normal: npt.NDArray[np.float64],
    rho_s: float,
    base_length: float,
    base_radius: float,
    taper_ratio: float,
    youngs_modulus: float,
    shear_modulus: float,
):

    rod_list = []
    arm_rod_list = []
    number_of_straight_rods = 8
    # First straight rod is at the center, remaining ring rods are around the first ring rod.
    angle_btw_straight_rods = (
        0 if number_of_straight_rods == 1 else 2 * np.pi / (number_of_straight_rods)
    )
    bank_angle = np.deg2rad(30)
    angle_wrt_center_rod = []
    for i in range(number_of_straight_rods):
        rotation_matrix = _get_rotation_matrix(
            angle_btw_straight_rods * i, direction.reshape(3, 1)
        ).reshape(3, 3)
        direction_from_center_to_rod = rotation_matrix @ binormal

        angle_wrt_center_rod.append(angle_btw_straight_rods * i)

        # Compute the rotation matrix, for getting the correct banked angle.
        normal_banked_rod = rotation_matrix @ normal
        # Rotate direction vector around new normal to get the new direction vector.
        # Note that we are doing ccw rotation and direction should be towards the center.
        rotation_matrix_banked_rod = _get_rotation_matrix(
            (np.pi / 2 - bank_angle), normal_banked_rod.reshape(3, 1)
        ).reshape(3, 3)
        direction_banked_rod = rotation_matrix_banked_rod @ direction

        start_rod = (
            start
            + (direction_from_center_to_rod)
            * (
                # center rod            # this rod
                +2 * base_radius
                + base_radius
            )
            * 0
        )

        radius = np.linspace(base_radius, base_radius / taper_ratio, n_elems + 1)
        radius_mean = (radius[:-1] + radius[1:]) / 2

        rod = ea.CosseratRod.straight_rod(
            n_elements=n_elems,
            start=start_rod,
            direction=direction_banked_rod,
            normal=normal_banked_rod,
            base_length=base_length,
            base_radius=radius_mean.copy(),
            density=rho_s,
            nu=0.0,  # internal damping constant, deprecated in v0.3.0
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )

        rod_list.append(rod)
        arm_rod_list.append(rod)

    # Octopus head initialization
    slenderness_ratio_head = 5.787

    octopus_head_length = 2 * base_radius * slenderness_ratio_head / 2
    octopus_head_n_elems = int(n_elems * octopus_head_length / base_length)

    octopus_head_radius = (
        2 * base_radius * np.linspace(0.9, 1.0, octopus_head_n_elems) ** 3
    )

    body_rod = ea.CosseratRod.straight_rod(
        n_elements=octopus_head_n_elems,
        start=start,
        direction=-direction,
        normal=normal,
        base_length=octopus_head_length,
        base_radius=octopus_head_radius,  # .copy(),
        density=rho_s,
        nu=0.0,  # internal damping constant, deprecated in v0.3.0
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    rod_list.append(body_rod)

    rigid_body_list = []
    sphere_com = body_rod.position_collection[..., -1]
    sphere_diameter = np.max(octopus_head_radius)
    sphere = ea.Sphere(center=sphere_com, base_radius=sphere_diameter, density=rho_s)
    rigid_body_list.append(sphere)

    return arm_rod_list, rod_list, rigid_body_list


def initialize_activation_functions(
    env: OctopusEnvironment,
    activation_period: float,
    activation_level_max: float,
    wave_number: float = 0.05,
    start_non_dim_length: float = 0.0,
    end_non_dim_length: float = 1.0,
):
    activations: list[list[npt.NDArray[np.float64]]] = []
    activation_functions: list[list[object]] = []
    for rod_id, rod in enumerate(env.arm_rod_list):
        activations.append([])
        activation_functions.append([])
        for m in range(len(env.rod_muscle_groups[rod_id])):
            activations[rod_id].append(
                np.zeros(env.rod_muscle_groups[rod_id][m].activation.shape)
            )
            if m == 4:
                activation_functions[rod_id].append(
                    OscillationActivation(
                        wave_number=wave_number,
                        frequency=1 / activation_period,  # f_p,
                        phase_shift=0,  # X_p,
                        start_time=0.0,
                        end_time=10000,
                        start_non_dim_length=start_non_dim_length,
                        end_non_dim_length=end_non_dim_length,
                        n_elems=rod.n_elems,
                        activation_level_max=activation_level_max,
                        a=10,
                        b=0.5,
                    )
                )
            else:
                activation_functions[rod_id].append(None)

    return activations, activation_functions
