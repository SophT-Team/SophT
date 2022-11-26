import numpy as np

import pytest

from sopht.numeric.immersed_boundary_ops import EulerianLagrangianGridCommunicator3D
from sopht.utils.precision import get_real_t, get_test_tol


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_local_eulerian_grid_support_of_lagrangian_grid_3d(n_values, precision):
    real_t = get_real_t(precision)
    grid_dim = 3
    eul_grid_size = n_values
    interp_kernel_width = 2
    num_lag_nodes = 3
    eul_domain_size = real_t(1.0)
    dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * dx)
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    local_eulerian_grid_support_of_lagrangian_grid_kernel = (
        eul_lag_communicator.local_eulerian_grid_support_of_lagrangian_grid_kernel
    )

    # generate reference manufactured solution for grid [0, 1]^3
    x = np.linspace(
        eul_grid_coord_shift, eul_domain_size - eul_grid_coord_shift, eul_grid_size
    ).astype(real_t)
    y = x.copy()
    z = x.copy()
    z_grid, y_grid, x_grid = np.meshgrid(z, y, x, indexing="ij")
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    # init lag. grid near domain center
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[2] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    lag_positions = ref_nearest_eul_grid_index_to_lag_grid * dx + eul_grid_coord_shift

    # find interpolation zone support for the lag. grid
    ref_local_eul_grid_support_of_lag_grid = np.zeros(
        (
            grid_dim,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            num_lag_nodes,
        )
    ).astype(real_t)
    for i in range(num_lag_nodes):
        ref_local_eul_grid_support_of_lag_grid[0, ..., i] = x_grid[
            ref_nearest_eul_grid_index_to_lag_grid[2, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[2, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[1, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[1, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[0, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[0, i]
            + interp_kernel_width
            + 1,
        ]
        ref_local_eul_grid_support_of_lag_grid[1, ..., i] = y_grid[
            ref_nearest_eul_grid_index_to_lag_grid[2, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[2, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[1, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[1, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[0, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[0, i]
            + interp_kernel_width
            + 1,
        ]
        ref_local_eul_grid_support_of_lag_grid[2, ..., i] = z_grid[
            ref_nearest_eul_grid_index_to_lag_grid[2, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[2, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[1, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[1, i]
            + interp_kernel_width
            + 1,
            ref_nearest_eul_grid_index_to_lag_grid[0, i]
            - interp_kernel_width
            + 1 : ref_nearest_eul_grid_index_to_lag_grid[0, i]
            + interp_kernel_width
            + 1,
        ]

    # get relative distance (support) of grid
    ref_local_eul_grid_support_of_lag_grid[
        ...
    ] = ref_local_eul_grid_support_of_lag_grid - lag_positions.reshape(
        grid_dim, 1, 1, 1, num_lag_nodes
    )

    # test against solution
    nearest_eul_grid_index_to_lag_grid = np.zeros_like(
        ref_nearest_eul_grid_index_to_lag_grid
    )
    local_eul_grid_support_of_lag_grid = np.zeros_like(
        ref_local_eul_grid_support_of_lag_grid
    )
    local_eulerian_grid_support_of_lagrangian_grid_kernel(
        local_eul_grid_support_of_lag_grid,
        nearest_eul_grid_index_to_lag_grid,
        lag_positions,
    )
    np.testing.assert_allclose(
        ref_nearest_eul_grid_index_to_lag_grid,
        nearest_eul_grid_index_to_lag_grid,
        atol=get_test_tol(precision),
    )
    np.testing.assert_allclose(
        ref_local_eul_grid_support_of_lag_grid,
        local_eul_grid_support_of_lag_grid,
        atol=get_test_tol(precision),
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_eulerian_to_lagrangian_grid_interpolation_kernel_3d(n_values, precision):
    real_t = get_real_t(precision)
    grid_dim = 3
    eul_grid_size = n_values
    interp_kernel_width = 2
    num_lag_nodes = 3
    eul_domain_size = real_t(1.0)
    dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * dx)
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    # init lag. grid near domain center
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[2] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    eulerian_to_lagrangian_grid_interpolation_kernel = (
        eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel
    )
    eul_grid_field = np.ones(
        (eul_grid_size, eul_grid_size, eul_grid_size), dtype=real_t
    )
    # set interp weight as a series of 0 to 8 * interp_kernel_width ** 3 - 1
    ref_interp_weights = np.arange(0, 8 * interp_kernel_width**3).reshape(
        2 * interp_kernel_width, 2 * interp_kernel_width, 2 * interp_kernel_width, 1
    )
    ref_interp_weights = np.tile(ref_interp_weights, reps=(1, 1, 1, num_lag_nodes))
    # summation formula for 1 to n
    ref_interp_weight_sum = (
        (8 * interp_kernel_width**3) * (8 * interp_kernel_width**3 - 1) / 2
    )

    lag_grid_field = np.zeros((num_lag_nodes), dtype=real_t)
    eulerian_to_lagrangian_grid_interpolation_kernel(
        lag_grid_field,
        eul_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )
    np.testing.assert_allclose(
        lag_grid_field,
        ref_interp_weight_sum * dx**grid_dim,
        atol=get_test_tol(precision),
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_eul_to_lag_grid_interpolation_kernel_3d(n_values, precision):
    real_t = get_real_t(precision)
    grid_dim = 3
    n_components = grid_dim
    eul_grid_size = n_values
    interp_kernel_width = 2
    num_lag_nodes = 3
    eul_domain_size = real_t(1.0)
    dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * dx)
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    # init lag. grid near domain center
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[2] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        n_components=n_components,
    )
    eulerian_to_lagrangian_grid_interpolation_kernel = (
        eul_lag_communicator.eulerian_to_lagrangian_grid_interpolation_kernel
    )
    eul_grid_field = np.ones(
        (n_components, eul_grid_size, eul_grid_size, eul_grid_size), dtype=real_t
    )
    # set interp weight as a series of 0 to 8 * interp_kernel_width ** 3 - 1
    ref_interp_weights = np.arange(0, 8 * interp_kernel_width**3).reshape(
        2 * interp_kernel_width, 2 * interp_kernel_width, 2 * interp_kernel_width, 1
    )
    ref_interp_weights = np.tile(ref_interp_weights, reps=(1, 1, 1, num_lag_nodes))
    # summation formula for 1 to n
    ref_interp_weight_sum = (
        (8 * interp_kernel_width**3) * (8 * interp_kernel_width**3 - 1) / 2
    )

    lag_grid_field = np.zeros((n_components, num_lag_nodes), dtype=real_t)
    eulerian_to_lagrangian_grid_interpolation_kernel(
        lag_grid_field,
        eul_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )
    np.testing.assert_allclose(
        lag_grid_field,
        ref_interp_weight_sum * dx**grid_dim,
        atol=get_test_tol(precision),
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_lagrangian_to_eulerian_grid_interpolation_kernel_3d(n_values, precision):
    real_t = get_real_t(precision)
    grid_dim = 3
    eul_grid_size = n_values
    interp_kernel_width = 2
    num_lag_nodes = 3
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    eul_domain_size = real_t(1.0)
    dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * dx)
    # init lag. grid near domain center
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[2] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
    )
    lagrangian_to_eulerian_grid_interpolation_kernel = (
        eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel
    )
    # init interp weights as all ones, essentially this should lead to
    # interpolation spreading ones onto the Eulerian grid
    ref_interp_weights = np.ones(
        (
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            num_lag_nodes,
        ),
        dtype=real_t,
    )
    prefactor_lag_field = 2
    ref_lag_grid_field = prefactor_lag_field * np.ones((num_lag_nodes), dtype=real_t)
    # reference integral of interpolated field onto Eulerian grid
    num_ones_in_ref_interp_weights = (
        (2 * interp_kernel_width)
        * (2 * interp_kernel_width)
        * (2 * interp_kernel_width)
    )
    ref_interpolated_field_sum = (
        num_lag_nodes * prefactor_lag_field * num_ones_in_ref_interp_weights
    )

    eul_grid_field = np.zeros(
        (eul_grid_size, eul_grid_size, eul_grid_size), dtype=real_t
    )
    lagrangian_to_eulerian_grid_interpolation_kernel(
        eul_grid_field,
        ref_lag_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )
    np.testing.assert_allclose(
        np.sum(eul_grid_field),
        ref_interpolated_field_sum,
        atol=get_test_tol(precision),
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_lag_to_eul_grid_interpolation_kernel_3d(n_values, precision):
    real_t = get_real_t(precision)
    grid_dim = 3
    n_components = grid_dim
    eul_grid_size = n_values
    interp_kernel_width = 2
    num_lag_nodes = 3
    ref_nearest_eul_grid_index_to_lag_grid = np.empty((grid_dim, num_lag_nodes)).astype(
        int
    )
    eul_domain_size = real_t(1.0)
    dx = real_t(eul_domain_size / eul_grid_size)
    eul_grid_coord_shift = real_t(0.5 * dx)
    # init lag. grid near domain center
    ref_nearest_eul_grid_index_to_lag_grid[0] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[1] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    ref_nearest_eul_grid_index_to_lag_grid[2] = np.arange(
        eul_grid_size // 3, eul_grid_size // 3 + num_lag_nodes
    )
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        n_components=n_components,
    )
    lagrangian_to_eulerian_grid_interpolation_kernel = (
        eul_lag_communicator.lagrangian_to_eulerian_grid_interpolation_kernel
    )
    # init interp weights as all ones, essentially this should lead to
    # interpolation spreading ones onto the Eulerian grid
    ref_interp_weights = np.ones(
        (
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            num_lag_nodes,
        ),
        dtype=real_t,
    )
    prefactor_lag_field = 2
    ref_lag_grid_field = prefactor_lag_field * np.ones(
        (n_components, num_lag_nodes), dtype=real_t
    )
    # reference integral of interpolated field onto Eulerian grid
    num_ones_in_ref_interp_weights = (
        (2 * interp_kernel_width)
        * (2 * interp_kernel_width)
        * (2 * interp_kernel_width)
    )
    ref_interpolated_field_sum = (
        num_lag_nodes * prefactor_lag_field * num_ones_in_ref_interp_weights
    )

    eul_grid_field = np.zeros(
        (n_components, eul_grid_size, eul_grid_size, eul_grid_size), dtype=real_t
    )
    lagrangian_to_eulerian_grid_interpolation_kernel(
        eul_grid_field,
        ref_lag_grid_field,
        ref_interp_weights,
        ref_nearest_eul_grid_index_to_lag_grid,
    )
    np.testing.assert_allclose(
        np.sum(eul_grid_field, axis=(1, 2, 3)),
        ref_interpolated_field_sum,
        atol=get_test_tol(precision),
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
@pytest.mark.parametrize("interp_kernel_type", ["cosine", "peskin"])
def test_interpolation_weights_kernel_on_nodes_3d(
    interp_kernel_type, n_values, precision
):
    real_t = get_real_t(precision)
    eul_grid_size = n_values
    grid_dim = 3
    dx = real_t(1.0 / eul_grid_size)
    interp_kernel_width = 2
    num_lag_nodes = 3
    eul_grid_coord_shift = real_t(0.5 * dx)
    eul_lag_communicator = EulerianLagrangianGridCommunicator3D(
        dx=dx,
        eul_grid_coord_shift=eul_grid_coord_shift,
        num_lag_nodes=num_lag_nodes,
        interp_kernel_width=interp_kernel_width,
        real_t=real_t,
        interp_kernel_type=interp_kernel_type,
    )
    interpolation_weights_kernel = eul_lag_communicator.interpolation_weights_kernel
    # generate manufactured solution (Z = 0, Y = 1, X = 1)
    eul_grid_support_of_lag_grid = np.zeros(
        (
            grid_dim,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            num_lag_nodes,
        ),
        dtype=real_t,
    )
    eul_grid_support_of_lag_grid[1:] = dx
    # expected answer
    ref_interp_weights = np.ones(
        (
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            2 * interp_kernel_width,
            num_lag_nodes,
        ),
        dtype=real_t,
    ) * real_t(2 * (0.25 / dx) ** grid_dim)

    interp_weights = np.zeros_like(ref_interp_weights)
    interpolation_weights_kernel(interp_weights, eul_grid_support_of_lag_grid)
    np.testing.assert_allclose(
        interp_weights,
        ref_interp_weights,
        atol=get_test_tol(precision),
    )
