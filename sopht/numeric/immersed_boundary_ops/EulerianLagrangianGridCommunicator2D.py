"""Eulerian-Lagrangian grid communicator in 2D."""
from numba import njit

import numpy as np


class EulerianLagrangianGridCommunicator2D:
    """Class for communication between Eulerian and Lagrangian grids in 2D.

    Sets up a communicator between Eulerian and Lagrangian grids in the
    domain, which consists of:
    1. Find grid intersections (nearest indices)
    2. Interpolate fields back and forth
    3. Compute interpolation weights for interpolation
    TODO add proper style docs
    """

    def __init__(
        self,
        dx,
        eul_grid_coord_shift,
        num_lag_nodes,
        interp_kernel_width,
        real_t,
        n_components=1,
        interp_kernel_type="cosine",
    ):
        """Class initialiser."""
        self.local_eulerian_grid_support_of_lagrangian_grid_kernel = (
            generate_local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
                dx=dx,
                eul_grid_coord_shift=eul_grid_coord_shift,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
            )
        )
        self.eulerian_to_lagrangian_grid_interpolation_kernel = (
            generate_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
                dx=dx,
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                n_components=n_components,
            )
        )
        self.lagrangian_to_eulerian_grid_interpolation_kernel = (
            generate_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
                num_lag_nodes=num_lag_nodes,
                interp_kernel_width=interp_kernel_width,
                n_components=n_components,
            )
        )
        if interp_kernel_type == "peskin":
            self.interpolation_weights_kernel = (
                generate_peskin_interpolation_weights_kernel_2d(
                    dx=dx, interp_kernel_width=interp_kernel_width, real_t=real_t
                )
            )
        elif interp_kernel_type == "cosine":
            self.interpolation_weights_kernel = (
                generate_cosine_interpolation_weights_kernel_2d(
                    dx=dx, interp_kernel_width=interp_kernel_width, real_t=real_t
                )
            )
        else:
            raise ValueError(
                "Invalid interpolation kernel type. Current supported types are"
                "'cosine' and 'peskin'."
            )


def generate_local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
    dx, eul_grid_coord_shift, num_lag_nodes, interp_kernel_width
):
    """
    Generate kernel that computes local Eulerian support of Lagrangian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx: Eulerian grid spacing
    eul_grid_coord_shift: shift of the coordinates of the Eulerian grid start from
    0 (usually dx / 2)
    num_lag_nodes: number of Lagrangian grid nodes

    """
    # grid/problem dimensions
    grid_dim = 2
    x = np.arange(-interp_kernel_width + 1, interp_kernel_width + 1)
    x_grid, y_grid = np.meshgrid(x, x)
    local_eul_grid_support_idx = np.stack((x_grid, y_grid))
    # The tiled version of the above array allows us to do broadcast operations
    # and thus vectorised calls in the following function. Could this become an
    # excessive memory allocation issue when num_lag_nodes is large? Hence, as
    # a fallback we keep the for loop version, commented out after the function below.
    local_eul_grid_support_indices = np.tile(
        local_eul_grid_support_idx.reshape(
            grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, 1
        ),
        reps=(1, 1, 1, num_lag_nodes),
    )

    @njit(cache=True, fastmath=True)
    def local_eulerian_grid_support_of_lagrangian_grid_kernel_2d(
        local_eul_grid_support_of_lag_grid,
        nearest_eul_grid_index_to_lag_grid,
        lag_positions,
    ):
        """Compute local Eulerian support of Lagrangian grid.

        Return nearest_eul_grid_index_to_lag_grid: size (grid_dim, num_lag_nodes)
        local_eul_grid_support_of_lag_grid: local Eulerian grid support of the Lagrangian grid
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)
        from input params:
        lag_positions: (grid_dim, num_lag_nodes)

        """
        # dtype of nearest_grid_index takes care of type casting to int
        nearest_eul_grid_index_to_lag_grid[...] = np.rint(
            (lag_positions - eul_grid_coord_shift) / dx
        )
        # TODO We need to add boundary exception handling! where the Lagrangian
        #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
        # get relative distance (support) of body
        # reshape done to broadcast
        local_eul_grid_support_of_lag_grid[...] = (
            (
                nearest_eul_grid_index_to_lag_grid.reshape(
                    grid_dim, 1, 1, num_lag_nodes
                )
                + local_eul_grid_support_indices
            )
            * dx
            + eul_grid_coord_shift
            - lag_positions.reshape(grid_dim, 1, 1, num_lag_nodes)
        )

    """
    # the version below is slower for num_lag_nodes < 500 for sure...
    @njit(cache=True, fastmath=True)
    def local_eulerian_grid_support_of_lagrangian_grid_kernel(
        local_eul_grid_support_of_lag_grid, nearest_eul_grid_index_to_lag_grid, lag_positions
    ):
        # dtype of nearest_grid_index takes care of type casting to int
        nearest_eul_grid_index_to_lag_grid[...] = (
            lag_positions - eul_grid_coord_shift
        ) // dx
        # Can we vectorize this?
        for i in range(0, num_lag_nodes):
            local_eul_grid_support_of_lag_grid[0, ..., i] = (
                (
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    + local_eul_grid_support_idx[0]
                )
                * dx
                + eul_grid_coord_shift
                - lag_positions[0, i]
            )
            local_eul_grid_support_of_lag_grid[1, ..., i] = (
                (
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    + local_eul_grid_support_idx[1]
                )
                * dx
                + eul_grid_coord_shift
                - lag_positions[0, i]
            )
    """
    return local_eulerian_grid_support_of_lagrangian_grid_kernel_2d


def generate_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
    dx, num_lag_nodes, interp_kernel_width, n_components
):
    """Generate kernel that interpolates a field from an Eulerian grid to a Lagrangian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx: Eulerian grid spacing
    num_lag_nodes: number of Lagrangian nodes
    n_components : number of components in Lagrangian field

    """
    assert (
        n_components == 1 or n_components == 2
    ), "invalid number of components for interpolation!"
    # grid/problem dimensions
    grid_dim = 2

    @njit(cache=True, fastmath=True)
    def eulerian_to_lagrangian_grid_interpolation_kernel_2d(
        lag_grid_field,
        eul_grid_field,
        interp_weights,
        nearest_eul_grid_index_to_lag_grid,
    ):
        """Interpolate an Eulerian field onto a Lagrangian field.

        Inputs:
        the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
        interpolation weights interp_weights of
        shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

        """
        # TODO We need to add boundary exception handling! where the Lagrangian
        #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
        for i in range(0, num_lag_nodes):
            lag_grid_field[i] = np.sum(
                eul_grid_field[
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                    + interp_kernel_width
                    + 1,
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                    + interp_kernel_width
                    + 1,
                ]
                * interp_weights[..., i]
            ) * (dx**grid_dim)

    @njit(cache=True, fastmath=True)
    def vector_field_eulerian_to_lagrangian_grid_interpolation_kernel_2d(
        lag_grid_field,
        eul_grid_field,
        interp_weights,
        nearest_eul_grid_index_to_lag_grid,
    ):
        """Interpolate an Eulerian vector field onto a Lagrangian vector field.

        Inputs:
        the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
        interpolation weights interp_weights of
        shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

        """
        # TODO We need to add boundary exception handling! where the Lagrangian
        #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
        for i in range(0, num_lag_nodes):
            # numba doesnt allow multiple axes for np.sum :/,
            # hence needs to be done serially
            lag_grid_field[0, i] = np.sum(
                eul_grid_field[
                    0,
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                    + interp_kernel_width
                    + 1,
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                    + interp_kernel_width
                    + 1,
                ]
                * interp_weights[..., i]
            ) * (dx**grid_dim)
            lag_grid_field[1, i] = np.sum(
                eul_grid_field[
                    1,
                    nearest_eul_grid_index_to_lag_grid[1, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                    + interp_kernel_width
                    + 1,
                    nearest_eul_grid_index_to_lag_grid[0, i]
                    - interp_kernel_width
                    + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                    + interp_kernel_width
                    + 1,
                ]
                * interp_weights[..., i]
            ) * (dx**grid_dim)

    if n_components == 1:
        return eulerian_to_lagrangian_grid_interpolation_kernel_2d
    else:
        return vector_field_eulerian_to_lagrangian_grid_interpolation_kernel_2d


def generate_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
    num_lag_nodes, interp_kernel_width, n_components=1
):
    """Generate kernel that interpolates a field from a Lagrangian grid to an Eulerian grid.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    num_lag_nodes: number of Lagrangian nodes
    n_components : number of components in Lagrangian field

    """
    assert (
        n_components == 1 or n_components == 2
    ), "invalid number of components for interpolation!"

    @njit(cache=True, fastmath=True)
    def lagrangian_to_eulerian_grid_interpolation_kernel_2d(
        eul_grid_field,
        lag_grid_field,
        interp_weights,
        nearest_eul_grid_index_to_lag_grid,
    ):
        """Interpolate a Lagrangian field onto an Eulerian field.

        Inputs:
        the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
        interpolation weights interp_weights of
        shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

        """
        # TODO We need to add boundary exception handling! where the Lagrangian
        #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
        for i in range(0, num_lag_nodes):
            eul_grid_field[
                nearest_eul_grid_index_to_lag_grid[1, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                + interp_kernel_width
                + 1,
                nearest_eul_grid_index_to_lag_grid[0, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                + interp_kernel_width
                + 1,
            ] += (
                lag_grid_field[..., i] * interp_weights[..., i]
            )

    @njit(cache=True, fastmath=True)
    def vector_field_lagrangian_to_eulerian_grid_interpolation_kernel_2d(
        eul_grid_field,
        lag_grid_field,
        interp_weights,
        nearest_eul_grid_index_to_lag_grid,
    ):
        """Interpolate a Lagrangian vector field onto an Eulerian field.

        Inputs:
        the nearest_eul_grid_index_to_lag_grid(grid_dim, num_lag_nodes) and
        interpolation weights interp_weights of
        shape (2 * interp_kernel_width, 2 * interp_kernel_width, num_lag_nodes)

        """
        # TODO We need to add boundary exception handling! where the Lagrangian
        #  node goes in `interp_kernel_width` boundary zone of the Eulerian grid
        for i in range(0, num_lag_nodes):
            eul_grid_field[
                ...,
                nearest_eul_grid_index_to_lag_grid[1, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                + interp_kernel_width
                + 1,
                nearest_eul_grid_index_to_lag_grid[0, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                + interp_kernel_width
                + 1,
            ] += (
                np.ascontiguousarray(lag_grid_field[..., i]).reshape(-1, 1, 1)
                * interp_weights[..., i]
            )
            """
            eul_grid_field[
                0,
                nearest_eul_grid_index_to_lag_grid[1, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                + interp_kernel_width
                + 1,
                nearest_eul_grid_index_to_lag_grid[0, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                + interp_kernel_width
                + 1,
            ] += (
                lag_grid_field[0, i] * interp_weights[..., i]
            )
            eul_grid_field[
                1,
                nearest_eul_grid_index_to_lag_grid[1, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[1, i]
                + interp_kernel_width
                + 1,
                nearest_eul_grid_index_to_lag_grid[0, i]
                - interp_kernel_width
                + 1 : nearest_eul_grid_index_to_lag_grid[0, i]
                + interp_kernel_width
                + 1,
            ] += (
                lag_grid_field[1, i] * interp_weights[..., i]
            )
            """

    if n_components == 1:
        return lagrangian_to_eulerian_grid_interpolation_kernel_2d
    else:
        return vector_field_lagrangian_to_eulerian_grid_interpolation_kernel_2d


def generate_cosine_interpolation_weights_kernel_2d(dx, interp_kernel_width, real_t):
    """Generate the kernel for computing interpolation weights using 2D cosine delta function.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx : Eulerian grid spacing

    """
    # grid/problem dimensions
    grid_dim = 2
    assert (
        interp_kernel_width == 2
    ), "Interpolation kernel inconsistent with interpolation kernel width!"

    @njit(cache=True, fastmath=True)
    def cosine_interpolation_weights_kernel_2d(
        interp_weights, local_eul_grid_support_of_lag_grid
    ):
        """Compute the interpolation weights using 2D cosine delta function.

        Result stored in interp_weights of shape
        (2 * interp_kernel_width, 2 * interp_kernel_width, ...) with
        input as eul_grid_support_of_lag_grid of shape
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, ...)
        Applicable for interp_kernel_width = 2
        """
        local_eul_grid_support_of_lag_grid /= dx
        interp_weights[...] = (
            real_t((0.25 / dx) ** grid_dim)
            * (
                real_t(1.0)
                + np.cos(real_t(0.5 * np.pi) * local_eul_grid_support_of_lag_grid[0])
            )
            * (
                real_t(1.0)
                + np.cos(real_t(0.5 * np.pi) * local_eul_grid_support_of_lag_grid[1])
            )
        )

    return cosine_interpolation_weights_kernel_2d


def generate_peskin_interpolation_weights_kernel_2d(dx, interp_kernel_width, real_t):
    """Generate the kernel for computing interpolation weights proposed by Peskin, 2002, 6.27.

    Inputs:
    interp_kernel_width: width of interpolation kernel
    dx : Eulerian grid spacing

    """
    # grid/problem dimensions
    grid_dim = 2
    assert (
        interp_kernel_width == 2
    ), "Interpolation kernel inconsistent with interpolation kernel width!"

    @njit(cache=True, fastmath=True)
    def peskin_interpolation_weights_kernel_2d(
        interp_weights, local_eul_grid_support_of_lag_grid
    ):
        """Compute the interpolation weights using 2D delta function by Peskin, 2002.

        Result stored in interp_weights of shape
        (2 * interp_kernel_width, 2 * interp_kernel_width, ...) with
        input as eul_grid_support_of_lag_grid of shape
        (grid_dim, 2 * interp_kernel_width, 2 * interp_kernel_width, ...)
        Applicable for interp_kernel_width = 2
        """
        local_eul_grid_support_of_lag_grid[...] = (
            np.fabs(local_eul_grid_support_of_lag_grid) / dx
        )
        interp_weights[...] = (
            (0.125 / dx) ** grid_dim
            * (
                (local_eul_grid_support_of_lag_grid[0] < 1.0)
                * (
                    3.0
                    - 2 * local_eul_grid_support_of_lag_grid[0]
                    + np.sqrt(
                        np.fabs(
                            1
                            + 4 * local_eul_grid_support_of_lag_grid[0]
                            - 4 * local_eul_grid_support_of_lag_grid[0] ** 2
                        )
                    )
                )
                + (local_eul_grid_support_of_lag_grid[0] >= 1.0)
                * (local_eul_grid_support_of_lag_grid[0] < 2.0)
                * (
                    5.0
                    - 2 * local_eul_grid_support_of_lag_grid[0]
                    - np.sqrt(
                        np.fabs(
                            -7
                            + 12 * local_eul_grid_support_of_lag_grid[0]
                            - 4 * local_eul_grid_support_of_lag_grid[0] ** 2
                        )
                    )
                )
            )
            * (
                (local_eul_grid_support_of_lag_grid[1] < 1.0)
                * (
                    3.0
                    - 2 * local_eul_grid_support_of_lag_grid[1]
                    + np.sqrt(
                        np.fabs(
                            1
                            + 4 * local_eul_grid_support_of_lag_grid[1]
                            - 4 * local_eul_grid_support_of_lag_grid[1] ** 2
                        )
                    )
                )
                + (local_eul_grid_support_of_lag_grid[1] >= 1.0)
                * (local_eul_grid_support_of_lag_grid[1] < 2.0)
                * (
                    5.0
                    - 2 * local_eul_grid_support_of_lag_grid[1]
                    - np.sqrt(
                        np.fabs(
                            -7
                            + 12 * local_eul_grid_support_of_lag_grid[1]
                            - 4 * local_eul_grid_support_of_lag_grid[1] ** 2
                        )
                    )
                )
            )
        ).astype(real_t)

    return peskin_interpolation_weights_kernel_2d
