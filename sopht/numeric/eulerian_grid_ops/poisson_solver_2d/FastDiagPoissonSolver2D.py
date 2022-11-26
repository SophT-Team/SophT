"""Poisson solver kernels in 2D via Fast Diagonalisation."""
import numpy as np
import numpy.linalg as la

import scipy.sparse as spp


class FastDiagPoissonSolver2D:
    """Class for Poisson solver in 2D via Fast Diagonalisation."""

    def __init__(
        self,
        grid_size_y,
        grid_size_x,
        dx,
        real_t=np.float64,
        bc_type="homogenous_neumann_along_xy",
    ):
        """Class initialiser."""
        self.dx = dx
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.real_t = real_t
        self.bc_type = bc_type

        poisson_matrix_x, poisson_matrix_y = self.construct_poisson_matrices()
        self.apply_boundary_conds_to_poisson_matrices(
            poisson_matrix_x, poisson_matrix_y
        )
        self.compute_spectral_decomp_of_poisson_matrices(
            poisson_matrix_x, poisson_matrix_y
        )

        # allocate buffer for spectral field manipulation
        self.spectral_field_buffer = np.zeros_like(self.inv_eig_val_matrix)

    def construct_poisson_matrices(self):
        """Construct the finite difference Poisson matrices."""
        # TODO can add higher order options..
        inv_dx2 = self.real_t(1 / self.dx / self.dx)
        poisson_matrix_x = inv_dx2 * spp.diags(
            [-1, 2, -1],
            [-1, 0, 1],
            shape=(self.grid_size_x, self.grid_size_x),
            format="csr",
        )
        poisson_matrix_x = poisson_matrix_x.toarray().astype(self.real_t)
        poisson_matrix_y = inv_dx2 * spp.diags(
            [-1, 2, -1],
            [-1, 0, 1],
            shape=(self.grid_size_y, self.grid_size_y),
            format="csr",
        )
        poisson_matrix_y = poisson_matrix_y.toarray().astype(self.real_t)

        return poisson_matrix_x, poisson_matrix_y

    def apply_boundary_conds_to_poisson_matrices(
        self, poisson_matrix_x, poisson_matrix_y
    ):
        """Apply boundary conditions to Poisson matrices."""
        inv_dx2 = self.real_t(1 / self.dx / self.dx)
        if self.bc_type == "homogenous_neumann_along_xy":
            # neumann at x/y=0 and x/y=L, but the modification below operates on
            # nodes at x=dx/2 and x=L-dx/2, because of the grid shift in sims.
            poisson_matrix_x[0, 0] = inv_dx2
            poisson_matrix_x[-1, -1] = inv_dx2
            poisson_matrix_y[0, 0] = inv_dx2
            poisson_matrix_y[-1, -1] = inv_dx2

    def compute_spectral_decomp_of_poisson_matrices(
        self, poisson_matrix_x, poisson_matrix_y
    ):
        """Compute spectral decomposition (eigenvalue and vectors) of the matrices."""
        eig_vals_x, eig_vecs_x = la.eig(poisson_matrix_x)
        # sort eigenvalues in decreasing order
        idx = eig_vals_x.argsort()[::-1]
        eig_vals_x[...] = eig_vals_x[idx]
        eig_vecs_x[...] = eig_vecs_x[:, idx]
        self.tranpose_of_eig_vecs_x = np.transpose(eig_vecs_x)
        self.tranpose_of_inv_of_eig_vecs_x = np.transpose(la.inv(eig_vecs_x))

        eig_vals_y, eig_vecs_y = la.eig(poisson_matrix_y)
        # sort eigenvalues in decreasing order
        idx = eig_vals_y.argsort()[::-1]
        eig_vals_y[...] = eig_vals_y[idx]
        eig_vecs_y[...] = eig_vecs_y[:, idx]
        self.eig_vecs_y = eig_vecs_y
        self.inv_of_eig_vecs_y = la.inv(eig_vecs_y)

        eig_val_matrix = np.tile(
            eig_vals_y.reshape(self.grid_size_y, 1), reps=(1, self.grid_size_x)
        ) + np.tile(eig_vals_x.reshape(1, self.grid_size_x), reps=(self.grid_size_y, 1))
        if self.bc_type == "homogenous_neumann_along_xy":
            # set mean mode to 0, since this bc leads to a null space having the mean mode.
            eig_val_matrix[-1, -1] = np.inf
        self.inv_eig_val_matrix = self.real_t(1) / eig_val_matrix

    def solve(self, solution_field, rhs_field):
        """Solve Poisson equation in 2D: -del^2(solution_field) = rhs_field."""
        # transform to spectral space ("forward transform")
        la.multi_dot(
            [self.inv_of_eig_vecs_y, rhs_field, self.tranpose_of_inv_of_eig_vecs_x],
            out=self.spectral_field_buffer,
        )

        # convolution (elementwise) in spectral space
        np.multiply(
            self.spectral_field_buffer,
            self.inv_eig_val_matrix,
            out=self.spectral_field_buffer,
        )

        # transform to physical space ("backward transform")
        la.multi_dot(
            [self.eig_vecs_y, self.spectral_field_buffer, self.tranpose_of_eig_vecs_x],
            out=solution_field,
        )
