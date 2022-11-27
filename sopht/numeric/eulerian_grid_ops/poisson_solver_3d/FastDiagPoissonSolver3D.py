"""Poisson solver kernels in 3D via Fast Diagonalisation."""
import numpy as np
import numpy.linalg as la

import scipy.sparse as spp


class FastDiagPoissonSolver3D:
    """Class for Poisson solver in 3D via Fast Diagonalisation."""

    def __init__(
        self,
        grid_size_z,
        grid_size_y,
        grid_size_x,
        dx,
        real_t=np.float64,
        bc_type="homogenous_neumann_along_xyz",
    ):
        """Class initialiser."""
        self.dx = dx
        self.grid_size_z = grid_size_z
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.real_t = real_t
        self.bc_type = bc_type

        (
            poisson_matrix_x,
            poisson_matrix_y,
            poisson_matrix_z,
        ) = self.construct_poisson_matrices()
        self.apply_boundary_conds_to_poisson_matrices(
            poisson_matrix_x, poisson_matrix_y, poisson_matrix_z
        )
        self.compute_spectral_decomp_of_poisson_matrices(
            poisson_matrix_x, poisson_matrix_y, poisson_matrix_z
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
        poisson_matrix_z = inv_dx2 * spp.diags(
            [-1, 2, -1],
            [-1, 0, 1],
            shape=(self.grid_size_z, self.grid_size_z),
            format="csr",
        )
        poisson_matrix_z = poisson_matrix_z.toarray().astype(self.real_t)

        return poisson_matrix_x, poisson_matrix_y, poisson_matrix_z

    def apply_boundary_conds_to_poisson_matrices(
        self,
        poisson_matrix_x,
        poisson_matrix_y,
        poisson_matrix_z,
    ):
        """Apply boundary conditions to Poisson matrices."""
        inv_dx2 = self.real_t(1 / self.dx / self.dx)
        if self.bc_type == "homogenous_neumann_along_xyz":
            # neumann at x/y/z=0 and x/y/z=L, but the modification below operates on
            # nodes at x=dx/2 and x=L-dx/2, because of the grid shift in sims.
            poisson_matrix_x[0, 0] = inv_dx2
            poisson_matrix_x[-1, -1] = inv_dx2
            poisson_matrix_y[0, 0] = inv_dx2
            poisson_matrix_y[-1, -1] = inv_dx2
            poisson_matrix_z[0, 0] = inv_dx2
            poisson_matrix_z[-1, -1] = inv_dx2

    def compute_spectral_decomp_of_poisson_matrices(
        self, poisson_matrix_x, poisson_matrix_y, poisson_matrix_z
    ):
        """Compute spectral decomposition (eigenvalue and vectors) of the matrices."""
        eig_vals_x, eig_vecs_x = la.eig(poisson_matrix_x)
        # sort eigenvalues in decreasing order
        idx = eig_vals_x.argsort()[::-1]
        eig_vals_x[...] = eig_vals_x[idx]
        eig_vecs_x[...] = eig_vecs_x[:, idx]
        self.eig_vecs_x = eig_vecs_x
        self.inv_of_eig_vecs_x = la.inv(eig_vecs_x)

        eig_vals_y, eig_vecs_y = la.eig(poisson_matrix_y)
        # sort eigenvalues in decreasing order
        idx = eig_vals_y.argsort()[::-1]
        eig_vals_y[...] = eig_vals_y[idx]
        eig_vecs_y[...] = eig_vecs_y[:, idx]
        self.eig_vecs_y = eig_vecs_y
        self.inv_of_eig_vecs_y = la.inv(eig_vecs_y)

        eig_vals_z, eig_vecs_z = la.eig(poisson_matrix_z)
        # sort eigenvalues in decreasing order
        idx = eig_vals_z.argsort()[::-1]
        eig_vals_z[...] = eig_vals_z[idx]
        eig_vecs_z[...] = eig_vecs_z[:, idx]
        self.eig_vecs_z = eig_vecs_z
        self.inv_of_eig_vecs_z = la.inv(eig_vecs_z)

        eig_val_matrix = (
            np.tile(
                eig_vals_z.reshape(self.grid_size_z, 1, 1),
                reps=(1, self.grid_size_y, self.grid_size_x),
            )
            + np.tile(
                eig_vals_y.reshape(1, self.grid_size_y, 1),
                reps=(self.grid_size_z, 1, self.grid_size_x),
            )
            + np.tile(
                eig_vals_x.reshape(1, 1, self.grid_size_x),
                reps=(self.grid_size_z, self.grid_size_y, 1),
            )
        )
        if self.bc_type == "homogenous_neumann_along_xyz":
            # set mean mode to 0, since this bc leads to a null space having the mean mode.
            eig_val_matrix[-1, -1, -1] = np.inf
        self.inv_eig_val_matrix = self.real_t(1) / eig_val_matrix

    def solve(self, solution_field, rhs_field):
        """Solve Poisson equation in 3D: -del^2(solution_field) = rhs_field."""
        # transform to spectral space ("forward transform")
        # hit last x index
        self.spectral_field_buffer[...] = np.tensordot(
            rhs_field, self.inv_of_eig_vecs_x, axes=(2, 1)
        )
        # hit middle y index
        self.spectral_field_buffer[...] = np.tensordot(
            self.inv_of_eig_vecs_y, self.spectral_field_buffer, axes=(1, 1)
        ).transpose((1, 0, 2))
        # hit first z index
        self.spectral_field_buffer[...] = np.tensordot(
            self.inv_of_eig_vecs_z, self.spectral_field_buffer, axes=(1, 0)
        )

        # convolution (elementwise) in spectral space
        np.multiply(
            self.spectral_field_buffer,
            self.inv_eig_val_matrix,
            out=self.spectral_field_buffer,
        )

        # transform to physical space ("backward transform")
        # hit last x index
        self.spectral_field_buffer[...] = np.tensordot(
            self.spectral_field_buffer, self.eig_vecs_x, axes=(2, 1)
        )
        # hit middle y index
        self.spectral_field_buffer[...] = np.tensordot(
            self.eig_vecs_y, self.spectral_field_buffer, axes=(1, 1)
        ).transpose((1, 0, 2))
        # hit first z index
        solution_field[...] = np.tensordot(
            self.eig_vecs_z, self.spectral_field_buffer, axes=(1, 0)
        )

    def vector_field_solve(self, solution_vector_field, rhs_vector_field):
        """Poisson equation solver method in 3D.

        Solves 3 Poisson equations in 3D for each of the components of:
        solution_vector_field and rhs_vector_field.
        """
        self.solve(
            solution_field=solution_vector_field[0], rhs_field=rhs_vector_field[0]
        )
        self.solve(
            solution_field=solution_vector_field[1], rhs_field=rhs_vector_field[1]
        )
        self.solve(
            solution_field=solution_vector_field[2], rhs_field=rhs_vector_field[2]
        )
