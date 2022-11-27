import numpy as np
import numpy.linalg as la

import psutil

import pytest

from scipy.fft import irfftn, rfftn

from sopht.numeric.eulerian_grid_ops import (
    FastDiagPoissonSolver2D,
)
from sopht.numeric.eulerian_grid_ops import (
    UnboundedPoissonSolverPYFFTW2D,
)
from sopht.utils.precision import get_real_t, get_test_tol


class UnboundedPoissonSolverSolution2D:
    def __init__(self, grid_size_y, grid_size_x, x_range, precision):
        self.real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.grid_size_y = grid_size_y
        self.grid_size_x = grid_size_x
        self.x_range = x_range
        self.y_range = self.x_range * (grid_size_y / grid_size_x)
        self.dx = self.real_t(x_range / grid_size_x)
        self.rhs_field = np.random.randn(self.grid_size_y, self.grid_size_x).astype(
            self.real_t
        )
        self.domain_doubled_rhs_field = np.zeros(
            (2 * self.grid_size_y, 2 * self.grid_size_x), dtype=self.real_t
        )
        complex_dtype = np.complex64 if self.real_t == np.float32 else np.complex128
        self.domain_doubled_fourier_buffer = np.zeros(
            (2 * self.grid_size_y, self.grid_size_x + 1), dtype=complex_dtype
        )
        self.fourier_greens_function_field = (
            self.construct_fourier_greens_function_field()
        )
        self.ref_solution_field = np.zeros_like(self.rhs_field)
        self.ref_solution_field[...] = self.poisson_solve_reference()

    def construct_fourier_greens_function_field(self):
        x_double = np.linspace(
            0, 2 * self.x_range - self.dx, 2 * self.grid_size_x
        ).astype(self.real_t)
        y_double = np.linspace(
            0, 2 * self.y_range - self.dx, 2 * self.grid_size_y
        ).astype(self.real_t)
        # operations after this preserve dtype
        x_grid_double, y_grid_double = np.meshgrid(x_double, y_double)
        even_reflected_distance_field = np.sqrt(
            np.minimum(x_grid_double, 2 * self.x_range - x_grid_double) ** 2
            + np.minimum(y_grid_double, 2 * self.y_range - y_grid_double) ** 2
        )
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            greens_function_field = -np.log(even_reflected_distance_field) / (2 * np.pi)
        # Regularization term
        greens_function_field[0, 0] = -(2 * np.log(self.dx / np.sqrt(np.pi)) - 1) / (
            4 * np.pi
        )
        return rfftn(greens_function_field)

    def poisson_solve_reference(self):
        self.domain_doubled_rhs_field[...] = 0
        self.domain_doubled_rhs_field[
            : self.grid_size_y, : self.grid_size_x
        ] = self.rhs_field
        self.domain_doubled_fourier_buffer[...] = rfftn(self.domain_doubled_rhs_field)
        # Greens function convolution
        self.domain_doubled_fourier_buffer[
            ...
        ] *= self.fourier_greens_function_field * (self.dx**2)
        return irfftn(self.domain_doubled_fourier_buffer)[
            : self.grid_size_y, : self.grid_size_x
        ]

    def check_equals(self, solution_field):
        np.testing.assert_allclose(
            self.ref_solution_field, solution_field, atol=self.test_tol
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_poisson_solve_pyfftw_2d(n_values, precision):
    real_t = get_real_t(precision)
    x_range = real_t(2.0)
    solution = UnboundedPoissonSolverSolution2D(
        grid_size_y=n_values, grid_size_x=n_values, x_range=x_range, precision=precision
    )
    unbounded_poisson_solver = UnboundedPoissonSolverPYFFTW2D(
        grid_size_y=n_values,
        grid_size_x=n_values,
        x_range=x_range,
        real_t=real_t,
        num_threads=psutil.cpu_count(logical=False),
    )
    solution_field = np.zeros_like(solution.rhs_field)
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.solve
    unbounded_poisson_solver_kernel(
        solution_field=solution_field, rhs_field=solution.rhs_field
    )
    # assert correct
    solution.check_equals(solution_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_unbounded_poisson_solve_neumann_fast_diag_2d(n_values, precision):
    real_t = get_real_t(precision)
    dx = real_t(1.0 / n_values)
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, n_values).astype(real_t)
    x_grid, y_grid = np.meshgrid(x, x)
    wave_number = real_t(2)
    rhs_field = np.cos(wave_number * np.pi * x_grid) * np.cos(
        wave_number * np.pi * y_grid
    )
    correct_solution_field = rhs_field / real_t(2 * (wave_number * np.pi) ** 2)
    solution_field = np.zeros_like(rhs_field)

    unbounded_poisson_solver = FastDiagPoissonSolver2D(
        grid_size_y=n_values,
        grid_size_x=n_values,
        dx=dx,
        real_t=real_t,
        bc_type="homogenous_neumann_along_xy",
    )
    unbounded_poisson_solver_kernel = unbounded_poisson_solver.solve
    unbounded_poisson_solver_kernel(solution_field=solution_field, rhs_field=rhs_field)

    error_field = solution_field - correct_solution_field
    l2_norm_error = la.norm(error_field) * dx
    linf_norm_error = np.amax(np.fabs(error_field))
    # check both errors less than 0.1%
    error_tol = real_t(1e-3)
    assert l2_norm_error <= error_tol
    assert linf_norm_error <= error_tol
