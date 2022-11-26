import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_penalise_field_boundary_pyst_kernel_3d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def penalise_field_boundary_reference_3d(
    penalised_field,
    field,
    x_grid_field,
    y_grid_field,
    z_grid_field,
    width,
    dx,
):
    x_grid_field_start = x_grid_field[0, 0, 0]
    y_grid_field_start = y_grid_field[0, 0, 0]
    z_grid_field_start = z_grid_field[0, 0, 0]
    x_grid_field_end = x_grid_field[0, 0, -1]
    y_grid_field_end = y_grid_field[0, -1, 0]
    z_grid_field_end = z_grid_field[-1, 0, 0]
    sine_prefactor = np.pi / 2 / width / dx
    # first along x
    penalised_field[:, :, :width] = (
        np.sin((x_grid_field[:, :, :width] - x_grid_field_start) * sine_prefactor)
        * field[:, :, (width - 1) : width]
    )
    penalised_field[:, :, -width:] = (
        np.sin((x_grid_field_end - x_grid_field[:, :, -width:]) * sine_prefactor)
        * field[:, :, -width : (-width + 1)]
    )
    # then along y
    penalised_field[:, :width, :] = (
        np.sin((y_grid_field[:, :width, :] - y_grid_field_start) * sine_prefactor)
        * penalised_field[:, (width - 1) : width, :]
    )
    penalised_field[:, -width:, :] = (
        np.sin((y_grid_field_end - y_grid_field[:, -width:, :]) * sine_prefactor)
        * penalised_field[:, -width : (-width + 1), :]
    )
    # then along z
    penalised_field[:width, :, :] = (
        np.sin((z_grid_field[:width, :, :] - z_grid_field_start) * sine_prefactor)
        * penalised_field[width - 1 : width, :, :]
    )
    penalised_field[-width:, :, :] = (
        np.sin((z_grid_field_end - z_grid_field[-width:, :, :]) * sine_prefactor)
        * penalised_field[-width : (-width + 1), :, :]
    )


class PenaliseFieldBoundarySolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples, n_samples).astype(real_t)
        self.width = 4
        self.dx = real_t(0.1)
        self.grid_coord_shift = real_t(self.dx / 2)
        x = np.linspace(
            self.grid_coord_shift, 1 - self.grid_coord_shift, n_samples
        ).astype(real_t)
        y = x.copy()
        z = x.copy()
        self.z_grid_field, self.y_grid_field, self.x_grid_field = np.meshgrid(
            z, y, x, indexing="ij"
        )
        self.ref_penalised_field = self.ref_field.copy()
        penalise_field_boundary_reference_3d(
            self.ref_penalised_field,
            self.ref_field,
            self.x_grid_field,
            self.y_grid_field,
            self.z_grid_field,
            self.width,
            self.dx,
        )
        self.ref_vector_field = np.random.randn(
            3, n_samples, n_samples, n_samples
        ).astype(real_t)
        self.ref_penalised_vector_field = self.ref_vector_field.copy()
        for i in range(3):
            penalise_field_boundary_reference_3d(
                self.ref_penalised_vector_field[i],
                self.ref_vector_field[i],
                self.x_grid_field,
                self.y_grid_field,
                self.z_grid_field,
                self.width,
                self.dx,
            )

    def check_equals(self, penalised_field):
        np.testing.assert_allclose(
            self.ref_penalised_field,
            penalised_field,
            atol=self.test_tol,
        )

    def check_vector_field_equals(self, penalised_vector_field):
        np.testing.assert_allclose(
            self.ref_penalised_vector_field,
            penalised_vector_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_penalise_field_boundary_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = PenaliseFieldBoundarySolution(n_values, precision)
    field = solution.ref_field.copy()
    penalise_field_towards_boundary_pyst_kernel = (
        gen_penalise_field_boundary_pyst_kernel_3d(
            width=solution.width,
            dx=solution.dx,
            x_grid_field=solution.x_grid_field,
            y_grid_field=solution.y_grid_field,
            z_grid_field=solution.z_grid_field,
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="scalar",
        )
    )
    penalise_field_towards_boundary_pyst_kernel(field=field)
    solution.check_equals(field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_penalise_vector_field_boundary_3d(n_values, precision):
    real_t = get_real_t(precision)
    solution = PenaliseFieldBoundarySolution(n_values, precision)
    vector_field = solution.ref_vector_field.copy()
    penalise_field_towards_boundary_pyst_kernel = (
        gen_penalise_field_boundary_pyst_kernel_3d(
            width=solution.width,
            dx=solution.dx,
            x_grid_field=solution.x_grid_field,
            y_grid_field=solution.y_grid_field,
            z_grid_field=solution.z_grid_field,
            real_t=real_t,
            fixed_grid_size=(n_values, n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="vector",
        )
    )
    penalise_field_towards_boundary_pyst_kernel(vector_field=vector_field)
    solution.check_vector_field_equals(vector_field)
