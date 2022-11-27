import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_brinkmann_penalise_pyst_kernel_2d,
    gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t, get_test_tol


def brinkmann_penalise_reference(
    penalty_factor, char_field, penalty_field, field, real_t
):
    penalised_field = np.zeros_like(field)
    penalised_field[...] = (field + penalty_factor * char_field * penalty_field) / (
        real_t(1) + penalty_factor * char_field
    )
    return penalised_field


class BrinkmannPenalisationSolution:
    def __init__(self, n_samples, precision="single"):
        real_t = get_real_t(precision)
        self.test_tol = get_test_tol(precision)
        self.ref_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_penalty_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.ref_char_field = np.random.randn(n_samples, n_samples).astype(real_t)
        self.penalty_factor = real_t(0.1)
        self.ref_penalised_field = brinkmann_penalise_reference(
            self.penalty_factor,
            self.ref_char_field,
            self.ref_penalty_field,
            self.ref_field,
            real_t,
        )

        self.ref_vector_field = np.random.randn(2, n_samples, n_samples).astype(real_t)
        self.ref_penalty_vector_field = np.random.randn(2, n_samples, n_samples).astype(
            real_t
        )
        self.ref_penalised_vector_field = brinkmann_penalise_reference(
            self.penalty_factor,
            self.ref_char_field,
            self.ref_penalty_vector_field,
            self.ref_vector_field,
            real_t,
        )

    def check_equals(self, penalized_field):
        np.testing.assert_allclose(
            self.ref_penalised_field,
            penalized_field,
            atol=self.test_tol,
        )

    def check_vector_field_equals(self, penalized_vector_field):
        np.testing.assert_allclose(
            self.ref_penalised_vector_field,
            penalized_vector_field,
            atol=self.test_tol,
        )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalise_scalar_field_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = BrinkmannPenalisationSolution(n_values, precision)
    penalised_field = np.zeros_like(solution.ref_penalised_field)
    brinkmann_penalise_pyst_kernel = gen_brinkmann_penalise_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    brinkmann_penalise_pyst_kernel(
        penalised_field=penalised_field,
        penalty_factor=solution.penalty_factor,
        char_field=solution.ref_char_field,
        penalty_field=solution.ref_penalty_field,
        field=solution.ref_field,
    )
    solution.check_equals(penalised_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalise_vector_field_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = BrinkmannPenalisationSolution(n_values, precision)
    penalised_vector_field = np.zeros_like(solution.ref_penalised_vector_field)
    brinkmann_penalise_vector_field_pyst_kernel = gen_brinkmann_penalise_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    brinkmann_penalise_vector_field_pyst_kernel(
        penalised_vector_field=penalised_vector_field,
        penalty_factor=solution.penalty_factor,
        char_field=solution.ref_char_field,
        penalty_vector_field=solution.ref_penalty_vector_field,
        vector_field=solution.ref_vector_field,
    )
    solution.check_vector_field_equals(penalised_vector_field)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalise_scalar_field_vs_fixed_val_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = BrinkmannPenalisationSolution(n_values, precision)
    penalised_field = np.zeros_like(solution.ref_penalised_field)
    brinkmann_penalise_field_vs_fixed_val_pyst_kernel = (
        gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="scalar",
        )
    )
    penalty_val = real_t(2.0)
    brinkmann_penalise_field_vs_fixed_val_pyst_kernel(
        penalised_field=penalised_field,
        penalty_factor=solution.penalty_factor,
        char_field=solution.ref_char_field,
        penalty_val=penalty_val,
        field=solution.ref_field,
    )
    np.testing.assert_allclose(
        penalised_field,
        (
            solution.ref_field
            + solution.penalty_factor * solution.ref_char_field * penalty_val
        )
        / (1 + solution.penalty_factor * solution.ref_char_field),
        atol=solution.test_tol,
    )


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_brinkmann_penalise_vector_field_vs_fixed_val_2d(n_values, precision):
    real_t = get_real_t(precision)
    solution = BrinkmannPenalisationSolution(n_values, precision)
    penalised_vector_field = np.zeros_like(solution.ref_penalised_vector_field)
    brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel = (
        gen_brinkmann_penalise_vs_fixed_val_pyst_kernel_2d(
            real_t=real_t,
            fixed_grid_size=(n_values, n_values),
            num_threads=psutil.cpu_count(logical=False),
            field_type="vector",
        )
    )
    penalty_val = np.array([2.0, 3.0]).astype(real_t)
    brinkmann_penalise_vector_field_vs_fixed_val_pyst_kernel(
        penalised_vector_field=penalised_vector_field,
        penalty_factor=solution.penalty_factor,
        char_field=solution.ref_char_field,
        penalty_val=penalty_val,
        vector_field=solution.ref_vector_field,
    )
    np.testing.assert_allclose(
        penalised_vector_field[0],
        (
            solution.ref_vector_field[0]
            + solution.penalty_factor * solution.ref_char_field * penalty_val[0]
        )
        / (1 + solution.penalty_factor * solution.ref_char_field),
        atol=solution.test_tol,
    )
    np.testing.assert_allclose(
        penalised_vector_field[1],
        (
            solution.ref_vector_field[1]
            + solution.penalty_factor * solution.ref_char_field * penalty_val[1]
        )
        / (1 + solution.penalty_factor * solution.ref_char_field),
        atol=solution.test_tol,
    )
