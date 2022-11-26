import numpy as np

import psutil

import pytest

from sopht.numeric.eulerian_grid_ops import (
    gen_add_fixed_val_pyst_kernel_2d,
    gen_elementwise_complex_product_pyst_kernel_2d,
    gen_elementwise_copy_pyst_kernel_2d,
    gen_elementwise_sum_pyst_kernel_2d,
    gen_set_fixed_val_at_boundaries_pyst_kernel_2d,
    gen_set_fixed_val_pyst_kernel_2d,
    gen_elementwise_saxpby_pyst_kernel_2d,
)
from sopht.utils.precision import get_real_t


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_elementwise_sum_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_sum_pyst_kernel = gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    field_1 = 2 * np.ones((n_values, n_values), dtype=real_t)
    field_2 = 3 * np.ones((n_values, n_values), dtype=real_t)
    sum_field = np.zeros_like(field_1)
    elementwise_sum_pyst_kernel(
        sum_field=sum_field,
        field_1=field_1,
        field_2=field_2,
    )
    np.testing.assert_allclose(sum_field, 5)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_elementwise_sum_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_sum_pyst_kernel = gen_elementwise_sum_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    field_1 = 2 * np.ones((2, n_values, n_values), dtype=real_t)
    field_2 = 3 * np.ones((2, n_values, n_values), dtype=real_t)
    sum_field = np.zeros_like(field_1)
    elementwise_sum_pyst_kernel(
        sum_field=sum_field,
        field_1=field_1,
        field_2=field_2,
    )
    np.testing.assert_allclose(sum_field, 5)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_set_fixed_val_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    set_fixed_val_pyst_kernel = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    field = np.ones((n_values, n_values), dtype=real_t)
    fixed_val = real_t(3)
    set_fixed_val_pyst_kernel(
        field=field,
        fixed_val=fixed_val,
    )
    np.testing.assert_allclose(field, fixed_val)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_set_fixed_val_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    set_fixed_val_pyst_kernel = gen_set_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    field = np.ones((2, n_values, n_values), dtype=real_t)
    fixed_vals = np.array((2, 3), dtype=real_t)
    set_fixed_val_pyst_kernel(
        vector_field=field,
        fixed_vals=fixed_vals,
    )
    np.testing.assert_allclose(field[0], 2)
    np.testing.assert_allclose(field[1], 3)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_elementwise_copy_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_copy_pyst_kernel = gen_elementwise_copy_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
    )
    field = 2 * np.ones((n_values, n_values), dtype=real_t)
    rhs_field = 3 * np.ones((n_values, n_values), dtype=real_t)
    elementwise_copy_pyst_kernel(
        field=field,
        rhs_field=rhs_field,
    )
    np.testing.assert_allclose(field, 3)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_elementwise_complex_product_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_complex_product_pyst_kernel = (
        gen_elementwise_complex_product_pyst_kernel_2d(
            real_t=real_t, num_threads=psutil.cpu_count(logical=False)
        )
    )
    complex_dtype = np.complex64 if real_t == np.float32 else np.complex128
    field_1 = (1 + 2j) * np.ones((n_values, n_values), dtype=complex_dtype)
    field_2 = (2 + 3j) * np.ones((n_values, n_values), dtype=complex_dtype)
    product_field = np.zeros_like(field_1)
    elementwise_complex_product_pyst_kernel(
        product_field=product_field,
        field_1=field_1,
        field_2=field_2,
    )
    np.testing.assert_allclose(product_field, (-4 + 7j))


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_set_fixed_val_at_boundaries_2d(n_values, precision):
    real_t = get_real_t(precision)
    width = 2
    set_fixed_val_at_boundaries = gen_set_fixed_val_at_boundaries_pyst_kernel_2d(
        real_t=real_t,
        width=width,
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    field = np.ones((n_values, n_values), dtype=real_t)
    fixed_val = real_t(3)
    set_fixed_val_at_boundaries(
        field=field,
        fixed_val=fixed_val,
    )
    np.testing.assert_allclose(field[:width, :], 3)
    np.testing.assert_allclose(field[-width:, :], 3)
    np.testing.assert_allclose(field[:, :width], 3)
    np.testing.assert_allclose(field[:, -width:], 3)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_set_fixed_val_at_boundaries_2d(n_values, precision):
    real_t = get_real_t(precision)
    width = 2
    dim = 2
    set_fixed_val_at_boundaries = gen_set_fixed_val_at_boundaries_pyst_kernel_2d(
        real_t=real_t,
        width=width,
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    field = np.ones((dim, n_values, n_values), dtype=real_t)
    fixed_vals = [2, 3]
    set_fixed_val_at_boundaries(
        vector_field=field,
        fixed_vals=fixed_vals,
    )
    for axis in range(dim):
        np.testing.assert_allclose(field[axis, :width, :], fixed_vals[axis])
        np.testing.assert_allclose(field[axis, -width:, :], fixed_vals[axis])
        np.testing.assert_allclose(field[axis, :, :width], fixed_vals[axis])
        np.testing.assert_allclose(field[axis, :, -width:], fixed_vals[axis])


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_add_fixed_val_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    add_fixed_val_pyst_kernel = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    field = np.ones((n_values, n_values), dtype=real_t)
    fixed_val = real_t(3)
    add_fixed_val_pyst_kernel(
        sum_field=field,
        field=field,
        fixed_val=fixed_val,
    )
    # 1 + 3 = 4
    np.testing.assert_allclose(field, 4)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_add_fixed_val_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    add_fixed_val_pyst_kernel = gen_add_fixed_val_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    field = np.ones((2, n_values, n_values), dtype=real_t)
    fixed_vals = np.array((2, 3), dtype=real_t)
    add_fixed_val_pyst_kernel(
        sum_field=field,
        vector_field=field,
        fixed_vals=fixed_vals,
    )
    # 1 + 2 = 3
    np.testing.assert_allclose(field[0], 3)
    # 1 + 3 = 4
    np.testing.assert_allclose(field[1], 4)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_elementwise_saxpby_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_saxpby_pyst_kernel = gen_elementwise_saxpby_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="scalar",
    )
    field_1 = 2 * np.ones((n_values, n_values), dtype=real_t)
    field_2 = 3 * np.ones((n_values, n_values), dtype=real_t)
    prefac_1 = real_t(2)
    prefac_2 = real_t(3)
    sum_field = np.zeros_like(field_1)
    elementwise_saxpby_pyst_kernel(
        sum_field=sum_field,
        field_1=field_1,
        field_2=field_2,
        field_1_prefac=prefac_1,
        field_2_prefac=prefac_2,
    )
    # 2 * 2 + 3 * 3 = 13
    np.testing.assert_allclose(sum_field, 13)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_vector_field_elementwise_saxpby_pyst_kernel_2d(n_values, precision):
    real_t = get_real_t(precision)
    elementwise_saxpby_pyst_kernel = gen_elementwise_saxpby_pyst_kernel_2d(
        real_t=real_t,
        fixed_grid_size=(n_values, n_values),
        num_threads=psutil.cpu_count(logical=False),
        field_type="vector",
    )
    field_1 = 2 * np.ones((2, n_values, n_values), dtype=real_t)
    field_2 = 3 * np.ones((2, n_values, n_values), dtype=real_t)
    prefac_1 = real_t(2)
    prefac_2 = real_t(3)
    sum_field = np.zeros_like(field_1)
    elementwise_saxpby_pyst_kernel(
        sum_field=sum_field,
        field_1=field_1,
        field_2=field_2,
        field_1_prefac=prefac_1,
        field_2_prefac=prefac_2,
    )
    # 2 * 2 + 3 * 3 = 13
    np.testing.assert_allclose(sum_field, 13)
