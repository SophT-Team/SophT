import os
import numpy as np
from numpy.testing import assert_allclose
import pytest
from sopht.utils import IO
from sopht.utils import get_real_t, get_test_tol


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_eulerian_2d_scalar_io(n_values, precision):
    real_t = get_real_t(precision)
    dim = 2
    # Initialize 2D eulerian domain
    grid_size_x = n_values
    grid_size_y_by_x = 2
    grid_size_y = grid_size_x * grid_size_y_by_x
    dx = 1.0 / grid_size_x
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x)
    y = np.linspace(eul_grid_shift, 1 * grid_size_y_by_x - eul_grid_shift, grid_size_y)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    # Initialize scalar field
    xpos = 0.5
    ypos = 1.0
    scale_x = 1
    scale_y = 2
    scalar_2d_eulerian_field = np.sqrt(
        ((x_grid - xpos) / scale_x) ** 2 + ((y_grid - ypos) / scale_y) ** 2
    )
    time = np.random.rand()

    # Initialize IO
    origin_io = np.array([y_grid.min(), x_grid.min()])
    dx_io = dx * np.ones(dim)
    grid_size_io = np.array([grid_size_y, grid_size_x])

    # Save field
    io = IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    # Add scalar field on eulerian grid
    io.add_as_eulerian_fields_for_io(scalar_2d=scalar_2d_eulerian_field)
    io.save(
        h5_file_name="test_2d_scalar_eulerian_field.h5",
        time=time,
    )

    # Load saved HDF5 file for checking
    del io
    scalar_2d_eulerian_field_saved = scalar_2d_eulerian_field.copy()
    scalar_2d_eulerian_field_loaded = np.zeros_like(scalar_2d_eulerian_field)
    io = IO(dim=2, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(scalar_2d=scalar_2d_eulerian_field_loaded)
    time_loaded = io.load(h5_file_name="test_2d_scalar_eulerian_field.h5")
    # Check values
    assert_allclose(
        scalar_2d_eulerian_field_saved,
        scalar_2d_eulerian_field_loaded,
        atol=get_test_tol(precision),
    )
    assert_allclose(time, time_loaded, atol=get_test_tol(precision))
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_eulerian_2d_vector_io(n_values, precision):
    real_t = get_real_t(precision)
    dim = 2
    # Initialize 2D domain
    grid_size_x = n_values
    grid_size_y_by_x = 2
    grid_size_y = grid_size_x * grid_size_y_by_x
    dx = 1.0 / grid_size_x
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x)
    y = np.linspace(eul_grid_shift, 1 * grid_size_y_by_x - eul_grid_shift, grid_size_y)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    # Initialize vector field (i.e. source)
    xpos = 0.5
    ypos = 1.0
    scale_x = 1
    scale_y = 2
    vector_2d_eulerian_field = np.zeros((dim, *x_grid.shape))
    vector_2d_eulerian_field[0, ...] = (x_grid - xpos) / scale_x
    vector_2d_eulerian_field[1, ...] = (y_grid - ypos) / scale_y
    time = np.random.rand()

    # Initialize IO
    origin_io = np.array([y_grid.min(), x_grid.min()])
    dx_io = dx * np.ones(dim)
    grid_size_io = np.array([grid_size_y, grid_size_x])

    # Save field
    io = IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    # Add vector field on eulerian grid
    io.add_as_eulerian_fields_for_io(vector_2d=vector_2d_eulerian_field)
    io.save(h5_file_name="test_2d_vector_eulerian_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    vector_2d_eulerian_field_saved = vector_2d_eulerian_field.copy()
    vector_2d_eulerian_field_loaded = np.zeros_like(vector_2d_eulerian_field)
    io = IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(vector_2d=vector_2d_eulerian_field_loaded)
    time_loaded = io.load(h5_file_name="test_2d_vector_eulerian_field.h5")
    # Check values
    assert_allclose(
        vector_2d_eulerian_field_saved,
        vector_2d_eulerian_field_loaded,
        atol=get_test_tol(precision),
    )
    assert_allclose(time, time_loaded, atol=get_test_tol(precision))
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [16])
def test_eulerian_2d_scalar_vector_io(n_values, precision):
    real_t = get_real_t(precision)
    dim = 2
    # Initialize 2D domain
    grid_size_x = n_values
    grid_size_y_by_x = 2
    grid_size_y = grid_size_x * grid_size_y_by_x
    dx = 1.0 / grid_size_x
    eul_grid_shift = dx / 2
    x = np.linspace(eul_grid_shift, 1 - eul_grid_shift, grid_size_x)
    y = np.linspace(eul_grid_shift, 1 * grid_size_y_by_x - eul_grid_shift, grid_size_y)
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")

    # Initialize scalar field
    xpos = 0.5
    ypos = 1.0
    scale_x = 1
    scale_y = 2
    scalar_2d_eulerian_field = np.sqrt(
        ((x_grid - xpos) / scale_x) ** 2 + ((y_grid - ypos) / scale_y) ** 2
    )
    vector_2d_eulerian_field = np.zeros((dim, *x_grid.shape))
    vector_2d_eulerian_field[0, ...] = (x_grid - xpos) / scale_x
    vector_2d_eulerian_field[1, ...] = (y_grid - ypos) / scale_y
    time = np.random.rand()

    # Initialize IO
    origin_io = np.array([y_grid.min(), x_grid.min()])
    dx_io = dx * np.ones(dim)
    grid_size_io = np.array([grid_size_y, grid_size_x])

    # Save field
    io = IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(
        scalar_2d=scalar_2d_eulerian_field, vector_2d=vector_2d_eulerian_field
    )
    io.save(
        h5_file_name="test_2d_scalar_vector_eulerian_fields.h5",
        time=time,
    )

    # Load saved HDF5 file for checking
    del io
    scalar_2d_eulerian_field_saved = scalar_2d_eulerian_field.copy()
    scalar_2d_eulerian_field_loaded = np.zeros_like(scalar_2d_eulerian_field)
    vector_2d_eulerian_field_saved = vector_2d_eulerian_field.copy()
    vector_2d_eulerian_field_loaded = np.zeros_like(vector_2d_eulerian_field)
    io = IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    # Add scalar and vector fields on eulerian grid
    io.add_as_eulerian_fields_for_io(
        scalar_2d=scalar_2d_eulerian_field_loaded,
        vector_2d=vector_2d_eulerian_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_2d_scalar_vector_eulerian_fields.h5")
    # Check values
    assert_allclose(
        scalar_2d_eulerian_field_saved,
        scalar_2d_eulerian_field_loaded,
        atol=get_test_tol(precision),
    )
    assert_allclose(
        vector_2d_eulerian_field_saved,
        vector_2d_eulerian_field_loaded,
        atol=get_test_tol(precision),
    )
    assert_allclose(time, time_loaded, atol=get_test_tol(precision))
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [64])
def test_lagrangian_2d_scalar_io(n_values, precision):
    real_t = get_real_t(precision)
    testing_atol = get_test_tol(precision)
    dim = 2
    num_lagrangian_nodes = n_values

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiral
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt

    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[0, :] = radius * np.cos(theta)
    lagrangian_grid_position[1, :] = radius * np.sin(theta)

    # Some scalar values that increases linearly with node position
    scalar_2d_lagrangian_field = np.linspace(0, 1, num_lagrangian_nodes)

    time = np.random.rand()

    # Initialize IO
    io = IO(dim=dim, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="helix",
        scalar_2d=scalar_2d_lagrangian_field,
    )

    # Save field
    io.save(h5_file_name="test_2d_scalar_lagrangian_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_2d_lagrangian_field_saved = scalar_2d_lagrangian_field.copy()
    scalar_2d_lagrangian_field_loaded = np.zeros_like(scalar_2d_lagrangian_field)
    io = IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="helix",
        scalar_2d=scalar_2d_lagrangian_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_2d_scalar_lagrangian_field.h5")
    # Check values
    assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    assert_allclose(
        scalar_2d_lagrangian_field_saved,
        scalar_2d_lagrangian_field_loaded,
        atol=testing_atol,
    )
    assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [64])
def test_lagrangian_2d_vector_io(n_values, precision):
    real_t = get_real_t(precision)
    testing_atol = get_test_tol(precision)
    dim = 2
    num_lagrangian_nodes = n_values

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiral
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt

    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[0, :] = radius * np.cos(theta)
    lagrangian_grid_position[1, :] = radius * np.sin(theta)

    # Here we consider vector fields as the tangent direction along the spiral
    vector_2d_lagrangian_field = np.zeros_like(lagrangian_grid_position)
    vector_2d_lagrangian_field[0, :] = -radius * np.sin(theta)
    vector_2d_lagrangian_field[1, :] = radius * np.cos(theta)

    time = np.random.rand()

    # Initialize IO
    io = IO(dim=dim, real_dtype=real_t)
    # Add vector field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="helix",
        vector_2d=vector_2d_lagrangian_field,
    )

    # Save field
    io.save(h5_file_name="test_2d_vector_lagrangian_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    vector_2d_lagrangian_field_saved = vector_2d_lagrangian_field.copy()
    vector_2d_lagrangian_field_loaded = np.zeros_like(vector_2d_lagrangian_field)
    io = IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="helix",
        vector_2d=vector_2d_lagrangian_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_2d_vector_lagrangian_field.h5")
    # Check values
    assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    assert_allclose(
        vector_2d_lagrangian_field_saved,
        vector_2d_lagrangian_field_loaded,
        atol=testing_atol,
    )
    assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("n_values", [64])
def test_lagrangian_2d_scalar_vector_io(n_values, precision):
    real_t = get_real_t(precision)
    testing_atol = get_test_tol(precision)
    dim = 2
    num_lagrangian_nodes = n_values

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiral
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt

    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[0, :] = radius * np.cos(theta)
    lagrangian_grid_position[1, :] = radius * np.sin(theta)

    # Some scalar values that increases linearly with node position
    scalar_2d_lagrangian_field = np.linspace(0, 1, num_lagrangian_nodes)
    # Here we consider vector fields as the tangent direction along the spiral
    vector_2d_lagrangian_field = np.zeros_like(lagrangian_grid_position)
    vector_2d_lagrangian_field[0, :] = -radius * np.sin(theta)
    vector_2d_lagrangian_field[1, :] = radius * np.cos(theta)

    time = np.random.rand()

    # Initialize IO
    io = IO(dim=dim, real_dtype=real_t)
    # Add scalar and vector fields on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="helix",
        scalar_2d=scalar_2d_lagrangian_field,
        vector_2d=vector_2d_lagrangian_field,
    )

    # Save field
    io.save(h5_file_name="test_2d_scalar_vector_lagrangian_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_2d_lagrangian_field_saved = scalar_2d_lagrangian_field.copy()
    scalar_2d_lagrangian_field_loaded = np.zeros_like(scalar_2d_lagrangian_field)
    vector_2d_lagrangian_field_saved = vector_2d_lagrangian_field.copy()
    vector_2d_lagrangian_field_loaded = np.zeros_like(vector_2d_lagrangian_field)
    io = IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="helix",
        scalar_2d=scalar_2d_lagrangian_field_loaded,
        vector_2d=vector_2d_lagrangian_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_2d_scalar_vector_lagrangian_field.h5")
    # Check values
    assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    assert_allclose(
        scalar_2d_lagrangian_field_saved,
        scalar_2d_lagrangian_field_loaded,
        atol=testing_atol,
    )
    assert_allclose(
        vector_2d_lagrangian_field_saved,
        vector_2d_lagrangian_field_loaded,
        atol=testing_atol,
    )
    assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")
