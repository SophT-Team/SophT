import pytest
import sopht.utils as spu
import elastica as ea
import numpy as np
import os


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_eulerian_grid_scalar_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)

    # Initialize eulerian params
    n_values = 16
    domain_aspect_ratio = np.arange(dim) + 1
    grid_size = n_values * domain_aspect_ratio
    x_range = 1.0
    dx = x_range / grid_size[spu.VectorField.x_axis_idx()]
    eul_grid_shift = dx / 2

    # Initialize scalar field
    scalar_field = np.random.randn(*np.flip(grid_size)).astype(real_t)
    time = np.random.rand()

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(dim)
    dx_io = dx * np.ones(dim)
    grid_size_io = np.flip(grid_size)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)

    # Save field
    io.add_as_eulerian_fields_for_io(scalar_field=scalar_field)
    io.save(
        h5_file_name="test_eulerian_grid_scalar_field.h5",
        time=time,
    )

    # Load saved HDF5 file for checking
    del io
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(scalar_field=scalar_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_scalar_field.h5")
    # Check values
    np.testing.assert_allclose(
        scalar_field_saved,
        scalar_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_eulerian_grid_vector_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)

    # Initialize eulerian params
    n_values = 16
    domain_aspect_ratio = np.arange(dim) + 1
    grid_size = n_values * domain_aspect_ratio
    x_range = 1.0
    dx = x_range / grid_size[spu.VectorField.x_axis_idx()]
    eul_grid_shift = dx / 2

    # Initialize vector field
    vector_field = np.random.randn(dim, *np.flip(grid_size)).astype(real_t)
    time = np.random.rand()

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(dim)
    dx_io = dx * np.ones(dim)
    grid_size_io = np.flip(grid_size)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)

    # Save field
    io.add_as_eulerian_fields_for_io(vector_field=vector_field)
    io.save(h5_file_name="test_eulerian_grid_vector_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(vector_field=vector_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_vector_field.h5")

    # Check values
    np.testing.assert_allclose(
        vector_field_saved,
        vector_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_eulerian_grid_multiple_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)

    # Initialize eulerian domain
    n_values = 16
    domain_aspect_ratio = np.arange(dim) + 1
    grid_size = n_values * domain_aspect_ratio
    x_range = 1.0
    dx = x_range / grid_size[spu.VectorField.x_axis_idx()]
    eul_grid_shift = dx / 2

    # Initialize vector field (i.e. source)
    scalar_field = np.random.randn(*np.flip(grid_size)).astype(real_t)
    vector_field = np.random.randn(dim, *np.flip(grid_size)).astype(real_t)
    time = np.random.rand()

    # Initialize IO
    origin_io = eul_grid_shift * np.ones(dim)
    dx_io = dx * np.ones(dim)
    grid_size_io = np.flip(grid_size)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)

    # Save field
    io.add_as_eulerian_fields_for_io(scalar_field=scalar_field)
    io.add_as_eulerian_fields_for_io(vector_field=vector_field)
    io.save(h5_file_name="test_eulerian_grid_multiple_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.define_eulerian_grid(origin=origin_io, dx=dx_io, grid_size=grid_size_io)
    io.add_as_eulerian_fields_for_io(scalar_field=scalar_field_loaded)
    io.add_as_eulerian_fields_for_io(vector_field=vector_field_loaded)
    time_loaded = io.load(h5_file_name="test_eulerian_grid_multiple_field.h5")

    # Check values
    np.testing.assert_allclose(
        scalar_field_saved,
        scalar_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(
        vector_field_saved,
        vector_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_lagrangian_grid_scalar_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)
    num_lagrangian_nodes = 64

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    if dim == 3:
        dzdt = 1.0
        z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
        lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize scalar field
    scalar_field = np.linspace(0, 1, num_lagrangian_nodes)
    time = np.random.rand()

    # Initialize IO
    io = spu.IO(dim=dim, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="test_grid",
        scalar_field=scalar_field,
    )

    # Save field
    io.save(h5_file_name="test_lagrangian_grid_scalar_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="test_grid",
        scalar_field=scalar_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_scalar_field.h5")

    # Check values
    np.testing.assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(
        scalar_field_saved,
        scalar_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_lagrangian_grid_vector_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)
    num_lagrangian_nodes = 64

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    if dim == 3:
        dzdt = 1.0
        z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
        lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize vector field
    # Here we consider vector fields as the tangent direction along the spiral
    vector_field = np.zeros_like(lagrangian_grid_position)
    vector_field[spu.VectorField.x_axis_idx(), :] = -radius * np.sin(theta)
    vector_field[spu.VectorField.y_axis_idx(), :] = radius * np.cos(theta)
    if dim == 3:
        vector_field[spu.VectorField.z_axis_idx(), :] = dzdt
    time = np.random.rand()

    # Initialize IO
    io = spu.IO(dim=dim, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="test_grid",
        vector_field=vector_field,
    )

    # Save field
    io.save(h5_file_name="test_lagrangian_grid_vector_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="test_grid",
        vector_field=vector_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_vector_field.h5")

    # Check values
    np.testing.assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(
        vector_field_saved,
        vector_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_lagrangian_grid_multiple_field_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)
    num_lagrangian_nodes = 64

    # Initialize lagrangian grid
    # Here we consider a grid with positions along a outward spiraling coil/helix
    drdt = 1.0
    num_revolutions = 4
    theta = np.linspace(0, num_revolutions * 2 * np.pi, num_lagrangian_nodes)
    radius = np.linspace(0, num_revolutions, num_lagrangian_nodes) * drdt
    lagrangian_grid_position = np.zeros((dim, num_lagrangian_nodes))
    lagrangian_grid_position[spu.VectorField.x_axis_idx(), :] = radius * np.cos(theta)
    lagrangian_grid_position[spu.VectorField.y_axis_idx(), :] = radius * np.sin(theta)
    if dim == 3:
        dzdt = 1.0
        z = np.linspace(0, num_revolutions, num_lagrangian_nodes) * dzdt
        lagrangian_grid_position[spu.VectorField.z_axis_idx(), :] = z

    # Initialize scalar and vector field
    scalar_field = np.random.randn(num_lagrangian_nodes)
    # Here we consider vector fields as the tangent direction along the spiral
    vector_field = np.zeros_like(lagrangian_grid_position)
    vector_field[spu.VectorField.x_axis_idx(), :] = -radius * np.sin(theta)
    vector_field[spu.VectorField.y_axis_idx(), :] = radius * np.cos(theta)
    if dim == 3:
        vector_field[spu.VectorField.z_axis_idx(), :] = dzdt
    time = np.random.rand()

    # Initialize IO
    io = spu.IO(dim=dim, real_dtype=real_t)
    # Add scalar field on lagrangian grid
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position,
        lagrangian_grid_name="test_grid",
        scalar_field=scalar_field,
        vector_field=vector_field,
    )

    # Save field
    io.save(h5_file_name="test_lagrangian_grid_multiple_field.h5", time=time)

    # Load saved HDF5 file for checking
    del io
    lagrangian_grid_position_saved = lagrangian_grid_position.copy()
    lagrangian_grid_position_loaded = np.zeros_like(lagrangian_grid_position)
    scalar_field_saved = scalar_field.copy()
    scalar_field_loaded = np.zeros_like(scalar_field)
    vector_field_saved = vector_field.copy()
    vector_field_loaded = np.zeros_like(vector_field)
    io = spu.IO(dim=dim, real_dtype=real_t)
    io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=lagrangian_grid_position_loaded,
        lagrangian_grid_name="test_grid",
        scalar_field=scalar_field_loaded,
        vector_field=vector_field_loaded,
    )
    time_loaded = io.load(h5_file_name="test_lagrangian_grid_multiple_field.h5")

    # Check values
    np.testing.assert_allclose(
        lagrangian_grid_position_saved,
        lagrangian_grid_position_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(
        scalar_field_saved,
        scalar_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(
        vector_field_saved,
        vector_field_loaded,
        atol=testing_atol,
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dim", [2, 3])
def test_cosserat_rod_io(dim, precision):
    real_t = spu.get_real_t(precision)
    testing_atol = spu.get_test_tol(precision)

    # Initialize mock rod
    n_element = 16
    rod_incline_angle = np.pi / 4.0
    start = np.zeros(3)
    direction = np.zeros_like(start)
    direction[spu.VectorField.x_axis_idx()] = np.cos(rod_incline_angle)
    direction[spu.VectorField.y_axis_idx()] = np.sin(rod_incline_angle)
    normal = np.array([0.0, 0.0, 1.0])
    rod_length = 1.0
    rod_element_radius = np.linspace(0.01, 0.5, n_element)
    density = 1.0
    nu = 1.0
    youngs_modulus = 1.0
    rod = ea.CosseratRod.straight_rod(
        n_element,
        start,
        direction,
        normal,
        rod_length,
        rod_element_radius,
        density,
        nu,
        youngs_modulus,
    )
    time = np.random.rand()

    # Initialize cosserat rod io
    rod_io = spu.CosseratRodIO(cosserat_rod=rod, dim=dim, real_dtype=real_t)
    # Save rod
    rod_io.save(h5_file_name="test_cosserat_rod_io.h5", time=time)

    # Load saved HDF5 file for checking
    del rod_io
    rod_element_position_saved = 0.5 * (
        rod.position_collection[:dim, 1:] + rod.position_collection[:dim, :-1]
    )
    rod_element_position_loaded = np.zeros((dim, n_element))
    rod_element_radius_saved = rod_element_radius.copy()
    rod_element_radius_loaded = np.zeros(n_element)
    base_io = spu.IO(dim=dim, real_dtype=real_t)
    base_io.add_as_lagrangian_fields_for_io(
        lagrangian_grid=rod_element_position_loaded,
        lagrangian_grid_name="rod",
        scalar_3d=rod_element_radius_loaded,
    )
    time_loaded = base_io.load(h5_file_name="test_cosserat_rod_io.h5")

    # Check values
    np.testing.assert_allclose(
        rod_element_position_saved, rod_element_position_loaded, atol=testing_atol
    )
    np.testing.assert_allclose(
        rod_element_radius_saved, rod_element_radius_loaded, atol=testing_atol
    )
    np.testing.assert_allclose(time, time_loaded, atol=testing_atol)
    os.system("rm -f *h5 *xmf")


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("grid_size_x", [8, 16])
def test_eulerian_field_io(grid_dim, precision, grid_size_x):
    real_t = spu.get_real_t(precision)
    grid_size = (grid_size_x,) * grid_dim
    scalar_field = np.random.rand(*grid_size).astype(real_t)
    vector_field = np.random.rand(grid_dim, *grid_size).astype(real_t)

    x_range = 2.0
    dx = real_t(x_range / grid_size_x)
    eul_grid_shift = dx / 2.0
    x = np.linspace(eul_grid_shift, x_range - eul_grid_shift, grid_size_x).astype(
        real_t
    )
    position_field = np.flipud(np.array(np.meshgrid(*((x,) * grid_dim), indexing="ij")))
    h5_file_name = "eulerian_field.h5"
    time = 2.0
    eulerian_field_dict = {"scalar_field": scalar_field, "vector_field": vector_field}
    test_io = spu.EulerianFieldIO(
        position_field=position_field, eulerian_fields_dict=eulerian_field_dict
    )
    test_io.save(h5_file_name=h5_file_name, time=time)
    del test_io

    # Load saved HDF5 file for checking
    scalar_field_loaded = np.zeros_like(scalar_field)
    vector_field_loaded = np.zeros_like(vector_field)
    io = spu.IO(dim=grid_dim, real_dtype=real_t)
    match grid_dim:
        case 2:
            io_origin = np.array(
                [
                    position_field[spu.VectorField.y_axis_idx()].min(),
                    position_field[spu.VectorField.x_axis_idx()].min(),
                ]
            )
        case 3:
            io_origin = np.array(
                [
                    position_field[spu.VectorField.z_axis_idx()].min(),
                    position_field[spu.VectorField.y_axis_idx()].min(),
                    position_field[spu.VectorField.x_axis_idx()].min(),
                ]
            )
        case _:
            raise ValueError("Position field of invalid shape.")
    io_dx = dx * np.ones(grid_dim)
    io_grid_size = np.array(grid_size)
    io.define_eulerian_grid(origin=io_origin, dx=io_dx, grid_size=io_grid_size)
    io.add_as_eulerian_fields_for_io(scalar_field=scalar_field_loaded)
    io.add_as_eulerian_fields_for_io(vector_field=vector_field_loaded)
    time_loaded = io.load(h5_file_name=h5_file_name)

    # Check values
    np.testing.assert_allclose(
        scalar_field,
        scalar_field_loaded,
        atol=spu.get_test_tol(precision),
    )
    np.testing.assert_allclose(
        vector_field,
        vector_field_loaded,
        atol=spu.get_test_tol(precision),
    )
    np.testing.assert_allclose(time, time_loaded, atol=spu.get_test_tol(precision))
    os.system("rm -f *h5 *xmf")
