import logging
import numpy as np
import sopht_simulator as sps


def mock_xy_plane(origin=np.array([1.0, 1.0, 1.0]), length=1.0, breadth=0.5):
    """Returns a mock XY plane for testing"""
    normal = np.array([0.0, 0.0, 1.0])
    tangent = np.array([1.0, 0.0, 0.0])
    plane = sps.RectangularPlane(
        origin=origin,
        plane_normal=normal,
        plane_tangent_along_length=tangent,
        plane_length=length,
        plane_breadth=breadth,
    )
    return plane


def test_rectangular_plane(caplog):
    plane_dim = 3
    origin = np.array([1.0, 1.0, 1.0])
    length = 1.0
    breadth = 0.5
    with caplog.at_level(logging.WARNING):
        plane = mock_xy_plane(origin, length, breadth)
    warning_message = (
        "==============================================="
        "\nInitialising rectangular plane object. Note:"
        "\nCurrently tracking dynamics in pyelastica is"
        "\nnot supported, please do not add the plane"
        "\nto the pyelastica simulator!"
        "\n==============================================="
    )
    assert warning_message in caplog.text
    assert plane.n_elems == 1
    assert plane.length == length
    assert plane.breadth == breadth
    correct_director = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ).reshape(plane_dim, plane_dim, 1)
    np.testing.assert_allclose(plane.director_collection, correct_director)
    np.testing.assert_allclose(plane.position_collection, origin.reshape(plane_dim, 1))
    np.testing.assert_allclose(plane.velocity_collection, 0.0)
    np.testing.assert_allclose(plane.omega_collection, 0.0)
