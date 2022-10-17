import numpy as np
import sopht_simulator as sps
from sopht.utils.precision import get_real_t
from tests.immersed_body.rigid_body.test_rigid_body_forcing_grids import (
    mock_2d_cylinder,
)


def test_rigid_body_flow_interaction():
    cylinder = mock_2d_cylinder()
    grid_size = (16, 16)
    forcing_grid_cls = sps.CircularCylinderForcingGrid
    cylinder_flow_interactor = sps.RigidBodyFlowInteraction(
        rigid_body=cylinder,
        eul_grid_forcing_field=np.zeros(grid_size),
        eul_grid_velocity_field=np.zeros(grid_size),
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=1.0,
        grid_dim=2,
        real_t=get_real_t(),
        forcing_grid_cls=forcing_grid_cls,
        num_forcing_points=16,
    )
    rigid_body_dim = 3
    np.testing.assert_allclose(
        cylinder_flow_interactor.body_flow_forces, np.zeros((rigid_body_dim, 1))
    )
    np.testing.assert_allclose(
        cylinder_flow_interactor.body_flow_torques, np.zeros((rigid_body_dim, 1))
    )
    assert isinstance(cylinder_flow_interactor.forcing_grid, forcing_grid_cls)
