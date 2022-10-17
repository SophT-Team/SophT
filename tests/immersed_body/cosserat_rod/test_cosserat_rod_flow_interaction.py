import pytest
import numpy as np
import sopht_simulator as sps
from sopht.utils.precision import get_real_t
from tests.immersed_body.cosserat_rod.test_cosserat_rod_forcing_grids import (
    mock_straight_rod,
)


@pytest.mark.parametrize("n_elems", [8, 16])
def test_cosserat_rod_flow_interaction(n_elems):
    cosserat_rod = mock_straight_rod(n_elems)
    grid_size = (16, 16)
    forcing_grid_cls = sps.CosseratRodElementCentricForcingGrid
    rod_flow_interactor = sps.CosseratRodFlowInteraction(
        cosserat_rod=cosserat_rod,
        eul_grid_forcing_field=np.zeros(grid_size),
        eul_grid_velocity_field=np.zeros(grid_size),
        virtual_boundary_stiffness_coeff=1.0,
        virtual_boundary_damping_coeff=1.0,
        dx=1.0,
        grid_dim=2,
        real_t=get_real_t(),
        forcing_grid_cls=forcing_grid_cls,
    )
    rod_dim = 3
    np.testing.assert_allclose(
        rod_flow_interactor.body_flow_forces,
        np.zeros((rod_dim, cosserat_rod.n_elems + 1)),
    )
    np.testing.assert_allclose(
        rod_flow_interactor.body_flow_torques, np.zeros((rod_dim, cosserat_rod.n_elems))
    )
    assert isinstance(rod_flow_interactor.forcing_grid, forcing_grid_cls)
