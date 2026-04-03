import logging

import numpy as np
import pytest
import sopht.simulator as sps


@pytest.mark.parametrize("grid_dim", [2, 3])
@pytest.mark.parametrize("num_lag_nodes", [8, 16])
def test_immersed_body_forcing_grid(grid_dim, num_lag_nodes, caplog):
    with caplog.at_level(logging.WARNING):
        forcing_grid = sps.ImmersedBodyForcingGrid(grid_dim=grid_dim, num_lag_nodes=num_lag_nodes)
    assert forcing_grid.grid_dim == grid_dim
    assert forcing_grid.num_lag_nodes == num_lag_nodes
    correct_forcing_grid_field = np.zeros((grid_dim, num_lag_nodes))
    np.testing.assert_allclose(forcing_grid.position_field, correct_forcing_grid_field)
    np.testing.assert_allclose(forcing_grid.velocity_field, correct_forcing_grid_field)

    # Check the warning message for 2D case
    if grid_dim == 2:
        expected_message = (
            "\n=================================================="
            "\n2D body forcing grid generated, this assumes the body"
            "\nmoves in XY plane! Please initialize your body such that"
            "\nensuing dynamics are constrained in XY plane!"
            "\n=================================================="
        )
        expected_tuple = (
            "sopht.simulator.immersed_body.immersed_body_forcing_grid",
            logging.WARNING,
            expected_message,
        )
        assert expected_tuple in caplog.record_tuples
