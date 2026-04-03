"""Common fixtures for testing."""

import pytest
from numpy.random import default_rng


@pytest.fixture
def rng():
    return default_rng(seed=42)
