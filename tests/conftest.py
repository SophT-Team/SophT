"""Common fixtures for testing."""

import os

import pytest
from numpy.random import Generator, default_rng


@pytest.fixture
def max_cpu_count() -> int:
    """Returns the maximum number of CPU cores available."""

    max_cpu: int = os.cpu_count()
    if max_cpu is None:
        max_cpu = 1
    return max_cpu


@pytest.fixture
def rng() -> Generator:
    """Returns a random number generator with a fixed seed for reproducibility."""
    return default_rng(seed=42)
