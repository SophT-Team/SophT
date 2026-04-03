"""Common fixtures for testing."""

import os

import pytest
from numpy.random import Generator, default_rng


@pytest.fixture
def max_cpu_count() -> int:
    """Returns the maximum number of CPUs available to the process."""
    if hasattr(os, "process_cpu_count"):
        cpu_count = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        cpu_count = len(os.sched_getaffinity(0))
    else:
        cpu_count = os.cpu_count()
    return cpu_count or 1


@pytest.fixture
def rng() -> Generator:
    """Returns a random number generator with a fixed seed for reproducibility."""
    return default_rng(seed=42)
