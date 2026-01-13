"""
Shared test fixtures for the hog_bisect test suite.

This module provides common test data and fixtures used across multiple test files.
"""

import tempfile

import numpy as np
import pytest
from pyod.models.lof import LOF

from hog_bisect.utils import fit_in_all_subspaces


@pytest.fixture
def seed():
    """Default random seed for reproducible tests."""
    return 42


@pytest.fixture
def small_data(seed):
    """Small synthetic dataset for quick tests (50 samples, 3 features)."""
    np.random.seed(seed)
    return np.random.randn(50, 3)


@pytest.fixture
def medium_data(seed):
    """Medium synthetic dataset (100 samples, 5 features)."""
    np.random.seed(seed)
    return np.random.randn(100, 5)


@pytest.fixture
def large_data(seed):
    """Larger dataset for stress tests (500 samples, 8 features)."""
    np.random.seed(seed)
    return np.random.randn(500, 8)


@pytest.fixture
def clustered_data(seed):
    """Dataset with two clear clusters for predictable outlier behavior."""
    np.random.seed(seed)
    cluster1 = np.random.randn(50, 3) * 0.5 + np.array([0, 0, 0])
    cluster2 = np.random.randn(50, 3) * 0.5 + np.array([5, 5, 5])
    return np.vstack([cluster1, cluster2])


@pytest.fixture
def outlier_indicator(small_data):
    """Binary outlier indicator array (all zeros = all inliers)."""
    return np.zeros(len(small_data))


@pytest.fixture
def mixed_outlier_indicator(small_data):
    """Binary outlier indicator with some outliers marked."""
    indicator = np.zeros(len(small_data))
    indicator[:5] = 1  # First 5 points are outliers
    return indicator


@pytest.fixture
def temp_dir():
    """Temporary directory for model storage, cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def fitted_models_small(small_data, temp_dir, seed):
    """Pre-fitted models on small data for all subspaces."""
    return fit_in_all_subspaces(
        outlier_detection_method=LOF,
        data=small_data,
        tempdir=temp_dir,
        subspace_limit=10,
        seed=seed,
        n_jobs=1,
    )


@pytest.fixture
def full_space_3d():
    """Full space tuple for 3D data."""
    return (0, 1, 2)


@pytest.fixture
def full_space_5d():
    """Full space tuple for 5D data."""
    return (0, 1, 2, 3, 4)
