"""
Tests for utility functions.
"""

import numpy as np
from pyod.models.lof import LOF

from hog_bisect.utils import (
    fit_in_all_subspaces,
    fit_model,
    gen_powerset,
    gen_rand_subspaces,
    random_unif_on_sphere,
    subspace_grab,
)


class TestRandomUnifOnSphere:
    """Tests for random_unif_on_sphere function."""

    def test_dimensions(self):
        """Generated points have correct dimensions."""
        points = random_unif_on_sphere(number=100, dimensions=3, r=1)
        assert points.shape == (100, 3)

    def test_on_sphere(self):
        """Generated points lie on the sphere with given radius."""
        points = random_unif_on_sphere(number=1000, dimensions=3, r=1)
        for point in points:
            distance = np.linalg.norm(point)
            assert abs(distance - 1.0) < 1e-5

    def test_different_radius(self):
        """Points lie on sphere with custom radius."""
        radius = 5.0
        points = random_unif_on_sphere(number=100, dimensions=3, r=radius)
        for point in points:
            distance = np.linalg.norm(point)
            assert abs(distance - radius) < 1e-5

    def test_high_dimensions(self):
        """Works with high-dimensional space."""
        points = random_unif_on_sphere(number=50, dimensions=10, r=1)
        assert points.shape == (50, 10)
        for point in points:
            distance = np.linalg.norm(point)
            assert abs(distance - 1.0) < 1e-5

    def test_reproducible(self):
        """Same random_state produces same results."""
        points1 = random_unif_on_sphere(number=10, dimensions=3, r=1, random_state=42)
        points2 = random_unif_on_sphere(number=10, dimensions=3, r=1, random_state=42)
        np.testing.assert_array_equal(points1, points2)

    def test_different_seeds_differ(self):
        """Different random_state produces different results."""
        points1 = random_unif_on_sphere(number=10, dimensions=3, r=1, random_state=42)
        points2 = random_unif_on_sphere(number=10, dimensions=3, r=1, random_state=99)
        assert not np.array_equal(points1, points2)


class TestGenPowerset:
    """Tests for gen_powerset function."""

    def test_three_dimensions(self):
        """Generates correct powerset for 3 dimensions."""
        result = gen_powerset(3)
        expected = {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)}
        assert result == expected

    def test_excludes_empty_and_full(self):
        """Result excludes empty set and full set."""
        result = gen_powerset(3)
        assert () not in result
        assert (0, 1, 2) not in result

    def test_count(self):
        """Generates 2^n - 2 subsets."""
        for n in range(2, 6):
            result = gen_powerset(n)
            expected_count = 2**n - 2
            assert len(result) == expected_count

    def test_two_dimensions(self):
        """Works for 2 dimensions."""
        result = gen_powerset(2)
        expected = {(0,), (1,)}
        assert result == expected


class TestSubspaceGrab:
    """Tests for subspace_grab function."""

    def test_single_column(self):
        """Extracts single column correctly."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = subspace_grab((1,), data)
        expected = np.array([[2], [5], [8]])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_columns(self):
        """Extracts multiple columns correctly."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = subspace_grab((0, 2), data)
        expected = np.array([[1, 3], [4, 6], [7, 9]])
        np.testing.assert_array_equal(result, expected)

    def test_all_columns(self):
        """Extracting all columns returns equivalent data."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        result = subspace_grab((0, 1, 2), data)
        np.testing.assert_array_equal(result, data)

    def test_preserves_rows(self):
        """Row count is preserved."""
        data = np.random.randn(100, 5)
        result = subspace_grab((1, 3), data)
        assert result.shape[0] == 100
        assert result.shape[1] == 2


class TestGenRandSubspaces:
    """Tests for gen_rand_subspaces function."""

    def test_returns_set(self):
        """Returns a set of tuples."""
        result = gen_rand_subspaces(dims=10, upper_limit=6, seed=42)
        assert isinstance(result, set)
        for item in result:
            assert isinstance(item, tuple)

    def test_count_limit(self):
        """Generates approximately 2^upper_limit - 2 subspaces."""
        upper_limit = 6
        result = gen_rand_subspaces(dims=10, upper_limit=upper_limit, seed=42)
        expected_count = 2**upper_limit - 2
        assert len(result) == expected_count

    def test_includes_singletons(self):
        """When include_all_attr=True, includes all singleton subspaces."""
        dims = 5
        result = gen_rand_subspaces(dims=dims, upper_limit=6, include_all_attr=True, seed=42)

        for i in range(dims):
            assert (i,) in result

    def test_reproducible(self):
        """Same seed produces same subspaces."""
        result1 = gen_rand_subspaces(dims=10, upper_limit=5, seed=42)
        result2 = gen_rand_subspaces(dims=10, upper_limit=5, seed=42)
        assert result1 == result2

    def test_no_full_space(self):
        """Does not include the full space."""
        dims = 5
        result = gen_rand_subspaces(dims=dims, upper_limit=6, seed=42)
        full_space = tuple(range(dims))
        assert full_space not in result


class TestFitModel:
    """Tests for fit_model function."""

    def test_returns_tuple(self, small_data, temp_dir):
        """fit_model returns (subspace, model) tuple."""
        subspace = (0, 1)
        result = fit_model(subspace, small_data, LOF, temp_dir)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == subspace

    def test_model_is_fitted(self, small_data, temp_dir):
        """Returned model is fitted and can predict."""
        subspace = (0, 1)
        _, model = fit_model(subspace, small_data, LOF, temp_dir)
        # Should be able to predict without error
        point = small_data[0, :2]  # First row, first 2 columns
        result = model.predict(point)
        assert isinstance(result, (bool, np.bool_))


class TestFitInAllSubspaces:
    """Tests for fit_in_all_subspaces function."""

    def test_returns_dict(self, small_data, temp_dir):
        """Returns dictionary of fitted models."""
        result = fit_in_all_subspaces(LOF, small_data, temp_dir, subspace_limit=10, seed=42)
        assert isinstance(result, dict)

    def test_includes_full_space(self, small_data, temp_dir):
        """Result includes model for full space."""
        result = fit_in_all_subspaces(LOF, small_data, temp_dir, subspace_limit=10, seed=42)
        full_space = tuple(range(small_data.shape[1]))
        assert full_space in result

    def test_subspace_limit_triggers_sampling(self, temp_dir):
        """High-dimensional data uses random subspace sampling."""
        # 12 dimensions > default subspace_limit
        high_dim_data = np.random.randn(50, 12)
        result = fit_in_all_subspaces(LOF, high_dim_data, temp_dir, subspace_limit=8, seed=42)
        # Should have ~2^8 - 2 + 1 = 255 subspaces (plus full space)
        # But not 2^12 - 2 = 4094
        assert len(result) < 500
        assert len(result) > 50

    def test_all_models_can_predict(self, small_data, temp_dir):
        """All models in result can make predictions."""
        result = fit_in_all_subspaces(LOF, small_data, temp_dir, subspace_limit=10, seed=42)

        for subspace, model in result.items():
            # Extract subspace columns from first data point
            point = small_data[0, list(subspace)]
            prediction = model.predict(point)
            assert isinstance(prediction, (bool, np.bool_))
