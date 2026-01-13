"""
Tests for origin calculation strategies.

Each origin method determines the starting point for bisection search.
"""

import numpy as np
import pytest

from hog_bisect.origin_method import (
    CentroidOrigin,
    LeastOutlierOrigin,
    OriginType,
    RandomOrigin,
    WeightedOrigin,
    get_origin,
)


class TestCentroidOrigin:
    """Tests for CentroidOrigin strategy."""

    def test_calculate_origin_returns_centroid(self, small_data, outlier_indicator):
        """CentroidOrigin returns the data mean."""
        origin = CentroidOrigin(small_data, outlier_indicator)
        result = origin.calculate_origin()

        expected = small_data.mean(axis=0)
        np.testing.assert_array_almost_equal(result, expected)

    def test_calculate_origin_deterministic(self, small_data, outlier_indicator):
        """CentroidOrigin returns same result on multiple calls."""
        origin = CentroidOrigin(small_data, outlier_indicator)
        result1 = origin.calculate_origin()
        result2 = origin.calculate_origin()

        np.testing.assert_array_equal(result1, result2)

    def test_class_type(self, small_data, outlier_indicator):
        """CentroidOrigin has correct class_type."""
        origin = CentroidOrigin(small_data, outlier_indicator)
        assert origin.class_type == OriginType.CENTROID


class TestLeastOutlierOrigin:
    """Tests for LeastOutlierOrigin strategy."""

    def test_calculate_origin_returns_point_in_data(self, small_data, outlier_indicator):
        """LeastOutlierOrigin returns a point that exists in the data."""
        origin = LeastOutlierOrigin(small_data, outlier_indicator)
        result = origin.calculate_origin()

        # Result should be one of the data points
        assert result.shape == (small_data.shape[1],)
        # Check that result is actually in the data
        distances = np.linalg.norm(small_data - result, axis=1)
        assert np.min(distances) < 1e-10  # Should match exactly

    def test_calculate_origin_deterministic(self, small_data, outlier_indicator):
        """LeastOutlierOrigin returns same result on multiple calls."""
        origin = LeastOutlierOrigin(small_data, outlier_indicator)
        result1 = origin.calculate_origin()
        result2 = origin.calculate_origin()

        np.testing.assert_array_equal(result1, result2)

    def test_class_type(self, small_data, outlier_indicator):
        """LeastOutlierOrigin has correct class_type."""
        origin = LeastOutlierOrigin(small_data, outlier_indicator)
        assert origin.class_type == OriginType.LEAST_OUTLIER


class TestRandomOrigin:
    """Tests for RandomOrigin strategy."""

    def test_calculate_origin_returns_inlier_point(self, small_data, outlier_indicator):
        """RandomOrigin returns a point from the inlier set."""
        origin = RandomOrigin(small_data, outlier_indicator)
        result = origin.calculate_origin()

        assert result.shape == (small_data.shape[1],)
        # Should be one of the original data points
        distances = np.linalg.norm(small_data - result, axis=1)
        assert np.min(distances) < 1e-10

    def test_calculate_origin_varies(self, small_data, outlier_indicator):
        """RandomOrigin returns different points on repeated calls."""
        origin = RandomOrigin(small_data, outlier_indicator)

        results = [origin.calculate_origin() for _ in range(10)]
        # At least some should be different (probabilistic, but very likely)
        unique_count = len({tuple(r) for r in results})
        assert unique_count > 1

    def test_excludes_outliers(self, small_data, mixed_outlier_indicator):
        """RandomOrigin only samples from inlier points."""
        origin = RandomOrigin(small_data, mixed_outlier_indicator)

        # First 5 points are marked as outliers
        outlier_points = small_data[:5]

        for _ in range(20):
            result = origin.calculate_origin()
            # Result should NOT be one of the outlier points
            for outlier in outlier_points:
                assert not np.allclose(result, outlier)

    def test_class_type(self, small_data, outlier_indicator):
        """RandomOrigin has correct class_type."""
        origin = RandomOrigin(small_data, outlier_indicator)
        assert origin.class_type == OriginType.RANDOM


class TestWeightedOrigin:
    """Tests for WeightedOrigin strategy."""

    def test_calculate_origin_returns_inlier_point(self, small_data, outlier_indicator):
        """WeightedOrigin returns a point from the inlier set."""
        origin = WeightedOrigin(small_data, outlier_indicator)
        result = origin.calculate_origin()

        assert result.shape == (small_data.shape[1],)
        # Should be one of the original data points
        distances = np.linalg.norm(small_data - result, axis=1)
        assert np.min(distances) < 1e-10

    def test_calculate_origin_varies(self, small_data, outlier_indicator):
        """WeightedOrigin returns different points on repeated calls."""
        origin = WeightedOrigin(small_data, outlier_indicator)

        results = [origin.calculate_origin() for _ in range(10)]
        unique_count = len({tuple(r) for r in results})
        assert unique_count > 1

    def test_class_type(self, small_data, outlier_indicator):
        """WeightedOrigin has correct class_type."""
        origin = WeightedOrigin(small_data, outlier_indicator)
        assert origin.class_type == OriginType.WEIGHTED


class TestOriginType:
    """Tests for OriginType enum."""

    def test_from_str_valid(self):
        """from_str converts valid strings to enum values."""
        assert OriginType.from_str("centroid") == OriginType.CENTROID
        assert OriginType.from_str("least outlier") == OriginType.LEAST_OUTLIER
        assert OriginType.from_str("random") == OriginType.RANDOM
        assert OriginType.from_str("weighted") == OriginType.WEIGHTED

    def test_from_str_invalid(self):
        """from_str raises ValueError for invalid strings."""
        with pytest.raises(ValueError):
            OriginType.from_str("invalid")

    def test_str_representation(self):
        """String representation matches value."""
        assert str(OriginType.CENTROID) == "centroid"
        assert str(OriginType.RANDOM) == "random"


class TestGetOriginFactory:
    """Tests for the get_origin factory function."""

    def test_get_origin_centroid(self, small_data, outlier_indicator):
        """get_origin creates CentroidOrigin for 'centroid' type."""
        origin = get_origin(small_data, outlier_indicator, "centroid")
        assert isinstance(origin, CentroidOrigin)

    def test_get_origin_least_outlier(self, small_data, outlier_indicator):
        """get_origin creates LeastOutlierOrigin for 'least outlier' type."""
        origin = get_origin(small_data, outlier_indicator, "least outlier")
        assert isinstance(origin, LeastOutlierOrigin)

    def test_get_origin_random(self, small_data, outlier_indicator):
        """get_origin creates RandomOrigin for 'random' type."""
        origin = get_origin(small_data, outlier_indicator, "random")
        assert isinstance(origin, RandomOrigin)

    def test_get_origin_weighted(self, small_data, outlier_indicator):
        """get_origin creates WeightedOrigin for 'weighted' type."""
        origin = get_origin(small_data, outlier_indicator, "weighted")
        assert isinstance(origin, WeightedOrigin)

    def test_get_origin_invalid(self, small_data, outlier_indicator):
        """get_origin raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Unknown origin method"):
            get_origin(small_data, outlier_indicator, "invalid_type")
