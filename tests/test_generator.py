"""
Tests for the BisectHOGen hidden outlier generator class.

These tests verify the main public API works correctly.
"""

import numpy as np
import pytest
from pyod.models.knn import KNN

from hog_bisect import BisectHOGen


class TestBisectHOGenInit:
    """Tests for BisectHOGen initialization."""

    def test_init_basic(self, small_data):
        """Generator initializes with required parameters."""
        gen = BisectHOGen(data=small_data)
        assert gen.data is small_data
        assert gen.dims == 3
        assert gen.full_space == (0, 1, 2)

    def test_init_with_custom_detector(self, small_data):
        """Generator accepts custom outlier detection method."""
        gen = BisectHOGen(data=small_data, outlier_detection_method=KNN)
        assert gen.outlier_detection_method is KNN

    def test_init_with_seed(self, small_data):
        """Generator accepts seed parameter."""
        gen = BisectHOGen(data=small_data, seed=123)
        assert gen.seed == 123

    def test_init_with_max_dimensions(self, small_data):
        """Generator accepts max_dimensions parameter."""
        gen = BisectHOGen(data=small_data, max_dimensions=5)
        assert gen.max_dimensions == 5


class TestBisectHOGenFitGenerate:
    """Tests for the fit_generate method."""

    def test_fit_generate_returns_array(self, small_data):
        """fit_generate returns a numpy array."""
        gen = BisectHOGen(data=small_data, seed=42)
        result = gen.fit_generate(gen_points=10, n_jobs=1)
        assert isinstance(result, np.ndarray)

    def test_fit_generate_correct_dimensions(self, small_data):
        """Generated points have same feature count as input data."""
        gen = BisectHOGen(data=small_data, seed=42)
        result = gen.fit_generate(gen_points=10, n_jobs=1)
        if len(result) > 0:
            assert result.shape[1] == small_data.shape[1]

    def test_fit_generate_stores_results(self, small_data):
        """fit_generate stores results in instance attributes."""
        gen = BisectHOGen(data=small_data, seed=42)
        gen.fit_generate(gen_points=10, n_jobs=1)
        assert gen.hidden_x_list is not None
        assert gen.hidden_x_type is not None
        assert gen.exec_time is not None
        assert gen.exec_time > 0

    def test_fit_generate_with_different_origins(self, small_data):
        """fit_generate works with all origin types."""
        origin_types = ["centroid", "weighted", "random", "least outlier"]
        for origin in origin_types:
            gen = BisectHOGen(data=small_data, seed=42)
            result = gen.fit_generate(gen_points=5, get_origin_type=origin, n_jobs=1)
            assert isinstance(result, np.ndarray)

    def test_fit_generate_reproducible(self, small_data):
        """Same seed produces same results."""
        gen1 = BisectHOGen(data=small_data, seed=42)
        result1 = gen1.fit_generate(gen_points=10, n_jobs=1)

        gen2 = BisectHOGen(data=small_data, seed=42)
        result2 = gen2.fit_generate(gen_points=10, n_jobs=1)

        np.testing.assert_array_equal(result1, result2)

    def test_fit_generate_different_seeds_differ(self, small_data):
        """Different seeds produce different results."""
        gen1 = BisectHOGen(data=small_data, seed=42)
        result1 = gen1.fit_generate(gen_points=20, n_jobs=1)

        gen2 = BisectHOGen(data=small_data, seed=99)
        result2 = gen2.fit_generate(gen_points=20, n_jobs=1)

        # Results should differ (unless both happen to find 0 outliers)
        if len(result1) > 0 and len(result2) > 0:
            assert not np.array_equal(result1, result2)


class TestBisectHOGenEdgeCases:
    """Edge case tests for BisectHOGen."""

    def test_single_feature_data(self):
        """Generator handles 1D data."""
        data = np.random.randn(50, 1)
        gen = BisectHOGen(data=data, seed=42)
        # Should not crash, though results may be limited
        result = gen.fit_generate(gen_points=5, n_jobs=1)
        assert isinstance(result, np.ndarray)

    def test_two_feature_data(self):
        """Generator handles 2D data."""
        data = np.random.randn(50, 2)
        gen = BisectHOGen(data=data, seed=42)
        result = gen.fit_generate(gen_points=10, n_jobs=1)
        assert isinstance(result, np.ndarray)

    def test_high_dimensional_data(self):
        """Generator handles high-dimensional data (triggers random subspaces)."""
        data = np.random.randn(100, 15)  # Above default max_dimensions=11
        gen = BisectHOGen(data=data, seed=42, max_dimensions=11)
        result = gen.fit_generate(gen_points=5, n_jobs=1)
        assert isinstance(result, np.ndarray)

    def test_small_dataset(self):
        """Generator handles small datasets."""
        data = np.random.randn(20, 3)
        gen = BisectHOGen(data=data, seed=42)
        result = gen.fit_generate(gen_points=5, n_jobs=1)
        assert isinstance(result, np.ndarray)


class TestBisectHOGenOutputMethods:
    """Tests for output methods (print_summary, save_to_csv)."""

    def test_print_summary_after_generate(self, small_data, capsys):
        """print_summary works after generation."""
        gen = BisectHOGen(data=small_data, seed=42)
        gen.fit_generate(gen_points=10, n_jobs=1)
        gen.print_summary()
        captured = capsys.readouterr()
        assert "Hidden Outlier Generation Summary" in captured.out
        assert "Features:" in captured.out

    def test_print_summary_before_generate(self, small_data, capsys):
        """print_summary handles case when fit_generate not called."""
        gen = BisectHOGen(data=small_data, seed=42)
        gen.print_summary()
        captured = capsys.readouterr()
        assert "No generation results available" in captured.out

    def test_save_to_csv(self, small_data, tmp_path):
        """save_to_csv writes results to file."""
        gen = BisectHOGen(data=small_data, seed=42)
        gen.fit_generate(gen_points=10, n_jobs=1)

        csv_path = tmp_path / "outliers.csv"
        gen.save_to_csv(str(csv_path))

        assert csv_path.exists()
        loaded = np.loadtxt(str(csv_path), delimiter=",")
        if len(gen.hidden_x_list) > 0:
            assert loaded.shape == gen.hidden_x_list.shape

    def test_save_to_csv_before_generate(self, small_data, tmp_path):
        """save_to_csv raises error if called before fit_generate."""
        gen = BisectHOGen(data=small_data, seed=42)
        csv_path = tmp_path / "outliers.csv"
        with pytest.raises(ValueError, match="No results to save"):
            gen.save_to_csv(str(csv_path))
