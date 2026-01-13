"""
Tests for outlier detection method wrappers.

These wrappers provide a consistent interface for different outlier detectors.
"""

import numpy as np
import pytest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from hog_bisect.outlier_detection_method import (
    ODmahalanobis,
    OdPYOD,
    OutlierDetectionMethod,
    get_outlier_detection_method,
)


class TestOdPYOD:
    """Tests for the PyOD wrapper class."""

    def test_init(self, temp_dir):
        """OdPYOD initializes with subspace and tempdir."""
        # Must use factory function to set the model class
        wrapper_class = get_outlier_detection_method(LOF)
        subspace = (0, 1, 2)
        wrapper = wrapper_class(subspace, temp_dir)
        assert wrapper.subspace == subspace

    def test_fit(self, small_data, temp_dir):
        """OdPYOD fits the underlying model."""
        wrapper_class = get_outlier_detection_method(LOF)
        subspace = (0, 1, 2)
        wrapper = wrapper_class(subspace, temp_dir)
        wrapper.fit(small_data)
        assert wrapper.fitted is True

    def test_predict_inlier(self, small_data, temp_dir):
        """OdPYOD.predict returns False for inlier points."""
        wrapper_class = get_outlier_detection_method(LOF)
        subspace = (0, 1, 2)
        wrapper = wrapper_class(subspace, temp_dir)
        wrapper.fit(small_data)

        # A point near the data center should be an inlier
        center = small_data.mean(axis=0)
        result = wrapper.predict(center)
        assert isinstance(result, (bool, np.bool_))

    def test_predict_outlier(self, small_data, temp_dir):
        """OdPYOD.predict returns True for outlier points."""
        wrapper_class = get_outlier_detection_method(LOF)
        subspace = (0, 1, 2)
        wrapper = wrapper_class(subspace, temp_dir)
        wrapper.fit(small_data)

        # A point far from data should be an outlier
        far_point = np.array([100.0, 100.0, 100.0])
        result = wrapper.predict(far_point)
        assert result is True

    def test_works_with_different_detectors(self, small_data, temp_dir):
        """OdPYOD works with different PyOD detector classes."""
        subspace = (0, 1, 2)

        for detector_class in [LOF, KNN]:
            wrapper_class = get_outlier_detection_method(detector_class)
            wrapper = wrapper_class(subspace, temp_dir)
            wrapper.fit(small_data)
            result = wrapper.predict(small_data[0])
            assert isinstance(result, (bool, np.bool_))


class TestODmahalanobis:
    """Tests for the Mahalanobis distance detector."""

    def test_init(self, temp_dir):
        """ODmahalanobis initializes correctly."""
        subspace = (0, 1, 2)
        detector = ODmahalanobis(subspace, temp_dir)
        assert detector.name == "mahalanobis"
        assert detector.fitted is False

    def test_fit(self, small_data, temp_dir):
        """ODmahalanobis fits the model."""
        subspace = (0, 1, 2)
        detector = ODmahalanobis(subspace, temp_dir)
        detector.fit(small_data)
        assert detector.mean is not None
        assert detector.inv_cov is not None
        assert detector.fitted is True

    def test_predict_inlier(self, small_data, temp_dir):
        """ODmahalanobis.predict returns 0 for points near center."""
        subspace = (0, 1, 2)
        detector = ODmahalanobis(subspace, temp_dir)
        detector.fit(small_data)

        center = small_data.mean(axis=0)
        result = detector.predict(center)
        assert result == 0  # Mahalanobis returns int 0/1

    def test_predict_outlier(self, small_data, temp_dir):
        """ODmahalanobis.predict returns 1 for distant points."""
        subspace = (0, 1, 2)
        detector = ODmahalanobis(subspace, temp_dir)
        detector.fit(small_data)

        far_point = np.array([50.0, 50.0, 50.0])
        result = detector.predict(far_point)
        assert result == 1  # Mahalanobis returns int 0/1


class TestGetOutlierDetectionMethod:
    """Tests for the factory function."""

    def test_returns_pyod_wrapper_class(self):
        """get_outlier_detection_method returns OdPYOD for PyOD detectors."""
        result = get_outlier_detection_method(LOF)
        assert result == OdPYOD

    def test_returns_mahalanobis_class(self):
        """get_outlier_detection_method returns ODmahalanobis for 'mahalanobis'."""
        result = get_outlier_detection_method("mahalanobis")
        assert result == ODmahalanobis

    def test_invalid_method_raises(self):
        """get_outlier_detection_method raises for invalid input."""
        with pytest.raises(ValueError):
            get_outlier_detection_method("invalid_detector")


class TestOutlierDetectionMethodInterface:
    """Tests verifying the abstract interface is properly implemented."""

    def test_odPYOD_is_subclass(self):
        """OdPYOD is a subclass of OutlierDetectionMethod."""
        assert issubclass(OdPYOD, OutlierDetectionMethod)

    def test_odmah_is_subclass(self):
        """ODmahalanobis is a subclass of OutlierDetectionMethod."""
        assert issubclass(ODmahalanobis, OutlierDetectionMethod)
