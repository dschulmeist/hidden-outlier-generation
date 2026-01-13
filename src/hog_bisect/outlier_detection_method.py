"""
Outlier detection method abstractions.

This module provides abstract base classes and implementations for outlier detection
methods used in the hidden outlier generation algorithm. The abstraction allows the
bisection algorithm to work with any outlier detection method that can fit on data
and predict whether a point is an outlier.

Two implementations are provided:
    - OdPYOD: Wrapper for any PyOD (Python Outlier Detection) model
    - ODmahalanobis: Statistical outlier detection using Mahalanobis distance

The PyOD wrapper persists fitted models to disk to manage memory when fitting
many subspace models in parallel. This is a tradeoff: disk I/O vs memory usage.
For datasets with many dimensions, hundreds of models may need to be fitted.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pyod.models
from pyod.models.base import BaseDetector
from scipy.spatial import distance
from scipy.stats import chi2

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OutlierDetectionMethod(ABC):
    """Abstract base class for outlier detection methods.

    All outlier detection methods must implement fit() to learn from data
    and predict() to classify new points as outliers or inliers.

    Attributes:
        name: Human-readable name of the detection method.
        model: The underlying model class (for PyOD wrappers).
        fitted: Whether the model has been fitted to data.
    """

    name: str = ""
    model: type | None = None

    def __init__(self) -> None:
        self.fitted: bool = False

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def fit(self, data: NDArray[np.floating[Any]]) -> None:
        """Fit the outlier detection model to the given data.

        Args:
            data: Training data array of shape (n_samples, n_features).
        """
        self.fitted = True

    @abstractmethod
    def predict(self, x: NDArray[np.floating[Any]]) -> bool | int:
        """Predict whether a point is an outlier.

        Args:
            x: Point to classify, shape (n_features,) or (1, n_features).

        Returns:
            True/1 if outlier, False/0 if inlier.
        """
        pass


def get_outlier_detection_method(method: type[BaseDetector] | str) -> type[OutlierDetectionMethod]:
    """Get the appropriate outlier detection wrapper for a given method.

    This factory function returns the correct OutlierDetectionMethod subclass
    for the given detection method. It supports any PyOD model or the built-in
    Mahalanobis distance method.

    Args:
        method: Either a PyOD BaseDetector subclass (e.g., pyod.models.lof.LOF)
                or the string "mahalanobis" for statistical detection.

    Returns:
        The corresponding OutlierDetectionMethod class (not an instance).

    Raises:
        ValueError: If the method is not a recognized PyOD model or "mahalanobis".

    Example:
        >>> from pyod.models.lof import LOF
        >>> detector_class = get_outlier_detection_method(LOF)
        >>> detector = detector_class(subspace=(0, 1), tempdir="/tmp")
        >>> detector.fit(data)
    """
    # Check string inputs first (before issubclass which requires a class)
    if isinstance(method, str):
        if method == "mahalanobis":
            return ODmahalanobis
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")

    # Check if it's a PyOD detector class
    if isinstance(method, type) and issubclass(method, pyod.models.base.BaseDetector):
        # Configure the class-level model for PyOD wrapper
        OdPYOD.model = method
        OdPYOD.name = method.__name__
        return OdPYOD

    raise ValueError(f"Unknown outlier detection method: {method}")


class OdPYOD(OutlierDetectionMethod):
    """Wrapper for PyOD (Python Outlier Detection) models.

    This class wraps any PyOD outlier detection model to work with the
    hidden outlier generation algorithm. Models are persisted to disk after
    fitting to manage memory when many subspace models are needed.

    The disk persistence strategy is necessary because:
    - For d dimensions, up to 2^d subspace models may be fitted
    - Each model can consume significant memory
    - Parallel fitting would exhaust memory without persistence

    Attributes:
        subspace: Tuple of dimension indices this model is fitted on.
        tempdir: Directory for storing serialized models.
        location: Full path to the serialized model file.

    Note:
        The `model` class attribute must be set before instantiation.
        This is done automatically by get_outlier_detection_method().
    """

    def __init__(self, subspace: tuple[int, ...], tempdir: str) -> None:
        """Initialize a PyOD wrapper for a specific subspace.

        Args:
            subspace: Tuple of dimension indices this detector covers.
            tempdir: Directory path for storing the serialized model.
        """
        super().__init__()
        self.tempdir = tempdir
        self.subspace = subspace
        # Use hash for unique filename - subspaces are immutable tuples
        self.name = f"ODM_on_{hash(subspace)}"
        self.location = f"{self.tempdir}/{self.name}.pkl"

    def fit(self, data: NDArray[np.floating[Any]]) -> None:
        """Fit the PyOD model and persist to disk.

        The model is fitted, serialized to disk, and then deleted from memory.
        This allows fitting many models in parallel without memory exhaustion.

        Args:
            data: Training data for this subspace, shape (n_samples, n_subspace_dims).
        """
        init_model = OdPYOD.model()
        init_model.fit(data)
        self.fitted = True
        self._dump(init_model)
        del init_model

    def _dump(self, model: BaseDetector) -> None:
        """Serialize the fitted model to disk.

        Args:
            model: The fitted PyOD model to persist.
        """
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)
        with open(self.location, "wb") as f:
            pickle.dump(model, f)

    def _load(self) -> BaseDetector:
        """Load the fitted model from disk.

        Returns:
            The deserialized PyOD model.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        with open(self.location, "rb") as f:
            return pickle.load(f)

    def predict(self, x: NDArray[np.floating[Any]]) -> bool:
        """Predict whether a point is an outlier.

        Loads the model from disk, makes prediction, then frees memory.
        This is slower than keeping models in memory but allows scaling
        to many subspaces.

        Args:
            x: Point to classify, shape (n_features,) or (1, n_features).

        Returns:
            True if the point is classified as an outlier, False otherwise.

        Raises:
            RuntimeError: If predict is called before fit.
        """
        if not self.fitted:
            raise RuntimeError("Cannot predict: model has not been fitted yet")
        fitted_model = self._load()
        x = x.reshape(1, -1)
        # PyOD predict returns array of 0/1, extract scalar and convert to bool
        decision = bool(fitted_model.predict(x)[0])
        del fitted_model
        return decision


class ODmahalanobis(OutlierDetectionMethod):
    """Statistical outlier detection using Mahalanobis distance.

    The Mahalanobis distance measures how far a point is from the center of
    a distribution, accounting for correlations between variables. Points
    with distance exceeding a chi-squared threshold are classified as outliers.

    This method assumes the data follows a multivariate normal distribution.
    It works well for elliptical clusters but may fail for non-Gaussian data.

    Attributes:
        mean: Mean vector of the fitted data.
        inv_cov: Inverse covariance matrix of the fitted data.
        crit_val: Critical distance threshold for outlier classification.
        shape: Shape of the training data (n_samples, n_features).
    """

    name: str = "mahalanobis"

    def __init__(
        self,
        _subspace: tuple[int, ...] | None = None,
        _tempdir: str | None = None,
    ) -> None:
        """Initialize the Mahalanobis detector.

        Args:
            _subspace: Unused, kept for interface compatibility with OdPYOD.
            _tempdir: Unused, kept for interface compatibility with OdPYOD.
        """
        super().__init__()
        self.crit_val: float | None = None
        self.mean: NDArray[np.floating[Any]] | None = None
        self.inv_cov: NDArray[np.floating[Any]] | None = None
        self.shape: tuple[int, int] | None = None

    def fit(self, data: NDArray[np.floating[Any]]) -> None:
        """Fit the Mahalanobis detector by computing mean and covariance.

        Args:
            data: Training data array of shape (n_samples, n_features).

        Raises:
            numpy.linalg.LinAlgError: If the covariance matrix is singular.
        """
        self.shape = data.shape
        self.mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        self.inv_cov = np.linalg.inv(cov)
        self.crit_val = self._compute_critical_value()
        self.fitted = True

    def predict(self, x: NDArray[np.floating[Any]]) -> int:
        """Predict whether a point is an outlier based on Mahalanobis distance.

        Args:
            x: Point to classify, shape (n_features,).

        Returns:
            1 if outlier (distance > critical value), 0 if inlier.

        Raises:
            RuntimeError: If predict is called before fit.
        """
        if not self.fitted:
            raise RuntimeError("Cannot predict: model has not been fitted yet")
        mahalanobis_dist = distance.mahalanobis(x, self.mean, self.inv_cov)
        return 1 if mahalanobis_dist > self.crit_val else 0

    def _compute_critical_value(self) -> float:
        """Calculate the critical Mahalanobis distance threshold.

        Uses the 95th percentile of the chi-squared distribution with
        degrees of freedom equal to the number of dimensions. This means
        approximately 5% of normally distributed data would be classified
        as outliers.

        Returns:
            The critical distance threshold.
        """
        return chi2.ppf(0.95, df=self.shape[1])
