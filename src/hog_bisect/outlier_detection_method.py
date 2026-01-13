"""
Outlier detection method abstractions.

This module provides abstract base classes and implementations for outlier detection
methods used in the hidden outlier generation algorithm. It supports both PyOD-based
detectors and custom implementations like Mahalanobis distance.
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pyod.models
from pyod.models.base import BaseDetector
from scipy.spatial import distance
from scipy.stats import chi2


class OutlierDetectionMethod(ABC):
    name = ""
    model = None

    def __init__(self):
        self.fitted = False

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self, data):
        self.fitted = True
        pass

    @abstractmethod
    def predict(self, x):
        pass


def get_outlier_detection_method(method: Any) -> type:
    """Get the appropriate outlier detection wrapper for a given method.

    Args:
        method: Either a PyOD BaseDetector subclass or the string "mahalanobis".

    Returns:
        The corresponding OutlierDetectionMethod class.

    Raises:
        ValueError: If the method is not recognized.
    """
    if pyod.models.base.BaseDetector.__subclasscheck__(method):
        OdPYOD.model = method
        OdPYOD.name = method.__name__
        return OdPYOD
    elif method == "mahalanobis":
        return ODmahalanobis
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


# class for pyod based outlier detection methods
class OdPYOD(OutlierDetectionMethod):
    def __init__(self, subspace, tempdir):
        super().__init__()
        self.tempdir = tempdir
        self.subspace = subspace
        self.name = "ODM_on_" + str(hash(subspace))
        self.location = f"{self.tempdir}/{self.name}.pkl"

    def fit(self, data):
        init_model = OdPYOD.model()
        init_model.fit(data)
        self.fitted = True
        self.dump(init_model)
        del init_model

    def dump(self, model):
        # check whether the directory tempdir exists or not
        # if not, create a new one
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)
        # dump the model to tempdir
        with open(self.location, "wb") as f:
            pickle.dump(model, f)
        del model

    def get_model(self) -> BaseDetector:
        # load model from disk
        with open(self.location, "rb") as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def predict(self, x) -> bool:
        if not self.fitted:
            raise RuntimeError("Cannot predict: model has not been fitted yet")
        fitted_model = self.get_model()
        x = x.reshape(1, -1)
        decision = bool(fitted_model.predict(x)[0,])
        del fitted_model
        return decision


class ODmahalanobis(OutlierDetectionMethod):
    """Mahalanobis distance-based outlier detection."""

    name = "mahalanobis"

    def __init__(self, _subspace=None, _tempdir=None):
        # Arguments kept for interface compatibility with OdPYOD
        super().__init__()
        self.crit_val = None
        self.mean = None
        self.inv_cov = None
        self.shape = None

    def fit(self, data):
        self.shape = data.shape
        self.mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        self.inv_cov = np.linalg.inv(cov)
        self.crit_val = self.critval()
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise RuntimeError("Cannot predict: model has not been fitted yet")
        mahalanobis_dist = distance.mahalanobis(x, self.mean, self.inv_cov)
        return 1 if mahalanobis_dist > self.crit_val else 0

    def critval(self):
        """Calculate the critical value for outlier classification.

        Uses chi-squared distribution at 95% confidence level based on the
        number of dimensions in the data.

        Returns:
            float: The critical Mahalanobis distance threshold.
        """
        return chi2.ppf(0.95, df=self.shape[1])


# %%
