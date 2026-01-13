"""
Hidden Outlier Generation using the Bisection Algorithm.

This package provides tools for generating synthetic hidden outliers - data points
that are outliers in some feature subspaces but not in the full feature space
(or vice versa). These hidden outliers are useful for:

- Benchmarking outlier detection algorithms
- Testing subspace-aware detection methods
- Understanding the phenomenon of outlier hiding in high dimensions

Main Classes:
    BisectHOGen: The main class for generating hidden outliers using bisection.

Quick Start:
    >>> from hog_bisect import BisectHOGen
    >>> from pyod.models.lof import LOF
    >>> import numpy as np
    >>>
    >>> # Generate some normal data
    >>> data = np.random.randn(100, 5)
    >>>
    >>> # Create generator and produce hidden outliers
    >>> generator = BisectHOGen(data, outlier_detection_method=LOF, seed=42)
    >>> hidden_outliers = generator.fit_generate(gen_points=50)

See Also:
    - OutlierResultType: Enum defining the possible outlier classifications
    - OriginMethod: Strategy classes for choosing the search origin
    - OutlierDetectionMethod: Wrappers for outlier detection algorithms
"""

from hog_bisect.bisect import BisectHOGen
from hog_bisect.origin_method import (
    CentroidOrigin,
    LeastOutlierOrigin,
    OriginMethod,
    OriginType,
    RandomOrigin,
    WeightedOrigin,
    get_origin,
)
from hog_bisect.outlier_detection_method import (
    ODmahalanobis,
    OdPYOD,
    OutlierDetectionMethod,
    get_outlier_detection_method,
)
from hog_bisect.outlier_result_type import OutlierResultType

__version__ = "1.0.1"

__all__ = [
    # Main class
    "BisectHOGen",
    # Result types
    "OutlierResultType",
    # Origin methods
    "OriginMethod",
    "OriginType",
    "CentroidOrigin",
    "LeastOutlierOrigin",
    "RandomOrigin",
    "WeightedOrigin",
    "get_origin",
    # Detection methods
    "OutlierDetectionMethod",
    "OdPYOD",
    "ODmahalanobis",
    "get_outlier_detection_method",
    # Version
    "__version__",
]
