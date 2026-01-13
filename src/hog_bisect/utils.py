"""
Utility functions for subspace operations and model fitting.

This module provides helper functions for:
    - Generating random directions on a unit sphere
    - Creating and manipulating subspaces of the feature space
    - Fitting outlier detection models across multiple subspaces in parallel

The subspace handling is central to hidden outlier detection: a point may be
an outlier in some subspaces but not others. By fitting models on all relevant
subspaces, we can identify these "hidden" outliers.
"""

from __future__ import annotations

import logging
import random
from itertools import chain, combinations
from typing import TYPE_CHECKING, Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from hog_bisect.outlier_detection_method import (
    OutlierDetectionMethod,
    get_outlier_detection_method,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyod.models.base import BaseDetector

# Timeout for parallel model fitting (seconds)
FITTING_TIMEOUT_TIME = 60

# Default random seed for reproducibility
DEFAULT_SEED = 5


def random_unif_on_sphere(
    number: int,
    dimensions: int,
    r: float = 1.0,
    random_state: int = 5,
) -> NDArray[np.floating[Any]]:
    """Generate uniformly distributed random points on a sphere surface.

    Uses the Muller method: generate normally distributed points and normalize
    them to lie on the sphere surface. This ensures uniform distribution over
    the sphere, unlike naive approaches that would concentrate at poles.

    Args:
        number: Number of points to generate.
        dimensions: Dimensionality of the space (sphere lives in R^dimensions).
        r: Radius of the sphere. Defaults to 1.0 (unit sphere).
        random_state: Seed for reproducible random generation.

    Returns:
        Array of shape (number, dimensions) with points on the sphere surface.

    Example:
        >>> directions = random_unif_on_sphere(100, 3)  # 100 directions in 3D
        >>> np.allclose(np.linalg.norm(directions, axis=1), 1.0)
        True
    """
    normal_deviates = norm.rvs(size=(number, dimensions), random_state=random_state)
    radius = np.sqrt((normal_deviates**2).sum(axis=1))[:, np.newaxis]
    points = normal_deviates / radius
    return points * r


def gen_powerset(dims: int) -> set[tuple[int, ...]]:
    """Generate all non-empty proper subsets of dimension indices.

    For d dimensions, generates all 2^d - 2 subsets (excluding empty set
    and the full set). Used when dimensionality is low enough to enumerate
    all subspaces.

    Args:
        dims: Number of dimensions (features) in the data.

    Returns:
        Set of tuples, each tuple contains indices forming a subspace.

    Example:
        >>> gen_powerset(3)
        {(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)}
    """
    return set(chain.from_iterable(combinations(range(dims), r) for r in range(1, dims)))


def subspace_grab(
    indices: tuple[int, ...],
    data: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Extract a subspace (subset of columns) from the data.

    Args:
        indices: Tuple of column indices to extract.
        data: Full dataset of shape (n_samples, n_features).

    Returns:
        Subspace data of shape (n_samples, len(indices)).

    Example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6]])
        >>> subspace_grab((0, 2), data)
        array([[1, 3], [4, 6]])
    """
    return data[:, np.array(indices)]


def gen_rand_subspaces(
    dims: int,
    upper_limit: int,
    include_all_attr: bool = True,
    seed: int = 5,
) -> set[tuple[int, ...]]:
    """Generate a random sample of subspaces when full enumeration is infeasible.

    For high-dimensional data, enumerating all 2^d subspaces is impossible.
    This function generates a random but representative sample that:
    - Includes all singleton subspaces (single features)
    - Includes at least one multi-feature subspace containing each feature
    - Randomly samples additional subspaces up to the limit

    The sampling ensures every feature is represented while keeping the
    total number of subspaces manageable.

    Args:
        dims: Number of dimensions (features) in the data.
        upper_limit: Generate up to 2^upper_limit - 2 subspaces.
        include_all_attr: If True, ensure every feature appears in at least
            one singleton and one multi-feature subspace.
        seed: Random seed for reproducible subspace selection.

    Returns:
        Set of tuples representing the selected subspaces.
    """
    rd = random.Random(seed)
    features = list(range(dims))
    rd.shuffle(features)
    subspaces: set[tuple[int, ...]] = set()

    # Ensure each feature is represented in both singleton and multi-feature subspaces
    if include_all_attr:
        for i in features:
            # Create a random multi-feature subspace containing feature i
            r = rd.randint(2, dims - 1)
            fts = rd.sample(range(dims), r)
            fts.append(i)
            subspace1 = tuple(fts)
            subspace2 = (i,)  # Singleton subspace

            if subspace1 not in subspaces:
                subspaces.add(subspace1)
            if subspace2 not in subspaces:
                subspaces.add(subspace2)

    # Fill remaining slots with random subspaces
    # Avoid singletons if they're already included from the loop above
    lower_limit = 2 if include_all_attr else 1

    target_count = (2**upper_limit) - 2
    while len(subspaces) < target_count:
        r = rd.randint(lower_limit, dims - 1)
        random_comb = tuple(rd.sample(range(dims), r))
        if random_comb not in subspaces:
            subspaces.add(random_comb)

    return subspaces


def fit_model(
    subspace: tuple[int, ...],
    data: NDArray[np.floating[Any]],
    outlier_detection_method: type[BaseDetector],
    tempdir: str,
) -> tuple[tuple[int, ...], OutlierDetectionMethod]:
    """Fit an outlier detection model for a single subspace.

    This function is designed to be called in parallel via joblib.
    It creates a detector, fits it on the subspace projection of the data,
    and returns the fitted model.

    Args:
        subspace: Tuple of dimension indices defining the subspace.
        data: Full dataset of shape (n_samples, n_features).
        outlier_detection_method: PyOD detector class to use.
        tempdir: Directory for storing serialized models.

    Returns:
        Tuple of (subspace, fitted_model) for building the subspace dictionary.
    """
    odm = get_outlier_detection_method(outlier_detection_method)
    model = odm(subspace, tempdir)
    model.fit(subspace_grab(subspace, data))
    return subspace, model


def fit_in_all_subspaces(
    outlier_detection_method: type[BaseDetector],
    data: NDArray[np.floating[Any]],
    tempdir: str,
    subspace_limit: int,
    seed: int = DEFAULT_SEED,
    n_jobs: int = 1,
) -> dict[tuple[int, ...], OutlierDetectionMethod]:
    """Fit outlier detection models for all relevant subspaces in parallel.

    For low-dimensional data (dims < subspace_limit), fits models on all
    possible subspaces. For high-dimensional data, uses random sampling
    to select a representative subset of subspaces.

    The full space is always included and fitted separately at the end.

    Args:
        outlier_detection_method: PyOD detector class to use (e.g., LOF).
        data: Dataset of shape (n_samples, n_features).
        tempdir: Directory for storing serialized models.
        subspace_limit: If dims >= this, use random sampling instead of
            full enumeration. Limits subspaces to ~2^subspace_limit.
        seed: Random seed for reproducible subspace selection.
        n_jobs: Number of parallel jobs for fitting. Use -1 for all cores.

    Returns:
        Dictionary mapping subspace tuples to fitted OutlierDetectionMethod
        instances, including the full space.

    Example:
        >>> from pyod.models.lof import LOF
        >>> models = fit_in_all_subspaces(LOF, data, "/tmp", subspace_limit=10)
        >>> full_space = tuple(range(data.shape[1]))
        >>> models[full_space].predict(data[0])
    """
    dims = data.shape[1]

    # Choose subspace enumeration strategy based on dimensionality
    if dims < subspace_limit:
        subspaces = gen_powerset(dims)
    else:
        subspaces = gen_rand_subspaces(dims, subspace_limit, include_all_attr=True, seed=seed)

    logging.info("Fitting models on all subspaces...")
    logging.debug(f"Number of subspaces: {len(subspaces)}, parallel jobs: {n_jobs}")

    # Parallel model fitting across subspaces
    results = Parallel(n_jobs=n_jobs, timeout=FITTING_TIMEOUT_TIME)(
        delayed(fit_model)(subspace, data, outlier_detection_method, tempdir)
        for subspace in subspaces
    )
    fitted_subspaces = dict(results)

    # Always fit on the full space separately
    logging.info("Fitting model on full space...")
    full_space = tuple(range(dims))
    fitted_subspaces[full_space] = get_outlier_detection_method(outlier_detection_method)(
        full_space, tempdir
    )
    fitted_subspaces[full_space].fit(data)

    del results  # Free memory from parallel results

    logging.info(f"Fitted {len(fitted_subspaces)} subspace models (including full space)")
    return fitted_subspaces
