"""
Bisection algorithm for hidden outlier generation.

This module implements the core bisection algorithm for generating synthetic
hidden outliers. Hidden outliers are points that exhibit outlier behavior in
some feature subspaces but not in the full feature space (H1), or vice versa (H2).

The algorithm works by:
1. Fitting outlier detection models on all relevant subspaces
2. Selecting a random direction from an origin point
3. Using bisection to find the boundary between inlier and outlier regions
4. Classifying the boundary point as H1, H2, or neither

Key Components:
    - BisectHOGen: Main class for generating hidden outliers
    - bisect: Core bisection algorithm function
    - outlier_check: Classifies a point as H1, H2, OB (outside bounds), or IL (inlier)
    - inference: Checks if a point is an outlier in a specific subspace

The bisection approach is more efficient than random sampling because it
specifically targets the boundary regions where hidden outliers exist.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pyod.models.lof
from joblib import Parallel, delayed
from pyod.models.base import BaseDetector

from hog_bisect import origin_method, utils
from hog_bisect.outlier_detection_method import (
    OutlierDetectionMethod,
    get_outlier_detection_method,
)
from hog_bisect.outlier_result_type import OutlierResultType
from hog_bisect.utils import fit_in_all_subspaces, subspace_grab

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Configuration constants
DEFAULT_MAX_DIMENSIONS = 11  # Max dims before switching to random subspace sampling
FITTING_TIMEOUT_TIME = 60  # Timeout for parallel fitting operations (seconds)
DEFAULT_SEED = 5  # Default random seed for reproducibility
DEFAULT_NUMBER_OF_ITERATIONS = 30  # Bisection iterations per point
DEFAULT_NUMBER_OF_GEN_POINTS = 100  # Default number of points to generate
DEFAULT_C = 0.5  # Initial midpoint for bisection
DEFAULT_NUMBER_OF_PARTS = 20  # Segments for interval checking


def outlier_check(
    x: NDArray[np.floating[Any]],
    full_space: tuple[int, ...],
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod],
    verb: bool = False,  # noqa: ARG001 - kept for API compatibility
    fast: bool = True,
) -> OutlierResultType:
    """Classify a point based on its outlier status in full space vs subspaces.

    This is the core classification function that determines whether a point
    is a hidden outlier. It checks the point against all fitted subspace models
    and the full space model.

    The classification logic:
        - H1: Outlier in subspace(s) but NOT in full space (hidden type 1)
        - H2: Outlier in full space but NOT in any subspace (hidden type 2)
        - OB: Outlier in both full space AND subspace(s) (outside bounds)
        - IL: Not an outlier anywhere (inlier)

    Args:
        x: Point to classify, shape (n_features,).
        full_space: Tuple of all dimension indices.
        fitted_subspaces: Dictionary mapping subspace tuples to fitted models.
        verb: Unused, kept for backwards compatibility.
        fast: If True, stop checking subspaces after finding first outlier.
            Faster but may miss some subspace outliers.

    Returns:
        OutlierResultType indicating the point's classification.
    """
    subspaces = fitted_subspaces.keys()
    index = np.zeros(len(subspaces))
    is_outlier_in_full_space = False
    is_outlier_in_subspace = False

    # Check if outlier in full space
    if inference(x, full_space, fitted_subspaces=fitted_subspaces):
        is_outlier_in_full_space = True

    # Check each subspace (excluding full space)
    for j, subspace in enumerate(subspaces):
        if subspace == full_space or len(subspace) == len(full_space):
            continue

        # Project point to subspace and check
        x_subspace = subspace_grab(subspace, x.reshape(1, x.shape[0]))
        if inference(x_subspace, subspace, fitted_subspaces=fitted_subspaces):
            index[j] = 1
            if fast:
                break

    if np.sum(index[:-1]) > 0:
        is_outlier_in_subspace = True

    # Classify based on outlier status in full space vs subspaces
    if is_outlier_in_subspace and not is_outlier_in_full_space:
        result = OutlierResultType.H1
    elif not is_outlier_in_subspace and is_outlier_in_full_space:
        logging.debug("Found H2: outlier in full space but not in subspaces")
        result = OutlierResultType.H2
    elif is_outlier_in_subspace and is_outlier_in_full_space:
        result = OutlierResultType.OB
    else:
        result = OutlierResultType.IL

    return result


def validate_subspace(
    subspace: tuple[int, ...],
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod],
) -> None:
    """Validate that a subspace exists in the fitted models dictionary.

    Args:
        subspace: Tuple of dimension indices to validate.
        fitted_subspaces: Dictionary of fitted models.

    Raises:
        ValueError: If subspace is not a tuple or not in fitted_subspaces.
    """
    if not isinstance(subspace, tuple):
        raise ValueError(f"Subspace must be a tuple, got {type(subspace)}")
    if subspace not in fitted_subspaces:
        raise ValueError(f"Subspace {subspace} not found in fitted models")


def inference(
    x: NDArray[np.floating[Any]],
    subspace: tuple[int, ...],
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod],
) -> bool:
    """Check if a point is an outlier in the specified subspace.

    Args:
        x: Point to check, already projected to the subspace dimensions.
        subspace: Tuple of dimension indices identifying which model to use.
        fitted_subspaces: Dictionary mapping subspace tuples to fitted models.

    Returns:
        True if the point is classified as an outlier, False otherwise.

    Raises:
        ValueError: If the subspace is invalid or not found.
    """
    validate_subspace(subspace, fitted_subspaces)
    return bool(fitted_subspaces[subspace].predict(x))


def get_segmentation_points(length: float, parts: int) -> NDArray[np.floating[Any]]:
    """Generate evenly spaced points along an interval for checking.

    Args:
        length: Total length of the interval.
        parts: Number of segments (will create parts+1 points).

    Returns:
        Array of evenly spaced values from 0 to length.
    """
    return np.linspace(0, length, num=parts)


def interval_check(
    length: float,
    direction: NDArray[np.floating[Any]],
    origin: NDArray[np.floating[Any]],
    full_space: tuple[int, ...],
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod],
    parts: int = DEFAULT_NUMBER_OF_PARTS,
) -> list[list[tuple[float, float] | tuple[int, int]]]:
    """Find intervals along a ray where outlier status changes.

    Checks points along origin + c*direction for c in [0, length] and
    identifies intervals where the outlier classification transitions.
    These transition intervals are where hidden outliers may exist.

    If no transitions are found, recursively doubles the length to search
    further from the origin.

    Args:
        length: How far to search along the direction.
        direction: Unit vector defining the search direction.
        origin: Starting point for the ray.
        full_space: Tuple of all dimension indices.
        fitted_subspaces: Dictionary of fitted models.
        parts: Number of points to check along the interval.

    Returns:
        List of intervals, each containing [(start, end), (start_label, end_label)]
        where labels are 1 (outlier) or -1 (inlier) in the full space.
    """
    segmentation_points = get_segmentation_points(length, parts)
    outlier_status = np.full(len(segmentation_points), -1)

    # Check outlier status at each segmentation point
    for i, c in enumerate(segmentation_points):
        point_to_check = c * direction + origin
        outlier_status[i] = 1 if inference(point_to_check, full_space, fitted_subspaces) else -1

    intervals = construct_intervals(segmentation_points, outlier_status)

    # If no transition intervals found, expand the search
    if not intervals:
        if outlier_status[0] == 1:
            logging.debug("No transitions found, returning full interval")
            return [
                [
                    (segmentation_points[0], segmentation_points[-1]),
                    (outlier_status[0], outlier_status[-1]),
                ]
            ]
        else:
            logging.debug("No transitions found, doubling search length")
            return interval_check(
                length * 2, direction, origin, full_space, fitted_subspaces, parts=parts
            )

    return intervals


def construct_intervals(
    segmentation_points: NDArray[np.floating[Any]],
    outlier_status: NDArray[np.integer[Any]],
) -> list[list[tuple[float, float] | tuple[int, int]]]:
    """Build list of intervals where outlier status transitions.

    Args:
        segmentation_points: Array of positions along the ray.
        outlier_status: Array of outlier labels (1 or -1) at each position.

    Returns:
        List of transition intervals with their endpoint labels.
    """
    intervals = []
    previous = outlier_status[0]

    for i in range(1, len(outlier_status)):
        if outlier_status[i] != previous:
            intervals.append(
                [
                    (segmentation_points[i - 1], segmentation_points[i]),
                    (outlier_status[i - 1], outlier_status[i]),
                ]
            )
        previous = outlier_status[i]

    return intervals


def bisect(
    direction: NDArray[np.floating[Any]],
    interval_length: float,
    origin: NDArray[np.floating[Any]],
    number_of_iterations: int = DEFAULT_NUMBER_OF_ITERATIONS,
    is_check_fast: bool = True,
    is_fixed_interval_length: bool = True,
    full_space: tuple[int, ...] | None = None,
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod] | None = None,
    is_verbose: bool = True,
) -> tuple[float, OutlierResultType]:
    """Execute the bisection algorithm to find a hidden outlier.

    The bisection algorithm searches for points at the boundary between
    outlier and inlier regions. By iteratively halving the search interval,
    it converges to a point that may be a hidden outlier (H1 or H2).

    Algorithm:
        1. Find intervals where outlier status transitions
        2. Randomly select one transition interval
        3. Bisect the interval, checking the midpoint classification
        4. Narrow to the half containing the boundary
        5. Repeat until finding H1/H2 or reaching max iterations

    Args:
        direction: Unit vector defining the search direction.
        interval_length: Initial length to search along direction.
        origin: Starting point for the search ray.
        number_of_iterations: Maximum bisection iterations.
        is_check_fast: If True, use fast outlier checking.
        is_fixed_interval_length: If False, randomly perturb the interval length.
        full_space: Tuple of all dimension indices.
        fitted_subspaces: Dictionary of fitted models.
        is_verbose: Enable verbose logging.

    Returns:
        Tuple of (c, result_type) where the found point is origin + c*direction
        and result_type is the OutlierResultType classification.
    """
    check_result = None
    c = DEFAULT_C

    # Validate required parameters
    assert full_space is not None, "full_space must be provided for bisection"
    assert fitted_subspaces is not None, "fitted_subspaces must be provided for bisection"

    # Optionally add randomness to interval length
    if not is_fixed_interval_length:
        interval_length = interval_length + np.random.uniform(-interval_length / 2, interval_length)

    # Find transition intervals
    found_sub_intervals = interval_check(
        interval_length, direction, origin, full_space=full_space, fitted_subspaces=fitted_subspaces
    )

    # Randomly select one interval to bisect
    choice = np.random.choice(len(found_sub_intervals), 1)[0]
    chosen_interval = found_sub_intervals[choice]
    interval_indicator = chosen_interval[1]
    interval = chosen_interval[0]

    a = interval[0]
    b = interval[1]

    # Bisection loop
    for i in range(number_of_iterations):
        c = (b + a) / 2
        check_result = outlier_check(
            c * direction + origin,
            fast=is_check_fast,
            full_space=full_space,
            fitted_subspaces=fitted_subspaces,
            verb=is_verbose,
        )

        # Found a hidden outlier - return immediately
        if check_result in (OutlierResultType.H1, OutlierResultType.H2):
            return c, check_result

        # Narrow the search interval based on classification
        outlier_indicator = int(check_result.indicator)
        if outlier_indicator == interval_indicator[1]:
            b = c
        else:
            a = c

        if i == number_of_iterations - 1:
            logging.debug(
                f"Max iterations reached without finding hidden outlier, "
                f"returning c={c}, type={check_result.name}"
            )

    # check_result is guaranteed to be set after at least one iteration
    assert check_result is not None, "No iterations were performed"
    return c, check_result


def parallel_routine_generate_point(
    iteration: int,
    interval_length: float,
    check_fast: bool,
    fixed_interval_length: bool,
    origin: NDArray[np.floating[Any]],
    full_space: tuple[int, ...],
    fitted_subspaces: dict[tuple[int, ...], OutlierDetectionMethod],
    seed: int,
    origin_method_instance: origin_method.OriginMethod,
    verbose: bool,
) -> NDArray[Any]:
    """Generate a single point using the bisection algorithm (for parallel execution).

    This function is designed to be called via joblib Parallel. It generates
    a random direction, optionally updates the origin, and runs bisection
    to find a potential hidden outlier.

    Args:
        iteration: Current iteration number (for logging).
        interval_length: Search distance from origin.
        check_fast: Use fast outlier checking.
        fixed_interval_length: Keep interval length fixed.
        origin: Base origin point.
        full_space: Tuple of all dimension indices.
        fitted_subspaces: Dictionary of fitted models.
        seed: Random seed for direction generation.
        origin_method_instance: Strategy for calculating origin.
        verbose: Enable verbose logging.

    Returns:
        Array containing the generated point coordinates followed by the
        result type name (e.g., "H1", "H2", "OB", "IL").
    """
    dims = len(full_space)

    # For random/weighted origins, recalculate origin each iteration
    if origin_method_instance.class_type in (
        origin_method.OriginType.RANDOM,
        origin_method.OriginType.WEIGHTED,
    ):
        origin = origin_method_instance.calculate_origin()

    # Generate random direction on unit sphere
    direction = utils.random_unif_on_sphere(2, dims, 1, seed)[0]

    # Run bisection
    bisection_results = bisect(
        interval_length=interval_length,
        direction=direction,
        is_check_fast=check_fast,
        is_fixed_interval_length=fixed_interval_length,
        origin=origin,
        full_space=full_space,
        fitted_subspaces=fitted_subspaces,
        is_verbose=verbose,
    )
    hidden_c, outlier_type = bisection_results

    # Package result: point coordinates + type label
    if outlier_type in (OutlierResultType.H1, OutlierResultType.H2):
        result_point = hidden_c * direction + origin
        result = np.append(result_point, outlier_type.name)
    else:
        # Non-hidden points stored as zeros (filtered out later)
        result_point = np.zeros((1, dims))
        result = np.append(result_point, outlier_type.name)

    if iteration % 100 == 0:
        logging.info(f"Progress: {iteration} points generated")

    return result


class BisectHOGen:
    """Hidden Outlier Generator using the bisection algorithm.

    This class generates synthetic hidden outliers by:
    1. Fitting outlier detection models on all relevant subspaces
    2. Repeatedly searching from an origin in random directions
    3. Using bisection to find boundary points
    4. Collecting points classified as H1 or H2 (hidden outliers)

    Hidden outliers are points that exhibit different outlier behavior
    when viewed in different feature subspaces:
        - H1: Outlier in some subspace but not the full space
        - H2: Outlier in full space but not in any subspace

    Attributes:
        data: The input dataset used for fitting models.
        dims: Number of features in the data.
        outlier_detection_method: PyOD model class used for detection.
        seed: Random seed for reproducibility.
        max_dimensions: Threshold for switching to random subspace sampling.
        full_space: Tuple of all dimension indices.
        hidden_x_list: Generated hidden outlier points (after fit_generate).
        hidden_x_type: Classification of each generated point.
        exec_time: Execution time of the last fit_generate call.

    Example:
        >>> from hog_bisect import BisectHOGen
        >>> from pyod.models.lof import LOF
        >>> import numpy as np
        >>>
        >>> data = np.random.randn(100, 5)
        >>> generator = BisectHOGen(data, outlier_detection_method=LOF, seed=42)
        >>> outliers = generator.fit_generate(gen_points=50)
        >>> print(f"Generated {len(outliers)} hidden outliers")
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        outlier_detection_method: type[BaseDetector] = pyod.models.lof.LOF,
        seed: int = DEFAULT_SEED,
        max_dimensions: int = DEFAULT_MAX_DIMENSIONS,
    ) -> None:
        """Initialize the hidden outlier generator.

        Args:
            data: Dataset to analyze, shape (n_samples, n_features).
            outlier_detection_method: PyOD detector class to use. Any class
                from pyod.models that inherits from BaseDetector.
                Defaults to Local Outlier Factor (LOF).
            seed: Random seed for reproducible results.
            max_dimensions: If data has >= this many dimensions, use random
                subspace sampling instead of full enumeration. This prevents
                exponential blowup in high dimensions.

        Raises:
            ValueError: If data is not a valid 2D array with sufficient samples.
            TypeError: If outlier_detection_method is not a valid PyOD detector class.
        """
        # Input validation
        if not isinstance(data, np.ndarray):
            raise TypeError(f"data must be a numpy array, got {type(data).__name__}")
        if data.ndim != 2:
            raise ValueError(f"data must be 2-dimensional, got {data.ndim} dimensions")
        if data.shape[0] < 2:
            raise ValueError(f"data must have at least 2 samples, got {data.shape[0]}")
        if data.shape[1] < 1:
            raise ValueError(f"data must have at least 1 feature, got {data.shape[1]}")

        if not isinstance(outlier_detection_method, type):
            raise TypeError(
                f"outlier_detection_method must be a class, got {type(outlier_detection_method).__name__}"
            )
        if not issubclass(outlier_detection_method, BaseDetector):
            raise TypeError(
                f"outlier_detection_method must be a PyOD BaseDetector subclass, "
                f"got {outlier_detection_method.__name__}"
            )

        if not isinstance(seed, int) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        if not isinstance(max_dimensions, int) or max_dimensions < 1:
            raise ValueError(f"max_dimensions must be a positive integer, got {max_dimensions}")

        np.random.seed(seed)
        self.hidden_x_type: NDArray[Any] | None = None
        self.hidden_x_list: NDArray[np.floating[Any]] | None = None
        self.start_time: float | None = None
        self.tempdir: str | None = None
        self.seed = seed
        self.data = data
        self.dims = self.data.shape[1]
        self.outlier_detection_method = outlier_detection_method
        self.fitted_subspaces_dict: dict[tuple[int, ...], OutlierDetectionMethod] | None = None
        self.outlier_indices: NDArray[np.floating[Any]] | None = None
        self.full_space = tuple(range(self.data.shape[1]))
        self.exec_time: float | None = None
        self.max_dimensions = max_dimensions

    def _initialize_fit_generate(
        self,
        n_jobs: int,
        get_origin_type: str,
    ) -> tuple[float, NDArray[np.floating[Any]], origin_method.OriginMethod]:
        """Set up models and compute initial parameters for generation.

        Fits outlier detection models on all subspaces, identifies existing
        outliers, and calculates the search parameters.

        Args:
            n_jobs: Number of parallel jobs for fitting.
            get_origin_type: Origin calculation strategy name.

        Returns:
            Tuple of (search_length, initial_origin, origin_method_instance).
        """
        self.start_time = time.time()

        # tempdir is set by fit_generate before calling this method
        assert self.tempdir is not None, "tempdir must be set before initialization"

        # Fit models on all relevant subspaces
        self.fitted_subspaces_dict = fit_in_all_subspaces(
            self.outlier_detection_method,
            self.data,
            seed=self.seed,
            tempdir=self.tempdir,
            subspace_limit=self.max_dimensions,
            n_jobs=n_jobs,
        )

        # Identify which points in the data are outliers
        self.outlier_indices = self._get_outlier_indices()

        # Calculate search length based on data extent
        length = float(np.max(np.sqrt(np.sum(self.data**2, axis=1))))
        logging.debug(f"Search length: {length}")

        # Set up origin calculation strategy
        origin_strategy = origin_method.get_origin(self.data, self.outlier_indices, get_origin_type)
        origin = origin_strategy.calculate_origin()
        logging.debug(f"Initial origin: {origin}")

        return length, origin, origin_strategy

    def _execute_parallel_routine(
        self,
        gen_points: int,
        length: float,
        origin: NDArray[np.floating[Any]],
        n_jobs: int,
        check_fast: bool,
        is_fixed_interval_length: bool,
        verbose: bool,
        origin_strategy: origin_method.OriginMethod,
    ) -> NDArray[Any]:
        """Generate points in parallel using joblib.

        Args:
            gen_points: Number of points to attempt generating.
            length: Search distance from origin.
            origin: Initial origin point.
            n_jobs: Number of parallel workers.
            check_fast: Use fast outlier checking.
            is_fixed_interval_length: Keep interval length fixed.
            verbose: Enable verbose logging.
            origin_strategy: Strategy for calculating origin.

        Returns:
            Array of results, each row contains point coordinates and type label.
        """
        with Parallel(n_jobs=n_jobs, timeout=FITTING_TIMEOUT_TIME) as parallel:
            return np.array(
                parallel(
                    delayed(parallel_routine_generate_point)(
                        i,
                        length,
                        check_fast,
                        is_fixed_interval_length,
                        origin,
                        self.full_space,
                        self.fitted_subspaces_dict,
                        self.seed,
                        origin_strategy,
                        verbose,
                    )
                    for i in range(gen_points)
                )
            )

    def _post_process_results(
        self,
        bisection_results: NDArray[Any],
    ) -> NDArray[np.floating[Any]]:
        """Filter and store results, keeping only hidden outliers.

        Args:
            bisection_results: Raw results from parallel generation.

        Returns:
            Array of hidden outlier points (non-zero rows only).
        """
        # Separate coordinates from type labels
        hidden_x_type = bisection_results[:, -1].reshape(-1, 1)
        hidden_x_list = bisection_results[:, :-1].astype(float)

        # Filter out non-hidden outliers (stored as zero vectors)
        hidden_x_list = hidden_x_list[np.sum(hidden_x_list, axis=1) != 0, :]

        # Cleanup temporary directory
        if self.tempdir and os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

        # start_time is set at the beginning of _initialize_fit_generate
        assert self.start_time is not None, "start_time must be set before post-processing"
        self.exec_time = time.time() - self.start_time
        self.hidden_x_list = hidden_x_list
        self.hidden_x_type = hidden_x_type

        return hidden_x_list

    def _get_outlier_indices(self) -> NDArray[np.floating[Any]]:
        """Identify which data points are outliers in the full space.

        Returns:
            Binary array where 1 indicates outlier, 0 indicates inlier.

        Raises:
            ValueError: If models haven't been fitted yet.
        """
        if self.fitted_subspaces_dict is None:
            raise ValueError("Models not fitted. Call fit_generate first.")

        n_rows = self.data.shape[0]
        outlier_indices = np.zeros(n_rows)

        for i in range(n_rows):
            outlier_indices[i] = inference(
                self.data[i, :], self.full_space, self.fitted_subspaces_dict
            )

        return outlier_indices

    def fit_generate(
        self,
        gen_points: int = 100,
        check_fast: bool = True,
        is_fixed_interval_length: bool = True,
        get_origin_type: str = "weighted",
        verbose: bool = False,
        n_jobs: int = 1,
    ) -> NDArray[np.floating[Any]]:
        """Generate hidden outliers using the bisection algorithm.

        This is the main entry point. It fits models, then generates the
        requested number of candidate points, filtering to keep only those
        classified as hidden outliers (H1 or H2).

        Args:
            gen_points: Number of candidate points to generate. The actual
                number of hidden outliers returned may be less.
            check_fast: If True, stop checking subspaces after finding first
                outlier. Faster but may affect classification accuracy.
            is_fixed_interval_length: If True, use consistent search distance.
                If False, add random perturbation to interval length.
            get_origin_type: Strategy for choosing the search origin:
                - "centroid": Data mean (stable, may miss asymmetric outliers)
                - "least outlier": Most normal point (stable)
                - "random": Random inlier each iteration (diverse)
                - "weighted": Weighted random toward normal points (recommended)
            verbose: Enable debug logging.
            n_jobs: Number of parallel workers. Use -1 for all CPU cores.

        Returns:
            Array of hidden outlier points, shape (n_hidden, n_features).
            May be empty if no hidden outliers were found.

        Raises:
            ValueError: If gen_points is not positive or get_origin_type is invalid.
        """
        # Input validation
        if not isinstance(gen_points, int) or gen_points < 1:
            raise ValueError(f"gen_points must be a positive integer, got {gen_points}")

        valid_origin_types = {"centroid", "least outlier", "random", "weighted"}
        if get_origin_type not in valid_origin_types:
            raise ValueError(
                f"get_origin_type must be one of {valid_origin_types}, got '{get_origin_type}'"
            )

        if not isinstance(n_jobs, int):
            raise ValueError(f"n_jobs must be an integer, got {type(n_jobs).__name__}")

        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.info(
            f"Starting generation: points={gen_points}, fast={check_fast}, "
            f"fixed_length={is_fixed_interval_length}, origin={get_origin_type}, n_jobs={n_jobs}"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            logging.debug(f"Using temporary directory: {temp_dir}")
            self.tempdir = temp_dir

            length, origin, origin_strategy = self._initialize_fit_generate(n_jobs, get_origin_type)

            logging.info(f"Generating {gen_points} candidate points...")
            bisection_results = self._execute_parallel_routine(
                gen_points,
                length,
                origin,
                n_jobs,
                check_fast,
                is_fixed_interval_length,
                verbose,
                origin_strategy,
            )
            hidden_x_list = self._post_process_results(bisection_results)

            logging.info(
                f"Complete. Found {len(hidden_x_list)} hidden outliers in {self.exec_time:.2f}s"
            )

        return hidden_x_list

    def print_summary(self) -> None:
        """Print a summary of the generation results.

        Displays statistics about the data, generated points, and breakdown
        by hidden outlier type (H1 vs H2).
        """
        if self.hidden_x_list is None or self.hidden_x_type is None:
            print("No generation results available. Run fit_generate first.")
            return

        db_cols = self.data.shape[1]
        db_rows = self.data.shape[0]
        ho_rows = len(self.hidden_x_type)

        ho_hidden = len(self.hidden_x_list) if isinstance(self.hidden_x_list, np.ndarray) else 0

        h1_count = np.sum(self.hidden_x_type == "H1")
        h2_count = np.sum(self.hidden_x_type == "H2")

        summary = f"""
Hidden Outlier Generation Summary
=================================
Detection method: {self.outlier_detection_method.__name__}
Generation method: bisect

Dataset:
  - Features: {db_cols}
  - Samples: {db_rows}

Results:
  - Candidates generated: {ho_rows}
  - Hidden outliers found: {ho_hidden}
    - H1 (subspace outliers): {h1_count}
    - H2 (full-space outliers): {h2_count}

Execution time: {self.exec_time:.2f}s
"""
        print(summary)

    def save_to_csv(self, file_name: str, include_type: bool = False) -> None:
        """Save generated hidden outliers to a CSV file.

        Args:
            file_name: Path to the output CSV file.
            include_type: If True, include the outlier type (H1/H2) as the
                last column.
        """
        if self.hidden_x_list is None or self.hidden_x_type is None:
            raise ValueError("No results to save. Run fit_generate first.")

        if not include_type:
            np.savetxt(file_name, self.hidden_x_list, delimiter=",")
        else:
            x_and_type = np.hstack((self.hidden_x_list, self.hidden_x_type))
            np.savetxt(file_name, x_and_type, delimiter=",", fmt="%s")
