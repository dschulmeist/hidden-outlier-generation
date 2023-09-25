import logging
import os
import shutil
import tempfile
import time
from datetime import datetime

import numpy as np
import pyod.models.lof
from joblib import Parallel, delayed
from numpy import ndarray

from hog_bisect import origin_method
from hog_bisect import utils
from hog_bisect.outlier_detection_method import get_outlier_detection_method
from hog_bisect.outlier_result_type import OutlierResultType
from hog_bisect.utils import subspace_grab, fit_in_all_subspaces

DEFAULT_MAX_DIMENSIONS = 11
FITTING_TIMEOUT_TIME = 60
DEFAULT_SEED = 5
DEFAULT_NUMBER_OF_ITERATIONS = 30
DEFAULT_NUMBER_OF_GEN_POINTS = 100
DEFAULT_C = 0.5
DEFAULT_NUMBER_OF_PARTS = 20

#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def outlier_check(x: ndarray, full_space: tuple, fitted_subspaces: dict, verb=False, fast=True) -> OutlierResultType:
    """
    Check if a point is an outlier in any subspace.

    Args:
        x (ndarray): The point to check.
        verb (bool, optional): Verbose flag. Defaults to False.
        fast (bool, optional): Fast check flag. Defaults to True.
        full_space (tuple): The full space.
        fitted_subspaces (dict): Dictionary of fitted models.

    Returns:
        OutlierResultType: The type of the point .
    """
    subspaces = fitted_subspaces.keys()
    index = np.zeros(len(subspaces))
    isOutlierInFullSpace = False
    isOutlierInSubspace = False

    if inference(x, full_space, fitted_subspaces=fitted_subspaces):
        isOutlierInFullSpace = True
    for j, subspace in enumerate(subspaces):
        if subspace == full_space or (len(subspace) == len(full_space)):
            continue
        if inference(subspace_grab(subspace, x.reshape(1, x.shape[0])), subspace, fitted_subspaces=fitted_subspaces):

            index[j] = 1
            if not isOutlierInFullSpace:
                # logging.debug(f"found x, outlier in {subspace} but not in full space")
                pass
            if fast:
                break

    if np.sum(index[:-1]) > 0:
        isOutlierInSubspace = True

    if isOutlierInSubspace and (not isOutlierInFullSpace):
        # logging.debug("found x in the hidden outlier area H1")
        result = OutlierResultType.H1
    elif (not isOutlierInSubspace) and isOutlierInFullSpace:
        logging.debug("found x in the hidden outlier area H2")
        result = OutlierResultType.H2
    elif isOutlierInSubspace and isOutlierInFullSpace:
        # if verb:
        #     logging.debug("x Outside of bounds")
        result = OutlierResultType.OB
    else:
        # if verb:
        #     logging.debug("x in the total acceptance area")
        result = OutlierResultType.IL

    return result


def validate_subspace(subspace, fitted_subspaces):
    if not isinstance(subspace, tuple):
        raise ValueError(f"Subspace needs to be a tuple, got {type(subspace)}")
    if subspace not in fitted_subspaces:
        raise ValueError(f"Subspace {subspace} not in the dictionary of fitted subspaces")


def inference(x, subspace, fitted_subspaces) -> bool:
    """
    Infer whether a given data point is an outlier in the specified subspace.

    Args:
        x (np.array): Data point to check.
        subspace (tuple): The subspace to check for the outlier.
        fitted_subspaces (dict): Dictionary containing models fitted to different subspaces.

    Returns:
        bool: True if the data point is an outlier, False otherwise.
    """
    validate_subspace(subspace, fitted_subspaces)
    return bool(fitted_subspaces[subspace].predict(x))


def get_segmentation_points(length, parts):
    return np.linspace(0, length, num=parts)


def interval_check(length, direction, origin, full_space, fitted_subspaces, parts=DEFAULT_NUMBER_OF_PARTS):
    segmentation_points = get_segmentation_points(length, parts)
    check_if_interval_is_outlier = np.full(len(segmentation_points), -1)

    for i, c in enumerate(segmentation_points):
        point_to_check = c * direction + origin
        check_if_interval_is_outlier[i] = 1 if inference(point_to_check, full_space, fitted_subspaces) else -1

    intervals = construct_intervals(segmentation_points, check_if_interval_is_outlier)

    if not intervals:
        if check_if_interval_is_outlier[0] == 1:
            logging.debug("No sub-intervals found, returning the whole interval")
            return [(segmentation_points[0], segmentation_points[-1]),
                    (check_if_interval_is_outlier[0], check_if_interval_is_outlier[-1])]
        else:
            logging.debug("No sub-intervals found, increasing interval length")
        return interval_check(length * 2, direction, origin, full_space, fitted_subspaces, parts=parts)

    return intervals


def construct_intervals(segmentation_points, check_if_interval_is_outlier):
    intervals = []
    previous = check_if_interval_is_outlier[0]

    for i in range(1, len(check_if_interval_is_outlier)):
        if check_if_interval_is_outlier[i] != previous:
            intervals.append([(segmentation_points[i - 1], segmentation_points[i]),
                              (check_if_interval_is_outlier[i - 1], check_if_interval_is_outlier[i])])
        previous = check_if_interval_is_outlier[i]

    # if not intervals:
    #     return []
    #     logging.debug("No sub-intervals found, returning the whole interval")
    #     intervals.append([(segmentation_points[0], segmentation_points[-1]),
    #                       (check_if_interval_is_outlier[0], check_if_interval_is_outlier[-1])])

    return intervals


def bisect(direction, interval_length, origin, number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS,
           is_check_fast=True,
           is_fixed_interval_length=True, full_space=None, fitted_subspaces=None, is_verbose=True) -> \
        (float, OutlierResultType):
    """
    Bisection algorithm function

    Performs the bisect algorithm to any given
    interval_length and direction. It outputs the value c in which the function f
    finds a 0 of the form: f(c*x + origin).

    Arguments:
    direction (ndarray): Directional vector
    interval_length: Length of the original interval
    origin: initial origin
    number_of_iterations: Number of iterations to perform
    """
    check_result = None
    c = DEFAULT_C
    if not is_fixed_interval_length:
        interval_length = interval_length + np.random.uniform(-interval_length / 2, interval_length)
    found_sub_intervals = interval_check(interval_length, direction, origin, full_space=full_space,
                                         fitted_subspaces=fitted_subspaces)
    choice = np.random.choice(len(found_sub_intervals), 1)[0,]
    chosen_interval = found_sub_intervals[choice]
    interval_indicator = chosen_interval[1]
    interval = chosen_interval[0]

    a = interval[0]
    b = interval[1]
    for i in range(number_of_iterations):
        c = (b + a) / 2
        check_result = outlier_check(c * direction + origin,
                                     fast=is_check_fast,
                                     full_space=full_space,
                                     fitted_subspaces=fitted_subspaces,
                                     verb=is_verbose)

        outlier_indicator = int(check_result.indicator)
        outlier_type = str(check_result.name)

        if (check_result == OutlierResultType.H1) or (check_result == OutlierResultType.H2):
            return c, check_result
        if outlier_indicator == interval_indicator[1]:
            b = c
        else:
            a = c
        if i == number_of_iterations - 1:
            logging.debug(f"reached max number of iterations and found no H1 or H2, returning {c} and {outlier_type}")
    return c, check_result


# inelegant solution, but had problems with doing this as a class method
def parallel_routine_generate_point(i, interval_length,
                                    check_fast, fixed_interval_length, origin,
                                    full_space, odm_dict, seed, om: origin_method.OriginMethod, verbose):
    dims = len(full_space)

    if om.class_type in [origin_method.OriginType.RANDOM, origin_method.OriginType.WEIGHTED]:
        origin = om.calculate_origin()

    direction = utils.random_unif_on_sphere(2, dims, 1, seed)[0,]
    # print(f"direction: {direction}")
    bisection_results = bisect(interval_length=interval_length, direction=direction,
                               is_check_fast=check_fast,
                               is_fixed_interval_length=fixed_interval_length,
                               origin=origin, full_space=full_space, fitted_subspaces=odm_dict,
                               is_verbose=verbose)
    hidden_c, outlier_type = bisection_results

    if outlier_type in [OutlierResultType.H1, OutlierResultType.H2]:
        result_point = hidden_c * direction + origin
        result = np.append(result_point, outlier_type.name)
    else:
        result_point = np.zeros((1, dims))
        result = np.append(result_point, outlier_type.name)
    if i % 100 == 0:
        #print(f"new origin: {origin}")
        logging.info(f"Progress: {i} points generated")
        # logging.debug(f" current result: {result}")
    return result


class BisectHOGen:
    """
    BisectHOGen is a class that generates synthetic hidden outliers.

    Attributes:
        data (np.array): The dataset to analyze.
        outlier_detection_method (callable): Method used for outlier detection.
        seed (int, optional): Seed for random number generation. Default is DEFAULT_SEED.

    """

    def __init__(self, data, outlier_detection_method=pyod.models.lof.LOF, seed=DEFAULT_SEED
                 , max_dimensions=DEFAULT_MAX_DIMENSIONS):
        """
        Initialize the HOGen class.

        Args:
            data (np.array): The dataset to analyze.
            outlier_detection_method (OutlierDetectionMethod): Method used for outlier detection.
             Default is LOF. Every Outlier Detection Method from pyod.models can be used.
            seed (int): Seed for random state. Default is DEFAULT_SEED.
        """
        np.random.seed(seed)
        self.hidden_x_type = None
        self.hidden_x_list = None
        self.start_time = None
        self.tempdir = None
        self.x_list = None
        self.seed = seed
        self.data = data
        self.dims = self.data.shape[1]
        self.outlier_detection_method = outlier_detection_method
        self.fitted_subspaces_dict = None
        self.outlier_indices = None
        self.full_space = tuple(np.arange(self.data.shape[1]))
        self.exec_time = None
        self.max_dimensions = max_dimensions

    def _initialize_fit_generate(self, n_jobs: int, get_origin_type: str):
        self.start_time = time.time()
        self.fitted_subspaces_dict = fit_in_all_subspaces(self.outlier_detection_method, self.data, seed=self.seed,
                                                          tempdir=self.tempdir, subspace_limit=self.max_dimensions,
                                                          n_jobs=n_jobs)
        self.outlier_indices = self._get_outlier_indices()
        length = np.max(np.sqrt(np.sum(self.data ** 2, axis=1)))
        logging.debug(f"Length: {length}")
        originmethod = origin_method.get_origin(self.data, self.outlier_indices, get_origin_type)
        origin = originmethod.calculate_origin()
        logging.debug(f"Origin: {origin}")
        return length, origin, originmethod

    def _execute_parallel_routine(self, gen_points: int, length: float, origin: float, n_jobs: int, check_fast: bool,
                                  is_fixed_interval_length: bool, verbose: bool,
                                  originmethod: origin_method.OriginMethod):
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
                        originmethod,
                        verbose
                    )
                    for i in range(gen_points)))

    def _post_process_results(self, bisection_results):
        hidden_x_type = bisection_results[:, -1].reshape(-1, 1)
        hidden_x_list = bisection_results[:, :-1].astype(float)
        hidden_x_list = hidden_x_list[np.sum(hidden_x_list, axis=1) != 0, :]
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)
        self.exec_time = time.time() - self.start_time
        self.hidden_x_list = hidden_x_list
        self.hidden_x_type = hidden_x_type
        return hidden_x_list

    def _get_outlier_indices(self) -> ndarray:
        """
        Obtain the indices of outliers in the data.

        Returns:
            np.array: An array containing indices of the outliers in the full data.
        """
        if self.fitted_subspaces_dict is None:
            raise ValueError("fitted_subspaces_dict is None. Please run fit_in_all_subspaces first.")
        # Calculate the number of rows and columns
        n_rows, n_cols = self.data.shape

        # Initialize the output array
        outlier_indices = np.zeros(n_rows)

        # Iterate over each row
        for i in range(n_rows):
            # Apply the inference method to the row with the tuple of column indices
            outlier_indices[i] = inference(self.data[i, :], self.full_space, self.fitted_subspaces_dict)

        return outlier_indices

    def fit_generate(self, gen_points: int = 100, check_fast: bool = True,
                     is_fixed_interval_length: bool = True, get_origin_type: str = "weighted", verbose: bool = False,
                     n_jobs: int = 1) -> ndarray:
        """
        Main function to perform bisection algorithm for generating synthetic hidden outliers in parallel.

        Args:
            gen_points (int): Number of points to generate.
            check_fast (bool): If True, terminates as soon as one outlier is found.
            is_fixed_interval_length (bool): If True, keeps the interval length fixed.
            get_origin_type (str): Method to determine the origin of the new points. One of:
                                    "random", "weighted", "least outlier", "centroid".
            verbose (bool): activate verbose mode
            n_jobs: number of jobs to run in parallel

        Returns:
            np.array: Array containing generated hidden outliers.

        """
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logging.basicConfig(level=level,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Starting with parameters: gen_points={gen_points}, check_fast={check_fast}, "
                     f"fixed_interval_length={is_fixed_interval_length}, get_origin_type={get_origin_type}, "
                     f"verbose={verbose}, n_jobs={n_jobs}")

        with tempfile.TemporaryDirectory() as temp_dir:
            logging.debug(f"Created temporary directory {temp_dir}")
            self.tempdir = temp_dir  # Set the temporary directory
            length, origin, originmethod = self._initialize_fit_generate(n_jobs, get_origin_type)

            logging.info("Generating {} hidden outlier points... ".format(gen_points))
            bisection_results = self._execute_parallel_routine(
                gen_points, length, origin, n_jobs, check_fast, is_fixed_interval_length, verbose, originmethod)
            hidden_x_list = self._post_process_results(bisection_results)

            logging.info("Done! Exec time: {}".format(self.exec_time))
        return hidden_x_list

    def print_summary(self):
        """
        Print a summary of the Hidden Outlier Generation, including data and outlier details.

        """
        db_cols = self.data.shape[1]
        db_rows = self.data.shape[0]
        ho_rows = len(self.hidden_x_type)
        ho_hidden = 1 if isinstance(self.hidden_x_list, list) and len(self.hidden_x_list) != 0 else (
            len(self.hidden_x_list) if isinstance(self.hidden_x_list, ndarray) else 0
        )
        h1_outliers = self.hidden_x_type[np.where(self.hidden_x_type == 'H1')].shape[0]
        h2_outliers = self.hidden_x_type[np.where(self.hidden_x_type == 'H2')].shape[0]

        print('Hidden Outlier Generation Method Object' + '\n\n' +
              'Outlier detection method used: ' + get_outlier_detection_method(self
                                                                               .outlier_detection_method).name + '\n' +
              'Synthetic HO generation method employed: ' + "bisect" + '.\n\n' +
              'Database summary' + ':\n\n' +
              '* ' + 'Number of features: ' + str(db_cols) + '\n' +
              '* ' + 'Total number of data points: ' + str(db_rows) + '\n' +
              '* ' + 'Total amount of synthetic data generated: ' + str(ho_rows) + '\n' +
              '\t' + '...of which hidden outliers: ' + str(ho_hidden) + '\n' +
              '* ' + 'Number of H1 outliers: ' + str(h1_outliers) + '\n' +
              '* ' + 'Number of H2 outliers: ' + str(h2_outliers) + '.\n\n' +
              'Total execution time: ' + str(self.exec_time) + '.')

    def save_to_csv(self, file_name, include_type=False):
        """
        Description: This function saves the generated hidden outliers to a CSV file.

        Args:
            file_name (str): The name of the CSV file to save.
            include_type (bool): If True, the type of the hidden outlier is included in the CSV file.
        Returns:
            None
        """
        if not include_type:
            np.savetxt(file_name, self.hidden_x_list, delimiter=",")
        else:
            x_and_type = np.hstack((self.hidden_x_list, self.hidden_x_type))
            np.savetxt(file_name, x_and_type, delimiter=",")
