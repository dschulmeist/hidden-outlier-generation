import logging
import os
import shutil
import time
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from numpy import ndarray

from multibisect import Utils, OriginMethod
from multibisect.OutlierDetectionMethod import OdLOF, OutlierDetectionMethod
from multibisect.ResultType import ResultType
from multibisect.Utils import subspace_grab, fit_in_all_subspaces

n_jobs = -2
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SUBSPACE_LIMIT = 12
FITTING_TIMEOUT_TIME = 60
DEFAULT_SEED = 5
DEFAULT_NUMBER_OF_ITERATIONS = 30
DEFAULT_NUMBER_OF_GEN_POINTS = 100
DEFAULT_C = 0.5


def outlier_check(x, full_space, fitted_subspaces, verb=False, fast=True) -> ResultType:
    """
    Check if a point is an outlier in any subspace.

    Args:
        x (ndarray): The point to check.
        verb (bool, optional): Verbose flag. Defaults to False.
        fast (bool, optional): Fast check flag. Defaults to True.
        full_space (tuple, optional): The full space.
        fitted_subspaces (dict, optional): Dictionary of fitted models.

    Returns:
        ResultType: The type of the point .
    """
    subspaces = fitted_subspaces.keys()
    index = np.zeros(len(subspaces))
    isOutlierInFullSpace = False
    isOutlierInSubspace = False

    if inference(x, full_space, fitted_subspaces=fitted_subspaces):
        isOutlierInFullSpace = True
    for j, subspace in enumerate(subspaces):
        if subspace == full_space:
            continue
        if inference(subspace_grab(subspace, x.reshape(1, x.shape[0])), subspace, fitted_subspaces=fitted_subspaces):
            index[j] = 1

            if fast:  # remove
                break

    if np.sum(index[:-1]) > 0:
        isOutlierInSubspace = True

    if isOutlierInSubspace and (not isOutlierInFullSpace):
        result = ResultType.H1
    elif (not isOutlierInSubspace) and isOutlierInFullSpace:
        result = ResultType.H2
    elif isOutlierInSubspace and isOutlierInFullSpace:
        if verb:
            logging.info("x Outside of bounds")
        result = ResultType.OB
    else:
        if verb:
            logging.info("x in the total acceptance area")
        result = ResultType.IL

    return result


def inference(x, subspace, fitted_subspaces=None) -> bool:
    """
    Infer whether a given data point is an outlier in the specified subspace.

    Args:
        x (np.array): Data point to check.
        subspace (tuple): The subspace to check for the outlier.
        fitted_subspaces (dict): Dictionary containing models fitted to different subspaces.

    Returns:
        bool: True if the data point is an outlier, False otherwise.
    """
    if not isinstance(subspace, tuple) :
        print("Subspace: ", subspace, type(subspace), " is not a tuple")
        raise ValueError(
            "Error in Code, subspace needs to be a tuple")

    if subspace not in fitted_subspaces.keys():
        logging.info("type of subsp: ", type(subspace), " subspace: ", list(subspace))
        raise ValueError(f"Subspace {subspace} not in the dictionary of fitted subspaces")
    return bool(fitted_subspaces[subspace].predict(x))


def interval_check(length, direction, origin, full_space, fitted_subspaces, parts=5):
    """
    Interval Refining function

    Breaks the interval in the selected number of parts
    (defaults to 5) and checks if each part is an Outlier or an Inlier
    in the global space. After that it stores each sub-interval
    in a list

    Arguments:
    length: Length of the original interval
    method: ODM
    direction: Directional vector selected
    origin: origin
    parts: Number of parts to which cut the interval
    fitted_subspaces: a dictionary containing fitted Subspaces (OutlierDetectionMethod)
    """

    segmentation_points = np.linspace(0, length, num=parts)
    check = np.full(len(segmentation_points), -1)
    # print("checking intervals in the global space")
    for i, c in enumerate(segmentation_points):

        if inference(c * direction + origin, subspace=full_space, fitted_subspaces=fitted_subspaces):
            check[i] = 1

    intervals = []
    previous = check[0]

    for i in range(len(check)):
        if check[i] != previous:
            intervals.append([(segmentation_points[i - 1], segmentation_points[i]), (check[i - 1], check[i])])
        previous = check[i]

    if len(intervals) == 0:
        intervals.append([(segmentation_points[0], segmentation_points[-1]), (check[-1], check[-1])])

    return intervals


def multi_bisect(direction, interval_length, origin, number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS,
                 is_check_fast=True,
                 fixed_interval_length=True, full_space=None, fitted_subspaces=None, is_verbose=True) -> \
        (float, ResultType):
    """
    Multi Bisection algorithm function

    Performs the multi bisection algorithm to any given
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
    if not fixed_interval_length:
        interval_length = interval_length + np.random.uniform(-interval_length / 2, interval_length)
    interval = interval_check(interval_length, direction, origin, full_space=full_space,
                              fitted_subspaces=fitted_subspaces)
    choice = np.random.choice(len(interval), 1)[0,]
    interval = interval[choice]
    interval_indicator = interval[1]
    interval = interval[0]

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

        if (check_result == ResultType.H1) or (check_result == ResultType.H2):
            return c, check_result
        if outlier_indicator == interval_indicator[1]:
            b = c
        else:
            a = c
    return c, check_result


# inelegant solution, but had problems with doing this as a class method
def parallel_routine_generate_point(i, interval_length,
                                    check_fast, fixed_interval_length, origin,
                                    full_space, odm_dict,
                                    get_origin_type, seed, om, verbose):
    dims = len(full_space)
    if verbose and i % 500 == 0:
        logging.info(f"Processing point {i}")
    if get_origin_type in ["random", "weighted"]:
        origin = om.calculate_origin()

    direction = Utils.random_unif_on_sphere(2, dims, 1, seed)[0,]
    bisection_results = multi_bisect(interval_length=interval_length, direction=direction,
                                     is_check_fast=check_fast,
                                     fixed_interval_length=fixed_interval_length,
                                     origin=origin, full_space=full_space, fitted_subspaces=odm_dict,
                                     is_verbose=verbose)
    hidden_c, outlier_type = bisection_results

    if outlier_type in [ResultType.H1, ResultType.H2]:
        result_point = hidden_c * direction + origin
        result = np.append(result_point, outlier_type.name)
    else:
        result_point = np.zeros((1, dims))
        result = np.append(result_point, outlier_type.name)

    return result


class MultiBisectHOGen:
    """
    MultiBisectHOGen is a class that generates synthetic hidden outliers.

    Attributes:
        data (np.array): The dataset to analyze.
        outlier_detection_method (callable): Method used for outlier detection.
        seed (int, optional): Seed for random number generation. Default is DEFAULT_SEED.

    """

    def __init__(self, data, outlier_detection_method=OdLOF, seed=DEFAULT_SEED):
        """
        Initialize the HOGen class.

        Args:
            data (np.array): The dataset to analyze.
            outlier_detection_method (OutlierDetectionMethod): Method used for outlier detection. Default is LOF
            seed (int): Seed for random state. Default is DEFAULT_SEED.
        """
        np.random.seed(seed)
        self.hidden_x_type = None
        self.hidden_x_list = None
        self.start_time = time.time()
        self.tempFName = f'HOGTemp{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}'
        self.x_list = None
        self.seed = seed
        self.data = data
        self.dims = self.data.shape[1]
        self.outlier_detection_method = outlier_detection_method
        self.fitted_subspaces_dict = None
        self.outlier_indices = None
        self.full_space = tuple(np.arange(self.data.shape[1]))
        self.exec_time = None

    def get_outlier_indices(self):
        """
        Obtain the indices of outliers in the data.

        Returns:
            np.array: An array containing indices of the outliers in the full data.
        """
        if self.fitted_subspaces_dict is None:
            pass
            # raise error
        # Calculate the number of rows and columns
        n_rows, n_cols = self.data.shape

        # Initialize the output array
        outlier_indices = np.zeros(n_rows)

        # Iterate over each row
        for i in range(n_rows):
            # Apply the inference method to the row with the tuple of column indices
            outlier_indices[i] = inference(self.data[i, :], self.full_space, self.fitted_subspaces_dict)

        return outlier_indices

    def main_multi_bisect(self, gen_points=100, check_fast=True,
                          fixed_interval_length=True, get_origin_type="weighted", verbose=True) -> ndarray:
        """
        Main function to perform multi-bisection algorithm for generating synthetic hidden outliers.

        Args:
            gen_points (int): Number of points to generate.
            check_fast (bool): If True, terminates as soon as one outlier is found.
            fixed_interval_length (bool): If True, keeps the interval length fixed.
            get_origin_type (str): Method to determine the origin of the new points.
            verbose (bool): activate verbose mode

        Returns:
            np.array: Array containing generated hidden outliers.
        """
        self.start_time = time.time()
        self.fitted_subspaces_dict = fit_in_all_subspaces(self.outlier_detection_method, self.data, seed=self.seed,
                                                          tempdir=self.tempFName, subspace_limit=SUBSPACE_LIMIT,
                                                          n_jobs=n_jobs)
        self.outlier_indices = self.get_outlier_indices()
        seed = self.seed
        length = np.max(np.sqrt(np.sum(self.data ** 2, axis=1)))  # length of the largest Vector in the dataset
        origin_method = OriginMethod.get_origin(self.data, self.outlier_indices, get_origin_type)
        origin = origin_method.calculate_origin()
        fitted_subspaces = self.fitted_subspaces_dict
        full_space = self.full_space
        print("Generating {} hidden outlier points...".format(gen_points))

        with Parallel(n_jobs=n_jobs, timeout=60) as parallel:
            print("n jobs: ", parallel.n_jobs)
            bisection_results = \
                np.array(
                    parallel(
                        delayed(parallel_routine_generate_point)(
                            i,
                            length,
                            check_fast,
                            fixed_interval_length,
                            origin,
                            full_space,
                            fitted_subspaces,
                            get_origin_type,
                            seed,
                            origin_method,
                            verbose
                        )
                        for i in range(gen_points)))

        hidden_x_type = bisection_results[:, -1].reshape(-1, 1)
        hidden_x_list = bisection_results[:, :-1].astype(float)
        hidden_x_list = hidden_x_list[np.sum(hidden_x_list, axis=1) != 0, :]
        # delete the tempdirHOGEN folder if it exists
        if os.path.exists(self.tempFName):
            shutil.rmtree(self.tempFName)

        self.exec_time = time.time() - self.start_time
        print("Done! Exec time: ", self.exec_time)
        self.hidden_x_list = hidden_x_list
        self.hidden_x_type = hidden_x_type

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
              'Outlier detection method used: ' + self.outlier_detection_method.name + '\n' +
              'Synthetic HO generation method employed: ' + "multi_bisection" + '.\n\n' +
              'Database summary' + ':\n\n' +
              '* ' + 'Number of features: ' + str(db_cols) + '\n' +
              '* ' + 'Total number of data points: ' + str(db_rows) + '\n' +
              '* ' + 'Total amount of synthetic data generated: ' + str(ho_rows) + '\n' +
              '\t' + '...of which hidden outliers: ' + str(ho_hidden) + '\n' +
              '* ' + 'Number of H1 outliers: ' + str(h1_outliers) + '\n' +
              '* ' + 'Number of H2 outliers: ' + str(h2_outliers) + '.\n\n' +
              'Total execution time: ' + str(self.exec_time) + '.')

    def save_to_csv(self, file_name):
        """
        Description: This function saves the generated hidden outliers to a CSV file.

        Args:
            file_name (str): The name of the CSV file to save.

        Returns:
            None
        """
        np.savetxt(file_name, self.hidden_x_list, delimiter=",")
