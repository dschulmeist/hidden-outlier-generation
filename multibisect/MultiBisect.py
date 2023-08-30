import os
import random
import shutil
import time
import logging
from datetime import datetime

from itertools import combinations

import numpy as np

from multibisect.OutlierDetectionMethod import OdLOF
from numpy import ndarray
from joblib import Parallel, delayed

from multibisect import Utils, OriginMethod

n_jobs = -2
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SUBSPACE_LIMIT = 12
FITTING_TIMEOUT_TIME = 60
DEFAULT_SEED = 5
DEFAULT_NUMBER_OF_ITERATIONS = 30
DEFAULT_NUMBER_OF_GEN_POINTS = 100


def fit_model(subspace, data, outlier_detection_method, tempdir):
    model = outlier_detection_method(subspace, tempdir)
    model.fit(subspace_grab(subspace, data))
    return subspace, model


def fit_in_all_subspaces(outlier_detection_method, data, tempdir, seed=DEFAULT_SEED):
    """
    Fits models for all subspaces.

    Args:
        outlier_detection_method (func): Outlier detection method
        data (np.array): Data.
        tempdir (str): Temporary directory.
        seed (int, optional): Seed. Defaults to 5.

    Returns:
        dict: Dictionary of models fitted on different subspaces.
    """
    dims = data.shape[1]

    if dims < SUBSPACE_LIMIT:
        subspaces = Utils.gen_powerset(dims)
    else:
        subspaces = Utils.gen_rand_subspaces(dims, SUBSPACE_LIMIT, include_all_attr=True, seed=seed)

    logging.info("Fitting all subspaces....")
    results = Parallel(n_jobs=n_jobs, timeout=FITTING_TIMEOUT_TIME)(
        delayed(fit_model)(subspace, data, outlier_detection_method, tempdir) for subspace in subspaces)
    fitted_subspaces = dict(results)

    logging.info("Fitting in the full space....")
    full_space = frozenset(range(0, dims))
    fitted_subspaces[full_space] = outlier_detection_method(full_space, tempdir)
    fitted_subspaces[full_space].fit(data)

    del results

    logging.info(f"Set of fitted Subspaces size: {len(fitted_subspaces)}")

    return fitted_subspaces


def fit_in_all_subspaces1(outl_detec_method, data, tempdir, seed=5):
    dims = data.shape[1]
    limit = 12
    if dims < limit:
        subspaces = set()
        for r in range(1, dims):
            for subspace in combinations(range(0, dims), r):
                subspaces.add(subspace)
        print("fitting all subspaces....")
        results = Parallel(n_jobs=n_jobs, timeout=60)(
            delayed(fit_model)(subspace, data, outl_detec_method, tempdir) for subspace in subspaces)
        odm_dict = dict(results)
    else:
        print("dims > ", limit - 1)
        rd = random.Random(seed)
        features = list(range(0, dims))
        rd.shuffle(features, )
        subspaces = set()
        for i in features:
            r = rd.randint(1, dims - 1)
            fts = rd.sample(range(dims), r)
            fts.append(i)
            subspace = frozenset(fts)
            if subspace not in subspaces:  # ensure it's a new subspace
                subspaces.add(subspace)

        while len(subspaces) < 2 ** limit - 2:
            r = rd.randint(1, dims - 1)
            random_comb = frozenset(rd.sample(range(dims), r))
            if random_comb not in subspaces:
                subspaces.add(random_comb)

        print("fitting all subspaces....")
        results = Parallel(n_jobs=n_jobs, timeout=60)(
            delayed(fit_model)(subspace, data, outl_detec_method, tempdir) for subspace in subspaces)

        odm_dict = dict(results)

    print("fitting in the full space....")
    full_space = tuple(range(0, dims))
    tupl = (outl_detec_method, full_space)
    odm_dict[tupl] = outl_detec_method(full_space, tempdir)
    odm_dict[tupl].fit(data)
    del results
    print("ODM size: ", len(odm_dict))
    return odm_dict


def outlier_check(x, verb=False, fast=True, full_space=None, odm_dict=None, odm=None):
    supS = odm_dict.keys()
    index = np.zeros(len(supS))
    isOutlierInFullSpace = False
    isOutlierInSubspace = False

    if inference(x, full_space, odm_dict=odm_dict, odm=odm):
        isOutlierInFullSpace = True
    for j, S in enumerate(supS):
        subspace = S[1]
        if subspace == full_space:
            continue
        if inference(subspace_grab(subspace, x.reshape(1, x.shape[0])), subspace, odm_dict=odm_dict, odm=odm):
            index[j] = 1

            if fast:  # remove
                break

    if np.sum(index[:-1]) > 0:
        isOutlierInSubspace = True

    if isOutlierInSubspace and (not isOutlierInFullSpace):
        result = [0, "H1"]
    elif (not isOutlierInSubspace) and isOutlierInFullSpace:
        result = [0, "H2"]
    elif isOutlierInSubspace and isOutlierInFullSpace:
        if verb:
            print("x Outside of bounds")
        result = [1, "OB"]
    else:
        if verb:
            print("x in the total acceptance area")
        result = [-1, "IL"]

    return result


def inference(x, subspace, odm_dict=None, odm=None):
    if not isinstance(subspace, tuple):
        print("Subspace: ", subspace, type(subspace), " is not a tuple")
        raise ValueError(
            "Error in Code, subspace needs to be a tuple")

    outl_detect_method = odm
    tupl = (outl_detect_method, subspace)
    if tupl not in odm_dict.keys():
        raise ("Subspace {} not in ODM_env".format(subspace))
        # odm_dict[tupl] = outl_detect_method(subspace_grab(subspace, self.data))
    return odm_dict[tupl].predict(x)


def interval_check(length, x, origin, parts=5, full_space=None, odm_dict=None, odm=None):
    """
    Interval Refining function

    Breaks the interval in however many parts selected
    (defaults to 5) and checks if each part is an Outlier or an Inlier
    in the global space (D). After that it stores each subinterval such
    in a single list, that then is returned

    Arguments:
    l: Length of the original interval
    method: ODM
    x: Directional vector selected
    parts: Number of parts to which cut the interval
    """

    segmentation_points = np.linspace(0, length, num=parts)
    check = np.full(len(segmentation_points), -1)
    # print("checking intervals in the global space")
    for i, c in enumerate(segmentation_points):

        if inference(c * x + origin, subspace=full_space, odm_dict=odm_dict, odm=odm):
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


def multi_bisect(x, interval_length, origin, iternum=DEFAULT_NUMBER_OF_ITERATIONS, check_version="fast",
                 l_val_option="fixed", full_space=None, odm_dict=None, odm=None):
    """
    Multi Bisection algorithm function

    Performs the multi bisection algorithm to any given
    L and direction x. It outputs the value c in which the function f
    finds a 0 of the form: f(c*x + origin).

    Arguments:
    x: Directional vector
    l: Length of the original interval
    iternum: Number of iterations to perform
    method: ODM
    """
    c = 0.5
    outlier_type = ''
    if l_val_option != "fixed":
        interval_length = interval_length + np.random.uniform(-interval_length / 2, interval_length)
    interval = interval_check(interval_length, x, origin, full_space=full_space, odm_dict=odm_dict, odm=odm)
    choice = np.random.choice(len(interval), 1)[0,]
    interval = interval[choice]
    interval_indicator = interval[1]
    interval = interval[0]

    a = interval[0]
    b = interval[1]
    for i in range(iternum):
        c = (b + a) / 2
        if check_version == "fast":
            check_if_outlier = outlier_check(c * x + origin, fast=True, full_space=full_space, odm_dict=odm_dict,
                                             odm=odm)

        else:
            check_if_outlier = outlier_check(c * x + origin, fast=False, full_space=full_space, odm_dict=odm_dict,
                                             odm=odm)

        outlier_indicator = check_if_outlier[0]
        outlier_type = check_if_outlier[1]

        if outlier_indicator == 0:
            # print(f"x in {outlier_type}")
            return [c, outlier_type]
        if outlier_indicator == interval_indicator[1]:
            b = c
        else:
            a = c
    return [c, outlier_type]


def process_point(i, interval_length,
                  check_version, l_val_option, origin,
                  full_space, odm_dict, odm, dims,
                  get_origin_type, seed, om):
    if i % 500 == 0:
        print("Processing point ", i)
    x = Utils.random_unif_on_sphere(2, dims, 1, seed)[0,]
    bisec_res = multi_bisect(interval_length=interval_length, x=x,
                             check_version=check_version,
                             l_val_option=l_val_option,
                             origin=origin, full_space=full_space, odm_dict=odm_dict, odm=odm)
    hidden_c = bisec_res[0]
    outlier_type = bisec_res[1]

    if get_origin_type in ["random", "weighted"]:
        origin = om.calculate_origin()

    if outlier_type in ["H1", "H2"]:
        result_point = hidden_c * x + origin
        result_point = np.append(result_point, outlier_type)
    else:
        result_point = np.zeros((1, dims))
        result_point = np.append(result_point, outlier_type)

    return result_point


class HOGen:
    def __init__(self, data, outlier_detection_method=OdLOF, seed=DEFAULT_SEED):
        self.hidden_x_type = None
        self.hidden_x_list = None
        self.start_time = time.time()
        self.tempFName = f'HOGTemp{datetime.now().strftime("%Y%m%d%H%M%S")}_{seed}'
        self.x_list = None
        self.seed = seed
        self.data = data
        self.dims = self.data.shape[1]
        self.outlier_detection_method = outlier_detection_method
        self.odm_dict = None
        self.outlier_indices = None
        self.full_space = tuple(np.arange(self.data.shape[1]))
        self.exec_time = None

    def get_outl_ind(self):
        """
        Description: This function returns an array with the results of applying
                     the inference method to each row of the data. The inference method
                     is applied with a tuple of all possible combinations of indices
                     for the number of columns.

        Returns:
        np.array: An array containing the results of the inference method.
        """
        # Calculate the number of rows and columns
        n_rows, n_cols = self.data.shape

        # Initialize the output array
        outl_ind = np.zeros(n_rows)

        # Create a tuple of all column indices
        col_indices = tuple(range(n_cols))

        # Iterate over each row
        for i in range(n_rows):
            # Apply the inference method to the row with the tuple of column indices
            outl_ind[i] = inference(self.data[i, :], col_indices, self.odm_dict, self.outlier_detection_method)

        return outl_ind

    def main_multi_bisect(self, gen_points=100, check_version="fast", num_workers=4,
                          l_val_option="fixed", get_origin_type="weighted"):
        self.start_time = time.time()
        self.odm_dict = fit_in_all_subspaces(self.outlier_detection_method, self.data, seed=self.seed, tempdir=self.tempFName)
        self.outlier_indices = self.get_outl_ind()
        seed = self.seed
        np.random.seed(seed)
        length = np.max(np.sqrt(np.sum(self.data ** 2, axis=1)))  # length of the largest Vector in the dataset
        om = OriginMethod.get_origin(self.data, self.outlier_indices, get_origin_type)
        origin = om.calculate_origin()
        odm_dict = self.odm_dict
        full_space = self.full_space
        odm = self.outlier_detection_method
        dims = self.dims
        print("Generating {} hidden outlier points...".format(gen_points))

        with Parallel(n_jobs=n_jobs, timeout=60) as parallel:
            print("n jobs: ", parallel.n_jobs)
            bisection_results = \
                np.array(
                    parallel(
                        delayed(
                            process_point)(
                            i, length,
                            check_version,
                            l_val_option, origin,
                            full_space, odm_dict,
                            odm, dims,
                            get_origin_type, seed, om)
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
        gen_result = HogMethod(self.data, gen_points, self.outlier_detection_method.name, "multi_bisection",
                               hidden_x_list, hidden_x_type, self.exec_time, self.x_list)

        return gen_result

    # print a summary, similar to summary_hog_method
    def print_summary(self):
        db_cols = self.data.shape[1]
        db_rows = self.data.shape[0]
        ho_rows = len(self.hidden_x_type)
        ho_hidden = 1 if isinstance(self.hidden_x_list, list) and len(self.hidden_x_list) != 0 else (
            len(self.hidden_x_list) if isinstance(self.hidden_x_list, ndarray) else 0
        )
        h1_outliers = self.hidden_x_type[self.hidden_x_type['V1'] == 'H1'].shape[0]
        h2_outliers = self.hidden_x_type[self.hidden_x_type['V1'] == 'H2'].shape[0]

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


class HogMethod:
    def __init__(self, data, B, od_name, hog_name, ho_list, ho_type, exec_time, directions=None):
        self.data = data
        self.B = B
        self.od_name = od_name
        self.hog_name = hog_name
        self.ho_list = ho_list
        self.ho_type = ho_type
        self.exec_time = exec_time
        self.directions = directions

    def print_hog_method(self):
        print('Hidden Outlier Generation Method Object' + '\n\n' +
              'Outlier Gen Method used: ' + self.od_name + '\n' +
              'Synthetic HO generation method employed: ' + self.hog_name + '.\n\n' +
              'Use summary() for a detailed description' + '\n')

    def summary_hog_method(self):
        db_cols = self.data.shape[1]
        db_rows = self.data.shape[0]
        ho_rows = len(self.ho_type)
        ho_hidden = 1 if isinstance(self.ho_list, list) and len(self.ho_list) != 0 else (
            len(self.ho_list) if isinstance(self.ho_list, ndarray) else 0
        )
        h1_outliers = self.ho_type[self.ho_type['V1'] == 'H1'].shape[0]
        h2_outliers = self.ho_type[self.ho_type['V1'] == 'H2'].shape[0]

        print('Hidden Outlier Generation Method Object' + '\n\n' +
              'Outlier detection method used: ' + self.od_name + '\n' +
              'Synthetic HO generation method employed: ' + self.hog_name + '.\n\n' +
              'Database summary' + ':\n\n' +
              '* ' + 'Number of features: ' + str(db_cols) + '\n' +
              '* ' + 'Total number of data points: ' + str(db_rows) + '\n' +
              '* ' + 'Total amount of synthetic data generated: ' + str(ho_rows) + '\n' +
              '\t' + '...of which hidden outliers: ' + str(ho_hidden) + '\n' +
              '* ' + 'Number of H1 outliers: ' + str(h1_outliers) + '\n' +
              '* ' + 'Number of H2 outliers: ' + str(h2_outliers) + '.\n\n' +
              'Total execution time: ' + str(self.exec_time) + '.')
