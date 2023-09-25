import logging
import random
from itertools import chain, combinations
from hog_bisect.outlier_detection_method import OutlierDetectionMethod, get_outlier_detection_method

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

FITTING_TIMEOUT_TIME = 60
DEFAULT_SEED = 5


def random_unif_on_sphere(number, dimensions, r=1, random_state=5):
    normal_deviates = norm.rvs(size=(number, dimensions), random_state=random_state)
    radius = np.sqrt((normal_deviates ** 2).sum(axis=1))[:, np.newaxis]
    points = normal_deviates / radius
    return points * r


def gen_powerset(dims):
    return set(chain.from_iterable(combinations(range(dims), r) for r in range(1, dims)))


def subspace_grab(indices, data):
    return data[:, np.array(indices)]


def gen_rand_subspaces(dims, upper_limit, include_all_attr=True, seed=5):
    rd = random.Random(seed)
    features = list(range(0, dims))
    rd.shuffle(features, )
    subspaces = set()
    # includes every attribute singleton and for every attribute a random subspace with more than 1 feature
    # containing it
    if include_all_attr:
        for i in features:
            r = rd.randint(2, dims - 1)
            fts = rd.sample(range(dims), r)
            fts.append(i)
            subspace1 = tuple(fts)
            subspace2 = tuple([i])
            if subspace1 not in subspaces:  # ensure it's a new subspace
                subspaces.add(subspace1)
            if subspace2 not in subspaces:
                subspaces.add(subspace2)

    # avoid sampling singletons, because they are already included
    if include_all_attr:
        lower_limit = 2
    else:
        lower_limit = 1

    while len(subspaces) < (2 ** upper_limit) - 2:
        r = rd.randint(lower_limit, dims - 1)
        random_comb = tuple(rd.sample(range(dims), r))
        if random_comb not in subspaces:
            subspaces.add(random_comb)
    return subspaces


def fit_model(subspace, data, outlier_detection_method, tempdir) -> (tuple, OutlierDetectionMethod):
    """
    Fit a model for a given subspace.

    Args:
    subspace (tuple): The subspace to fit the model on.
    data (ndarray): The dataset.
    outlier_detection_method (class): The outlier detection model class.
    tempdir (str): Temporary directory for storing data.

    Returns:
    tuple: The subspace and the fitted model.
    """
    odm = get_outlier_detection_method(outlier_detection_method)
    model = odm(subspace, tempdir)
    model.fit(subspace_grab(subspace, data))
    return subspace, model


def fit_in_all_subspaces(outlier_detection_method, data, tempdir, subspace_limit, seed=DEFAULT_SEED, n_jobs=1) -> dict:
    """
    Fits models for all possible subspaces of the given data.

    Args:
        outlier_detection_method (class): The outlier detection model class.
        data (ndarray): The dataset.
        tempdir (str): Temporary directory for storing data.
        subspace_limit: 2^subspace_limit will be the maximum amount of subspaces fitted
        seed (int, optional): Seed for random number generator. Defaults to 5.
        n_jobs (int): number of cores to use

    Returns:
        dict: Dictionary of models fitted on different subspaces.

    """
    # Determine the number of dimensions in the data
    dims = data.shape[1]

    # Choose subspaces either from powerset or randomly based on dimensionality
    if dims < subspace_limit:
        subspaces = gen_powerset(dims)
    else:
        subspaces = gen_rand_subspaces(dims, subspace_limit, include_all_attr=True, seed=seed)
    # Log information
    logging.info("Fitting all subspaces....")
    logging.debug(f"number of jobs in parallel: {n_jobs}")
    # Parallel execution for model fitting
    results = Parallel(n_jobs=n_jobs, timeout=FITTING_TIMEOUT_TIME)(
        delayed(fit_model)(subspace, data, outlier_detection_method, tempdir) for subspace in subspaces)
    fitted_subspaces = dict(results)

    # Additional handling for full space
    logging.info("Fitting in the full space....")
    full_space = tuple(range(0, dims))
    fitted_subspaces[full_space] = get_outlier_detection_method(outlier_detection_method)(full_space, tempdir)
    fitted_subspaces[full_space].fit(data)
    del results  # Free up memory

    logging.info(f"number of fitted subspaces including full space: {len(fitted_subspaces)}")
    return fitted_subspaces
