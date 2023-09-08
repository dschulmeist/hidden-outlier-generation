import os
import shutil
import unittest

import numpy as np
import pyod

from src.hog_bisect.bisect import outlier_check, interval_check, inference
from src.hog_bisect.outlier_detection_method import OdPYOD
from src.hog_bisect import outlier_result_type
from src.hog_bisect.utils import fit_model, fit_in_all_subspaces
from src.hog_bisect.outlier_result_type import OutlierResultType


class TestBisectBasic(unittest.TestCase):
    def setUp(self):
        self.seed = 5
        np.random.seed(self.seed)
        dummy_data = np.random.rand(100, 5)
        self.data = dummy_data  # np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Dummy data
        self.tempdir = "./tempdir"  # Dummy temporary directory
        self.subspace = (0, 1)  # Dummy subspace
        self.full_space = (0, 1, 2, 3, 4)  # Dummy full space
        self.outlier_detection_method = pyod.models.lof.LOF

        self.fitted_subspaces = fit_in_all_subspaces(self.outlier_detection_method, self.data, self.tempdir,
                                                     seed=self.seed, subspace_limit=12, n_jobs=-2)

    def tearDown(self):
        # Delete the temporary directory and its contents
        if os.path.exists(self.tempdir):
            shutil.rmtree(self.tempdir)

    def test_fit_in_all_subspaces(self):
        # fitted_subspaces = fit_in_all_subspaces(self.outlier_detection_method, self.data, self.tempdir,
        # seed=self.seed)
        self.assertIsInstance(self.fitted_subspaces, dict)

    def test_fit_model(self):
        subspace, model = fit_model(self.subspace, self.data, self.outlier_detection_method, self.tempdir)
        self.assertEqual(subspace, self.subspace)
        self.assertIsInstance(model, OdPYOD)

    def test_outlier_check(self):
        # fitted_subspaces = fit_in_all_subspaces(self.outlier_detection_method, self.data, self.tempdir,
        # seed=self.seed)

        result = outlier_check(self.data[0], self.full_space,
                               fitted_subspaces=self.fitted_subspaces, verb=False, fast=True)
        self.assertIsInstance(result, OutlierResultType)

    def test_interval_check(self):
        length = 5
        direction = np.array([1, 0, 0, 0, 0])
        origin = np.array([0, 0, 0, 0, 0])
        # fitted_subspaces = fit_in_all_subspaces(self.outlier_detection_method, self.data, self.tempdir,
        # seed=self.seed)
        result = interval_check(length, direction, origin, parts=5, full_space=self.full_space,
                                fitted_subspaces=self.fitted_subspaces)
        self.assertIsInstance(result, list)

    def test_inference(self):
        # fitted_subspaces = fit_in_all_subspaces(self.outlier_detection_method, self.data, self.tempdir,
        # seed=self.seed)
        result = inference(self.data[0], self.full_space,
                           fitted_subspaces=self.fitted_subspaces)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
