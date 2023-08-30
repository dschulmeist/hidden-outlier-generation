import os
import pickle
from abc import abstractmethod, ABC
from datetime import time

from pyod.models import deep_svdd, abod, ecod
from pyod.models.lof import LOF
from scipy.spatial import distance
import numpy as np
from scipy.stats import chi2


class OutlierDetectionMethod(ABC):
    name = ""

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


class OdLOF(OutlierDetectionMethod):

    def __init__(self, subspace, tempdir):
        super().__init__()
        self.model = None
        self.tempdir = tempdir
        self.subspace = subspace
        self.name = "LOF_on_" + str(hash(subspace))
        self.location = f'{self.tempdir}/{self.name}.pkl'

    def fit(self, data):
        model = LOF()
        model.fit(data)
        self.fitted = True
        self.dump(model)
        del model

    def dump(self, model):
        # check whether the directory tempdir exists or not
        # if not, create a new one
        if not os.path.exists(self.tempdir):
            os.makedirs(self.tempdir)
        # dump the model to tempdir
        with open(self.location, 'wb') as f:
            pickle.dump(model, f)
        del model

    def get_model(self) -> LOF:
        # load model from disk
        with open(self.location, 'rb') as f:
            loaded_model = pickle.load(f)
        return loaded_model

    def predict(self, x) -> bool:
        if not self.fitted:
            raise "trying to predict OdLOF that was not fitted yet"
        model = self.get_model()
        x = x.reshape(1, -1)
        decision = bool(model.predict(x)[0, ])
        del model
        return decision


class OdDeepSVDD(OutlierDetectionMethod):
    def __init__(self):
        super().__init__()
        self.model = deep_svdd.DeepSVDD()

    name = "DeepSVDD"

    def fit(self, data):
        self.model.fit(data)
        self.fitted = True
        pass

    def predict(self, x):
        pass


class ODmahalanobis(OutlierDetectionMethod):
    def __init__(self, data):
        super().__init__()
        self.crit_val = None
        self.mean = None
        self.inv_cov = None

    name = "mahalonis"

    def fit(self, data):
        self.shape = data.shape
        self.mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        self.inv_cov = np.linalg.inv(cov)
        self.crit_val = self.critval()
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise "trying to predict ODMahalonis that was not fitted yet"
        mahalanobis_dist = distance.mahalanobis(x, self.mean, self.inv_cov)
        return 1 if mahalanobis_dist > self.crit_val else 0

    def critval(self):
        """Critical value given by a normal DB while using the MD.
    
        Args:
            S (set): Set of indices of the features conforming the DB.
        """
        return chi2.ppf(0.95, df=self.shape[1])  # Updated to work with numpy array


outl_detect_methods = [LOF, deep_svdd.DeepSVDD, abod.ABOD, ecod.ECOD]
