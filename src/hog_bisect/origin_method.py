import logging
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import cpu_count, Manager


class OriginMethod(ABC):

    def __init__(self, data, out_indicator, class_type):
        self.class_type = class_type
        self.data = data
        self.out_indicator = out_indicator

    @abstractmethod
    def calculate_origin(self):
        pass


class CentroidOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator, OriginType.CENTROID)
        self.mean = data.mean(axis=0)

    def calculate_origin(self):
        return self.mean


class LeastOutlierOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator, OriginType.LEAST_OUTLIER)
        lof = LocalOutlierFactor()
        lof.fit(data)
        self.index = np.argmax(-lof.negative_outlier_factor_)
        self.origin = data[self.index, :]

    def calculate_origin(self):
        return self.origin


class RandomOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator, OriginType.RANDOM)
        self.out_data = data[out_indicator == 0]
        self.out_data_length = self.out_data.shape[0]

    def calculate_origin(self):
        index = np.random.choice(self.out_data_length)
        return self.out_data[index, :]


class WeightedOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator, OriginType.WEIGHTED)
        logging.debug("Calculating probability vector...")
        lof = LocalOutlierFactor()
        lof.fit(self.data)
        self.proba_vector = -lof.negative_outlier_factor_
        self.proba_vector /= np.sum(self.proba_vector)
        logging.debug("Done!")
        self.out_df = data[out_indicator == 0]
        self.proba_vector_out = self.proba_vector[out_indicator == 0]
        self.proba_vector_out_sum = self.proba_vector_out.sum()
        self.out_df_length = self.out_df.shape[0]

    def calculate_origin(self):
        index = np.random.choice(self.out_df_length, p=self.proba_vector_out / self.proba_vector_out_sum)
        return self.out_df[index, :]


class OriginType(Enum):
    CENTROID = "centroid"
    LEAST_OUTLIER = "least outlier"
    RANDOM = "random"
    WEIGHTED = "weighted"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(s):
        return OriginType(s)

    @classmethod
    def get_class_for_origin_type(cls, origin_type):
        mapping_to_classes = {
            "centroid": CentroidOrigin,
            "least outlier": LeastOutlierOrigin,
            "random": RandomOrigin,
            "weighted": WeightedOrigin
        }
        return mapping_to_classes[origin_type]


def get_origin(data, out_indicator, or_type) -> OriginMethod:
    logging.debug(f"given Origin method: {or_type}")
    try:
        class_type = OriginType.from_str(or_type)
        method = class_type.get_class_for_origin_type(or_type)

        return method(data, out_indicator)
    except ValueError:
        raise ValueError(
            "No such origin method: " + str(or_type))
