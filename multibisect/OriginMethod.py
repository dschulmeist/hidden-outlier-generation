from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from multiprocessing import cpu_count, Manager


class OriginMethod(ABC):
    def __init__(self, data, out_indicator):
        self.data = data
        self.out_indicator = out_indicator

    @abstractmethod
    def calculate_origin(self):
        pass


class CentroidOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator)
        self.mean = data.mean(axis=0)

    def calculate_origin(self):
        return self.mean


class LeastOutlierOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator)
        lof = LocalOutlierFactor()
        lof.fit(data)
        self.index = np.argmax(-lof.negative_outlier_factor_)
        self.origin = data[self.index, :]

    def calculate_origin(self):
        return self.origin


class RandomOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator)
        self.out_data = data[out_indicator == 0]
        self.out_data_length = self.out_data.shape[0]

    def calculate_origin(self):
        index = np.random.choice(self.out_data_length)
        return self.out_data[index, :]


class WeightedOrigin(OriginMethod):
    def __init__(self, data, out_indicator):
        super().__init__(data, out_indicator)
        print("Calculating probability vector...")
        lof = LocalOutlierFactor()
        lof.fit(self.data)
        self.proba_vector = -lof.negative_outlier_factor_
        self.proba_vector /= np.sum(self.proba_vector)
        print("Done!")
        self.out_df = data[out_indicator == 0]
        self.proba_vector_out = self.proba_vector[out_indicator == 0]
        self.proba_vector_out_sum = self.proba_vector_out.sum()
        self.out_df_length = self.out_df.shape[0]

    def calculate_origin(self):
        index = np.random.choice(self.out_df_length, p=self.proba_vector_out / self.proba_vector_out_sum)
        return self.out_df[index, :]


# Create a dictionary to map strings to classes
origin_method_classes = {
    "centroid": CentroidOrigin,
    "least_outlier": LeastOutlierOrigin,
    "random": RandomOrigin,
    "weighted": WeightedOrigin,
}


# Use the classes
def get_origin(data, out_indicator, or_type):
    print(f"given Origin method: {or_type}")
    if or_type in origin_method_classes:
        method = origin_method_classes[or_type]
        return method(data, out_indicator)
    else:
        raise ValueError(
            "Invalid type argument provided. Should be one of 'centroid', 'least_outlier', 'random', 'weighted'")
