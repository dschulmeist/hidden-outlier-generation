"""
Origin calculation strategies for hidden outlier generation.

This module implements the Strategy pattern for calculating the origin point
from which the bisection algorithm searches outward. The choice of origin
significantly affects which hidden outliers are discovered.

Different strategies suit different data distributions:
    - CentroidOrigin: Good for symmetric, unimodal distributions
    - LeastOutlierOrigin: Finds the most "normal" point as origin
    - RandomOrigin: Explores from random inlier points each iteration
    - WeightedOrigin: Biases toward more normal points probabilistically

The random and weighted strategies allow discovering more diverse hidden
outliers by varying the search origin across iterations.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

if TYPE_CHECKING:
    from numpy.typing import NDArray


class OriginMethod(ABC):
    """Abstract base class for origin calculation strategies.

    The origin is the starting point for the bisection search. Different
    strategies can find different hidden outliers depending on the data
    distribution and where the search begins.

    Attributes:
        class_type: The OriginType enum value for this strategy.
        data: The full dataset used for origin calculation.
        out_indicator: Binary array where 1 = outlier, 0 = inlier.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        out_indicator: NDArray[np.floating[Any]],
        class_type: OriginType,
    ) -> None:
        """Initialize the origin method with data and outlier labels.

        Args:
            data: Dataset array of shape (n_samples, n_features).
            out_indicator: Binary array of shape (n_samples,) indicating outliers.
            class_type: The OriginType enum value identifying this strategy.
        """
        self.class_type = class_type
        self.data = data
        self.out_indicator = out_indicator

    @abstractmethod
    def calculate_origin(self) -> NDArray[np.floating[Any]]:
        """Calculate and return the origin point.

        Returns:
            Origin point as array of shape (n_features,).
        """
        pass


class CentroidOrigin(OriginMethod):
    """Use the data centroid (mean) as the origin.

    Simple and deterministic - always returns the same point. Works well
    when the data is roughly symmetric around its center. May miss hidden
    outliers in asymmetric distributions.

    Attributes:
        mean: Precomputed centroid of the data.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        out_indicator: NDArray[np.floating[Any]],
    ) -> None:
        super().__init__(data, out_indicator, OriginType.CENTROID)
        self.mean = data.mean(axis=0)

    def calculate_origin(self) -> NDArray[np.floating[Any]]:
        """Return the precomputed data centroid."""
        return self.mean


class LeastOutlierOrigin(OriginMethod):
    """Use the least outlying point as the origin.

    Finds the point with the lowest Local Outlier Factor score, meaning
    it's the most "normal" point in the dataset. This provides a stable
    origin that's guaranteed to be deep within the inlier region.

    Useful when you want consistent results and the data has a clear
    dense core region.

    Attributes:
        index: Index of the least outlying point in the data.
        origin: The least outlying point itself.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        out_indicator: NDArray[np.floating[Any]],
    ) -> None:
        super().__init__(data, out_indicator, OriginType.LEAST_OUTLIER)
        # LOF returns negative scores where more negative = more outlying
        # We want the least outlying, so find max of negative scores
        lof = LocalOutlierFactor()
        lof.fit(data)
        self.index = np.argmax(-lof.negative_outlier_factor_)
        self.origin = data[self.index, :]

    def calculate_origin(self) -> NDArray[np.floating[Any]]:
        """Return the precomputed least outlying point."""
        return self.origin


class RandomOrigin(OriginMethod):
    """Use a random inlier point as the origin.

    Each call to calculate_origin() returns a different random point
    from the inlier set. This allows discovering hidden outliers in
    different directions across multiple bisection iterations.

    Only samples from points marked as inliers (out_indicator == 0).

    Attributes:
        out_data: Subset of data containing only inlier points.
        out_data_length: Number of inlier points available.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        out_indicator: NDArray[np.floating[Any]],
    ) -> None:
        super().__init__(data, out_indicator, OriginType.RANDOM)
        # Filter to only inlier points (out_indicator == 0 means inlier)
        self.out_data = data[out_indicator == 0]
        self.out_data_length = self.out_data.shape[0]

    def calculate_origin(self) -> NDArray[np.floating[Any]]:
        """Return a uniformly random inlier point."""
        index = np.random.choice(self.out_data_length)
        return self.out_data[index, :]


class WeightedOrigin(OriginMethod):
    """Use a weighted random inlier point as the origin.

    Similar to RandomOrigin but biases toward more "normal" points.
    Points with lower LOF scores (less outlying) have higher probability
    of being selected. This focuses the search around the dense core
    while still allowing variation.

    The weighting uses inverse LOF scores normalized to a probability
    distribution over inlier points.

    Attributes:
        out_df: Subset of data containing only inlier points.
        proba_vector_out: Probability weights for each inlier point.
        out_df_length: Number of inlier points available.
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]],
        out_indicator: NDArray[np.floating[Any]],
    ) -> None:
        super().__init__(data, out_indicator, OriginType.WEIGHTED)

        logging.debug("Calculating probability vector for weighted sampling...")
        lof = LocalOutlierFactor()
        lof.fit(self.data)

        # Convert LOF scores to probabilities (higher = less outlying = more likely)
        self.proba_vector = -lof.negative_outlier_factor_
        self.proba_vector /= np.sum(self.proba_vector)
        logging.debug("Probability vector computed.")

        # Filter to inliers only
        self.out_df = data[out_indicator == 0]
        self.proba_vector_out = self.proba_vector[out_indicator == 0]
        self.proba_vector_out_sum = self.proba_vector_out.sum()
        self.out_df_length = self.out_df.shape[0]

    def calculate_origin(self) -> NDArray[np.floating[Any]]:
        """Return a weighted random inlier point, biased toward normal points."""
        index = np.random.choice(
            self.out_df_length,
            p=self.proba_vector_out / self.proba_vector_out_sum,
        )
        return self.out_df[index, :]


class OriginType(Enum):
    """Enumeration of available origin calculation strategies."""

    CENTROID = "centroid"
    LEAST_OUTLIER = "least outlier"
    RANDOM = "random"
    WEIGHTED = "weighted"

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def from_str(s: str) -> OriginType:
        """Create OriginType from string representation.

        Args:
            s: String matching one of the enum values.

        Returns:
            The corresponding OriginType enum member.

        Raises:
            ValueError: If the string doesn't match any origin type.
        """
        return OriginType(s)

    @classmethod
    def get_class_for_origin_type(cls, origin_type: str) -> type[OriginMethod]:
        """Get the OriginMethod subclass for a given type string.

        Args:
            origin_type: One of "centroid", "least outlier", "random", "weighted".

        Returns:
            The corresponding OriginMethod subclass (not an instance).
        """
        mapping: dict[str, type[OriginMethod]] = {
            "centroid": CentroidOrigin,
            "least outlier": LeastOutlierOrigin,
            "random": RandomOrigin,
            "weighted": WeightedOrigin,
        }
        return mapping[origin_type]


def get_origin(
    data: NDArray[np.floating[Any]],
    out_indicator: NDArray[np.floating[Any]],
    or_type: str,
) -> OriginMethod:
    """Factory function to create an origin calculation strategy.

    This is the main entry point for obtaining an origin method. It handles
    the mapping from string type names to the appropriate class and instantiation.

    Args:
        data: Dataset array of shape (n_samples, n_features).
        out_indicator: Binary array where 1 = outlier, 0 = inlier.
        or_type: Origin method name, one of:
            - "centroid": Use data mean
            - "least outlier": Use most normal point
            - "random": Random inlier each call
            - "weighted": Weighted random toward normal points

    Returns:
        An initialized OriginMethod instance ready to use.

    Raises:
        ValueError: If or_type is not a recognized origin method name.

    Example:
        >>> origin_method = get_origin(data, outlier_labels, "weighted")
        >>> origin_point = origin_method.calculate_origin()
    """
    logging.debug(f"Creating origin method: {or_type}")
    try:
        class_type = OriginType.from_str(or_type)
        method_class = class_type.get_class_for_origin_type(or_type)
        return method_class(data, out_indicator)
    except ValueError as err:
        raise ValueError(f"Unknown origin method: {or_type}") from err
