"""
Result type definitions for outlier classification.

This module defines the possible outcomes when checking whether a generated
point is a hidden outlier. The bisection algorithm produces points that fall
into one of four categories based on their outlier status in the full space
versus subspaces.
"""

from enum import Enum


class OutlierResultType(Enum):
    """Classification result for a point checked by the outlier detection algorithm.

    Each result type has an associated indicator value used during the bisection
    algorithm to determine which direction to search next.

    Result Types:
        H1: Hidden outlier Type 1 - outlier in at least one subspace but NOT in
            the full space. These are the primary target of the generation algorithm.
        H2: Hidden outlier Type 2 - outlier in the full space but NOT in any
            subspace. These represent the inverse hiding phenomenon.
        OB: Outside Bounds - outlier in both the full space AND at least one
            subspace. Not a hidden outlier, used as boundary marker.
        IL: Inlier - not an outlier in any space. Used as boundary marker on
            the opposite side from OB.

    The indicator values (0, 1, -1) are used by the bisection algorithm to
    navigate toward hidden outlier regions. H1 and H2 (indicator=0) are the
    target states, while OB (indicator=1) and IL (indicator=-1) define the
    boundaries of the search interval.
    """

    def __new__(cls, value: str, indicator: int) -> "OutlierResultType":
        obj = object.__new__(cls)
        obj._value_ = str(value)
        obj.indicator = int(indicator)
        return obj

    # Hidden outliers - the targets we want to generate
    H1 = ("H1", 0)  # Outlier in subspace only
    H2 = ("H2", 0)  # Outlier in full space only

    # Boundary markers for bisection
    OB = ("Outside Bounds", 1)  # Outlier everywhere
    IL = ("Inlier", -1)  # Outlier nowhere

    # Type hint for the indicator attribute added by __new__
    indicator: int

    def is_hidden_outlier(self) -> bool:
        """Check if this result represents a hidden outlier (H1 or H2)."""
        return self in (OutlierResultType.H1, OutlierResultType.H2)
