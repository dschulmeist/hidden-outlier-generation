from enum import Enum


class OutlierResultType(Enum):
    """
    Enum for the possible result types for the outlier check.

    """
    def __new__(cls, value, indicator):
        obj = object.__new__(cls)
        obj._value_ = str(value)
        obj.indicator = int(indicator)
        return obj

    H1 = ("H1", 0)
    H2 = ("H2", 0)
    OB = ("Outside Bounds", 1)
    IL = ("Inlier", -1)

