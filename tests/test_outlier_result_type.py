"""
Tests for OutlierResultType enum.
"""

from hog_bisect.outlier_result_type import OutlierResultType


class TestOutlierResultType:
    """Tests for the OutlierResultType enumeration."""

    def test_h1_properties(self):
        """H1 has correct properties."""
        assert OutlierResultType.H1.name == "H1"
        assert OutlierResultType.H1.indicator == 0  # Hidden outliers have indicator 0

    def test_h2_properties(self):
        """H2 has correct properties."""
        assert OutlierResultType.H2.name == "H2"
        assert OutlierResultType.H2.indicator == 0  # Hidden outliers have indicator 0

    def test_ob_properties(self):
        """OB (Outside Bounds) has correct properties."""
        assert OutlierResultType.OB.name == "OB"
        assert OutlierResultType.OB.indicator == 1

    def test_il_properties(self):
        """IL (Inlier) has correct properties."""
        assert OutlierResultType.IL.name == "IL"
        assert OutlierResultType.IL.indicator == -1

    def test_is_hidden_outlier_h1(self):
        """H1 is a hidden outlier."""
        assert OutlierResultType.H1.is_hidden_outlier() is True

    def test_is_hidden_outlier_h2(self):
        """H2 is a hidden outlier."""
        assert OutlierResultType.H2.is_hidden_outlier() is True

    def test_is_hidden_outlier_ob(self):
        """OB is not a hidden outlier."""
        assert OutlierResultType.OB.is_hidden_outlier() is False

    def test_is_hidden_outlier_il(self):
        """IL is not a hidden outlier."""
        assert OutlierResultType.IL.is_hidden_outlier() is False

    def test_all_types_exist(self):
        """All four result types exist."""
        types = list(OutlierResultType)
        assert len(types) == 4
        assert OutlierResultType.H1 in types
        assert OutlierResultType.H2 in types
        assert OutlierResultType.OB in types
        assert OutlierResultType.IL in types

    def test_types_are_distinct(self):
        """All result types are distinct."""
        types = list(OutlierResultType)
        assert len(types) == len(set(types))

    def test_indicator_values(self):
        """Hidden outliers have indicator 0, boundaries have +1/-1."""
        # H1 and H2 are targets (indicator 0)
        assert OutlierResultType.H1.indicator == 0
        assert OutlierResultType.H2.indicator == 0
        # OB and IL are boundaries (indicator +1/-1)
        assert OutlierResultType.OB.indicator == 1
        assert OutlierResultType.IL.indicator == -1
