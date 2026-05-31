"""Tests for skxperiments.core.potential_outcomes."""

import numpy as np
import pytest

from skxperiments.core.exceptions import InsufficientDataError
from skxperiments.core.potential_outcomes import PotentialOutcomes


class TestPotentialOutcomesCreation:
    """Tests for PotentialOutcomes instantiation."""

    def test_basic_creation(self) -> None:
        """Should create PotentialOutcomes with valid arrays."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0, 3.0]),
            y1=np.array([2.0, 3.0, 5.0]),
        )
        assert po.n == 3

    def test_creation_with_unit_ids(self) -> None:
        """Should accept unit_ids parameter."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0]),
            y1=np.array([3.0, 4.0]),
            unit_ids=np.array(["a", "b"]),
        )
        assert po.n == 2

    def test_creation_from_lists(self) -> None:
        """Should accept plain lists and convert to arrays."""
        po = PotentialOutcomes(y0=[1, 2, 3], y1=[4, 5, 6])
        assert po.n == 3

    def test_raises_on_different_lengths(self) -> None:
        """Should raise InsufficientDataError if y0 and y1 differ in length."""
        with pytest.raises(InsufficientDataError):
            PotentialOutcomes(
                y0=np.array([1.0, 2.0, 3.0]),
                y1=np.array([2.0, 3.0]),
            )

    def test_raises_on_empty_y0(self) -> None:
        """Should raise InsufficientDataError if y0 is empty."""
        with pytest.raises(InsufficientDataError):
            PotentialOutcomes(y0=np.array([]), y1=np.array([]))

    def test_raises_on_empty_y1(self) -> None:
        """Should raise InsufficientDataError if y1 is empty."""
        with pytest.raises(InsufficientDataError):
            PotentialOutcomes(y0=np.array([1.0]), y1=np.array([]))

    def test_raises_on_mismatched_unit_ids(self) -> None:
        """Should raise InsufficientDataError if unit_ids length differs."""
        with pytest.raises(InsufficientDataError):
            PotentialOutcomes(
                y0=np.array([1.0, 2.0]),
                y1=np.array([3.0, 4.0]),
                unit_ids=np.array(["a"]),
            )


class TestPotentialOutcomesProperties:
    """Tests for PotentialOutcomes computed properties."""

    def test_ate_is_correct(self) -> None:
        """ATE should equal mean(y1 - y0)."""
        y0 = np.array([1.0, 2.0, 3.0, 4.0])
        y1 = np.array([3.0, 4.0, 5.0, 6.0])
        po = PotentialOutcomes(y0=y0, y1=y1)
        expected_ate = float(np.mean(y1 - y0))
        assert po.ate == pytest.approx(expected_ate)

    def test_ite_is_correct(self) -> None:
        """ITE should equal y1 - y0 element-wise."""
        y0 = np.array([1.0, 2.0, 3.0])
        y1 = np.array([2.0, 5.0, 3.5])
        po = PotentialOutcomes(y0=y0, y1=y1)
        expected_ite = y1 - y0
        np.testing.assert_array_almost_equal(po.ite, expected_ite)

    def test_ite_has_same_shape_as_y0(self) -> None:
        """ITE should have the same shape as y0."""
        y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y1 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        po = PotentialOutcomes(y0=y0, y1=y1)
        assert po.ite.shape == y0.shape

    def test_n_returns_correct_count(self) -> None:
        """n should return the number of units."""
        po = PotentialOutcomes(y0=np.ones(10), y1=np.ones(10))
        assert po.n == 10

    def test_ate_with_zero_effect(self) -> None:
        """ATE should be 0 when y0 == y1."""
        y = np.array([1.0, 2.0, 3.0])
        po = PotentialOutcomes(y0=y, y1=y)
        assert po.ate == pytest.approx(0.0)


class TestPotentialOutcomesMethods:
    """Tests for PotentialOutcomes methods."""

    def test_to_dataframe_columns_without_ids(self) -> None:
        """DataFrame should have y0, y1, ite columns without unit_ids."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0]),
            y1=np.array([3.0, 4.0]),
        )
        df = po.to_dataframe()
        assert list(df.columns) == ["y0", "y1", "ite"]

    def test_to_dataframe_columns_with_ids(self) -> None:
        """DataFrame should have unit_id, y0, y1, ite columns with unit_ids."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0]),
            y1=np.array([3.0, 4.0]),
            unit_ids=np.array(["a", "b"]),
        )
        df = po.to_dataframe()
        assert list(df.columns) == ["unit_id", "y0", "y1", "ite"]

    def test_to_dataframe_values(self) -> None:
        """DataFrame values should match the potential outcomes."""
        y0 = np.array([1.0, 2.0])
        y1 = np.array([3.0, 5.0])
        po = PotentialOutcomes(y0=y0, y1=y1)
        df = po.to_dataframe()
        np.testing.assert_array_almost_equal(df["y0"].values, y0)
        np.testing.assert_array_almost_equal(df["y1"].values, y1)
        np.testing.assert_array_almost_equal(df["ite"].values, y1 - y0)

    def test_summary_returns_nonempty_string(self) -> None:
        """summary() should return a non-empty string."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0, 3.0]),
            y1=np.array([2.0, 3.0, 5.0]),
        )
        s = po.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_key_info(self) -> None:
        """summary() should contain N units, ATE, ITE std, min, max."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0, 3.0]),
            y1=np.array([2.0, 3.0, 5.0]),
        )
        s = po.summary()
        assert "N units" in s
        assert "ATE" in s
        assert "ITE std" in s
        assert "ITE min" in s
        assert "ITE max" in s

    def test_repr_format(self) -> None:
        """__repr__ should follow expected format."""
        po = PotentialOutcomes(
            y0=np.array([1.0, 2.0]),
            y1=np.array([3.0, 4.0]),
        )
        r = repr(po)
        assert r.startswith("PotentialOutcomes(n=")
        assert "ate=" in r