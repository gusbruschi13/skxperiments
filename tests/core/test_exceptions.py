"""Tests for skxperiments.core.exceptions."""

import pytest

from skxperiments.core.exceptions import (
    DesignEstimatorMismatch,
    InsufficientDataError,
    InvalidDesignError,
    NotFittedError,
    SkxperimentsError,
)


class TestSkxperimentsError:
    """Tests for the base exception class."""

    def test_inherits_from_exception(self) -> None:
        """SkxperimentsError should inherit from Exception."""
        assert issubclass(SkxperimentsError, Exception)

    def test_message_stored(self) -> None:
        """Message should be stored as attribute."""
        err = SkxperimentsError("test message")
        assert err.message == "test message"

    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable with except Exception."""
        with pytest.raises(Exception):
            raise SkxperimentsError("test")


class TestDesignEstimatorMismatch:
    """Tests for DesignEstimatorMismatch."""

    def test_inherits_from_base(self) -> None:
        """Should inherit from SkxperimentsError."""
        assert issubclass(DesignEstimatorMismatch, SkxperimentsError)

    def test_message_without_suggestion(self) -> None:
        """Message should be formatted correctly without suggestion."""
        err = DesignEstimatorMismatch(
            estimator_name="DifferenceInMeans",
            received_type="BlockedAssignment",
            expected_type="CRDAssignment",
        )
        assert "[DifferenceInMeans]" in str(err)
        assert "CRDAssignment" in str(err)
        assert "BlockedAssignment" in str(err)
        assert "Suggestion" not in str(err)

    def test_message_with_suggestion(self) -> None:
        """Message should include suggestion when provided."""
        err = DesignEstimatorMismatch(
            estimator_name="DifferenceInMeans",
            received_type="BlockedAssignment",
            expected_type="CRDAssignment",
            suggestion="BlockedDifferenceInMeans",
        )
        assert "Suggestion: use BlockedDifferenceInMeans instead." in str(err)

    def test_attributes_stored(self) -> None:
        """All init parameters should be stored as attributes."""
        err = DesignEstimatorMismatch(
            estimator_name="A",
            received_type="B",
            expected_type="C",
            suggestion="D",
        )
        assert err.estimator_name == "A"
        assert err.received_type == "B"
        assert err.expected_type == "C"
        assert err.suggestion == "D"

    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable with except Exception."""
        with pytest.raises(Exception):
            raise DesignEstimatorMismatch(
                estimator_name="X",
                received_type="Y",
                expected_type="Z",
            )


class TestNotFittedError:
    """Tests for NotFittedError."""

    def test_inherits_from_base(self) -> None:
        """Should inherit from SkxperimentsError."""
        assert issubclass(NotFittedError, SkxperimentsError)

    def test_message_format(self) -> None:
        """Message should follow expected format."""
        err = NotFittedError(
            class_name="DifferenceInMeans",
            required_methods=["fit"],
        )
        assert "[DifferenceInMeans]" in str(err)
        assert "is not fitted" in str(err)
        assert "fit()" in str(err)

    def test_multiple_methods(self) -> None:
        """Message should list multiple methods."""
        err = NotFittedError(
            class_name="MyEstimator",
            required_methods=["fit", "transform"],
        )
        assert "fit()" in str(err)
        assert "transform()" in str(err)

    def test_attributes_stored(self) -> None:
        """All init parameters should be stored."""
        err = NotFittedError(class_name="X", required_methods=["fit"])
        assert err.class_name == "X"
        assert err.required_methods == ["fit"]

    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable with except Exception."""
        with pytest.raises(Exception):
            raise NotFittedError(class_name="X", required_methods=["fit"])


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""

    def test_inherits_from_base(self) -> None:
        """Should inherit from SkxperimentsError."""
        assert issubclass(InsufficientDataError, SkxperimentsError)

    def test_message_format(self) -> None:
        """Message should follow expected format."""
        err = InsufficientDataError(
            context="CRD randomization",
            minimum=2,
            received=1,
        )
        assert "CRD randomization" in str(err)
        assert "at least 2" in str(err)
        assert "received 1" in str(err)

    def test_attributes_stored(self) -> None:
        """All init parameters should be stored."""
        err = InsufficientDataError(context="test", minimum=5, received=3)
        assert err.context == "test"
        assert err.minimum == 5
        assert err.received == 3

    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable with except Exception."""
        with pytest.raises(Exception):
            raise InsufficientDataError(context="x", minimum=1, received=0)


class TestInvalidDesignError:
    """Tests for InvalidDesignError."""

    def test_inherits_from_base(self) -> None:
        """Should inherit from SkxperimentsError."""
        assert issubclass(InvalidDesignError, SkxperimentsError)

    def test_message_passthrough(self) -> None:
        """Message should be passed through directly."""
        err = InvalidDesignError("Custom error message")
        assert str(err) == "Custom error message"

    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable with except Exception."""
        with pytest.raises(Exception):
            raise InvalidDesignError("test")