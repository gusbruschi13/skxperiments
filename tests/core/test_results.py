"""Tests for skxperiments.core.results."""

import pandas as pd
import pytest

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


# --- Scalar mode ---


class TestResultsScalarCreation:
    """Tests for Results instantiation in scalar mode."""

    def test_basic_creation(self) -> None:
        """Should create Results with just ate."""
        result = Results(ate=0.5)
        assert result.ate == 0.5
        assert result.effects is None
        assert result.is_multi_effect is False

    def test_full_creation(self) -> None:
        """Should create Results with all parameters."""
        result = Results(
            ate=0.142,
            se=0.056,
            ci=(0.031, 0.253),
            p_value=0.011,
            alpha=0.05,
            n_obs=1000,
            n_treated=500,
            n_control=500,
            estimator_name="DifferenceInMeans",
            design_name="CRD",
            inference_name="RandomizationTest",
            extra={"n_permutations": 10000},
        )
        assert result.ate == 0.142
        assert result.ci == (0.031, 0.253)


class TestResultsScalarValidation:
    """Tests for scalar-mode validation."""

    def test_ate_must_be_numeric(self) -> None:
        with pytest.raises(TypeError):
            Results(ate="not a number")  # type: ignore[arg-type]

    def test_ci_must_be_tuple_of_two(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, ci=(0.1,))  # type: ignore[arg-type]

    def test_ci_lower_must_be_le_upper(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, ci=(0.5, 0.1))

    def test_p_value_must_be_in_range(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, p_value=1.5)

    def test_p_value_negative_raises(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, p_value=-0.1)

    def test_alpha_must_be_in_open_interval(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, alpha=0.0)

    def test_alpha_one_raises(self) -> None:
        with pytest.raises(InvalidDesignError):
            Results(ate=0.5, alpha=1.0)


class TestResultsMutualExclusivity:
    """Tests that ate and effects are mutually exclusive."""

    def test_neither_ate_nor_effects_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="exactly one"):
            Results()

    def test_both_ate_and_effects_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="exactly one"):
            Results(ate=0.5, effects={("A",): 0.3})


class TestResultsIsSignificantScalar:
    """Tests for is_significant in scalar mode."""

    def test_significant_when_p_below_alpha(self) -> None:
        result = Results(ate=0.5, p_value=0.01, alpha=0.05)
        assert result.is_significant() is True

    def test_not_significant_when_p_above_alpha(self) -> None:
        result = Results(ate=0.5, p_value=0.10, alpha=0.05)
        assert result.is_significant() is False

    def test_not_significant_when_p_is_none(self) -> None:
        result = Results(ate=0.5)
        assert result.is_significant() is False

    def test_not_significant_at_boundary(self) -> None:
        result = Results(ate=0.5, p_value=0.05, alpha=0.05)
        assert result.is_significant() is False

    def test_key_in_scalar_mode_raises(self) -> None:
        result = Results(ate=0.5, p_value=0.01)
        with pytest.raises(InvalidDesignError, match="does not accept a key"):
            result.is_significant(key=("A",))


class TestResultsScalarToDict:
    """Tests for to_dict in scalar mode."""

    def test_omits_none_values(self) -> None:
        result = Results(ate=0.5)
        d = result.to_dict()
        assert "se" not in d
        assert "ci" not in d
        assert "p_value" not in d
        assert "n_obs" not in d

    def test_includes_non_none_values(self) -> None:
        result = Results(ate=0.5, se=0.1, p_value=0.03)
        d = result.to_dict()
        assert d["ate"] == 0.5
        assert d["se"] == 0.1
        assert d["p_value"] == 0.03

    def test_extra_expanded_as_top_level(self) -> None:
        result = Results(
            ate=0.5, extra={"n_permutations": 1000, "method": "fisher"}
        )
        d = result.to_dict()
        assert d["n_permutations"] == 1000
        assert d["method"] == "fisher"
        assert "extra" not in d

    def test_alpha_included(self) -> None:
        result = Results(ate=0.5)
        d = result.to_dict()
        assert d["alpha"] == 0.05


class TestResultsScalarToDataframe:
    """Tests for to_dataframe in scalar mode."""

    def test_returns_single_row(self) -> None:
        result = Results(ate=0.5, se=0.1)
        df = result.to_dataframe()
        assert len(df) == 1


class TestResultsScalarRepr:
    def test_repr_format(self) -> None:
        result = Results(ate=0.5, ci=(0.1, 0.9), p_value=0.03)
        r = repr(result)
        assert "Results(" in r
        assert "ate=0.5" in r
        assert "ci=(0.1, 0.9)" in r
        assert "p_value=0.03" in r


# --- Multi-effect mode ---


class TestResultsMultiEffectCreation:
    """Tests for Results instantiation in multi-effect mode."""

    def test_basic_creation(self) -> None:
        result = Results(
            effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1}
        )
        assert result.ate is None
        assert result.is_multi_effect is True
        assert result.effects[("A",)] == 0.5
        assert result.effects[("A", "B")] == 0.1

    def test_with_se_and_p_value(self) -> None:
        result = Results(
            effects={("A",): 0.5, ("B",): 0.3},
            se={("A",): 0.05, ("B",): 0.04},
            p_value={("A",): 0.001, ("B",): 0.05},
        )
        assert result.se[("A",)] == 0.05
        assert result.p_value[("B",)] == 0.05


class TestResultsMultiEffectValidation:
    """Tests for multi-effect mode validation."""

    def test_empty_effects_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="non-empty"):
            Results(effects={})

    def test_non_tuple_key_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="tuples of strings"):
            Results(effects={"A": 0.5})  # type: ignore[dict-item]

    def test_non_string_in_tuple_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="tuples of strings"):
            Results(effects={(1,): 0.5})  # type: ignore[dict-item]

    def test_non_numeric_value_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="numeric"):
            Results(effects={("A",): "not a number"})  # type: ignore[dict-item]

    def test_p_value_dict_unknown_key_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="not in effects"):
            Results(
                effects={("A",): 0.5},
                p_value={("Z",): 0.01},
            )

    def test_p_value_dict_out_of_range_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match=r"\[0\.0, 1\.0\]"):
            Results(
                effects={("A",): 0.5},
                p_value={("A",): 1.5},
            )

    def test_ci_dict_invalid_tuple_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="tuple of two"):
            Results(
                effects={("A",): 0.5},
                ci={("A",): (0.1,)},  # type: ignore[dict-item]
            )

    def test_ci_dict_lower_gt_upper_raises(self) -> None:
        with pytest.raises(InvalidDesignError, match="<= upper"):
            Results(
                effects={("A",): 0.5},
                ci={("A",): (0.9, 0.1)},
            )

    def test_p_value_must_be_dict_in_multi_mode(self) -> None:
        with pytest.raises(InvalidDesignError, match="dict in multi"):
            Results(
                effects={("A",): 0.5},
                p_value=0.05,  # type: ignore[arg-type]
            )

    def test_se_must_be_dict_in_multi_mode(self) -> None:
        with pytest.raises(InvalidDesignError, match="dict in multi"):
            Results(
                effects={("A",): 0.5},
                se=0.1,  # type: ignore[arg-type]
            )


class TestResultsIsSignificantMultiEffect:
    """Tests for is_significant in multi-effect mode."""

    def test_significant_with_key(self) -> None:
        result = Results(
            effects={("A",): 0.5, ("B",): 0.3},
            p_value={("A",): 0.001, ("B",): 0.10},
            alpha=0.05,
        )
        assert result.is_significant(("A",)) is True
        assert result.is_significant(("B",)) is False

    def test_no_key_in_multi_mode_raises(self) -> None:
        result = Results(effects={("A",): 0.5})
        with pytest.raises(InvalidDesignError, match="requires a key"):
            result.is_significant()

    def test_unknown_key_raises(self) -> None:
        result = Results(
            effects={("A",): 0.5}, p_value={("A",): 0.01}
        )
        with pytest.raises(InvalidDesignError, match="not in effects"):
            result.is_significant(("Z",))

    def test_returns_false_when_no_p_values(self) -> None:
        result = Results(effects={("A",): 0.5})
        assert result.is_significant(("A",)) is False


class TestResultsMultiEffectToDataframe:
    """Tests for to_dataframe in multi-effect mode."""

    def test_one_row_per_effect(self) -> None:
        result = Results(
            effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1}
        )
        df = result.to_dataframe()
        assert len(df) == 3

    def test_columns_include_effect_and_estimate(self) -> None:
        result = Results(effects={("A",): 0.5})
        df = result.to_dataframe()
        assert "effect" in df.columns
        assert "estimate" in df.columns

    def test_se_and_p_value_columns_when_present(self) -> None:
        result = Results(
            effects={("A",): 0.5},
            se={("A",): 0.05},
            p_value={("A",): 0.01},
        )
        df = result.to_dataframe()
        assert "se" in df.columns
        assert "p_value" in df.columns

    def test_ci_split_into_lower_upper(self) -> None:
        result = Results(
            effects={("A",): 0.5},
            ci={("A",): (0.4, 0.6)},
        )
        df = result.to_dataframe()
        assert "ci_lower" in df.columns
        assert "ci_upper" in df.columns
        assert df.iloc[0]["ci_lower"] == 0.4
        assert df.iloc[0]["ci_upper"] == 0.6


class TestResultsMultiEffectRepr:
    def test_repr_shows_effect_count(self) -> None:
        result = Results(
            effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1}
        )
        r = repr(result)
        assert "effects=3 keys" in r


# --- Common ---


class TestResultsToMarkdown:
    """Tests for to_markdown."""

    def test_scalar_returns_string(self) -> None:
        result = Results(ate=0.5)
        md = result.to_markdown()
        assert isinstance(md, str)
        assert "ATE" in md

    def test_multi_effect_returns_string(self) -> None:
        result = Results(effects={("A",): 0.5, ("B",): 0.3})
        md = result.to_markdown()
        assert isinstance(md, str)
        assert "effect" in md or "estimate" in md


class TestResultsSummary:
    def test_returns_self_scalar(self, capsys: pytest.CaptureFixture) -> None:
        result = Results(ate=0.5)
        returned = result.summary()
        assert returned is result

    def test_returns_self_multi_effect(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        result = Results(effects={("A",): 0.5})
        returned = result.summary()
        assert returned is result

    def test_chainable(self, capsys: pytest.CaptureFixture) -> None:
        result = Results(ate=0.5, p_value=0.03)
        sig = result.summary().is_significant()
        assert sig is True