"""Multiple testing correction for p-values from estimators or inference.

Implements ``MultipleTestingCorrection``, a utility class that applies
Bonferroni, Holm, or Benjamini-Hochberg correction to a family of
p-values. Accepts either a multi-effect ``Results`` (typical output of
``FactorialEstimator`` after inference) or a list of scalar ``Results``
(typical when comparing multiple independent experiments).

The correction is purely post-processing: ``correct()`` produces new
``Results`` objects with adjusted ``p_value`` and ``alpha`` (set to
``self.alpha``); ``effects``, ``ate``, ``se``, and ``ci`` are preserved
unchanged.

References
----------
Bonferroni, C. E. (1936). Teoria statistica delle classi e calcolo delle
    probabilità. Pubblicazioni del R Istituto Superiore di Scienze
    Economiche e Commerciali di Firenze.
Holm, S. (1979). A simple sequentially rejective multiple test
    procedure. Scandinavian Journal of Statistics, 6(2), 65-70.
Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing. Journal
    of the Royal Statistical Society: Series B, 57(1), 289-300.
"""

import numpy as np

from skxperiments.core.exceptions import InvalidDesignError
from skxperiments.core.results import Results


def _apply_correction(
    p_values: np.ndarray,
    method: str,
    m: int,
) -> np.ndarray:
    """Apply the specified correction to a flat array of p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw p-values, shape (m,).
    method : str
        One of "bonferroni", "holm", "bh".
    m : int
        Family size. Equal to len(p_values), passed explicitly for
        clarity.

    Returns
    -------
    np.ndarray
        Corrected p-values, clipped to [0, 1], in the same order as
        the input.
    """
    if method == "bonferroni":
        return np.clip(p_values * m, 0.0, 1.0)

    # Holm and BH both need sort-then-unsort.
    order = np.argsort(p_values)
    p_sorted = p_values[order]

    if method == "holm":
        # p_holm[i] = p_sorted[i] * (m - i), then enforce
        # non-decreasing monotonicity (cumulative max).
        multipliers = m - np.arange(m)
        p_adj_sorted = p_sorted * multipliers
        p_adj_sorted = np.maximum.accumulate(p_adj_sorted)
    elif method == "bh":
        # p_bh[i] = p_sorted[i] * m / (i + 1), then enforce
        # non-increasing monotonicity from the right (cumulative min
        # reversed).
        ranks = np.arange(1, m + 1)
        p_adj_sorted = p_sorted * m / ranks
        # Reverse cumulative min.
        p_adj_sorted = np.minimum.accumulate(p_adj_sorted[::-1])[::-1]
    else:
        # Should be unreachable given __init__ validation.
        raise InvalidDesignError(
            f"Unknown correction method: {method!r}."
        )

    p_adj_sorted = np.clip(p_adj_sorted, 0.0, 1.0)

    # Reorder to original positions.
    p_corrected = np.empty_like(p_adj_sorted)
    p_corrected[order] = p_adj_sorted
    return p_corrected


class MultipleTestingCorrection:
    """Apply multiple-testing correction to a family of p-values.

    Accepts a multi-effect ``Results`` (with ``effects: dict`` and
    ``p_value: dict``) or a list of scalar ``Results`` (each with
    ``ate: float`` and ``p_value: float``). Returns the same format,
    with corrected p-values, ``alpha`` set to ``self.alpha``, and
    family-level metadata recorded in ``Results.extra``.

    Parameters
    ----------
    method : {"bonferroni", "holm", "bh"}, optional
        Correction method, by default ``"holm"``.

        - ``"bonferroni"`` and ``"holm"`` control the **family-wise
          error rate** (FWER): the probability of at least one false
          positive across the family. Holm uniformly dominates
          Bonferroni in power and is the recommended default.
        - ``"bh"`` (Benjamini-Hochberg) controls the **false discovery
          rate** (FDR): the expected proportion of false positives
          among rejections. FDR is a fundamentally different criterion
          from FWER; choose consciously based on the inferential goal.

    alpha : float, optional
        Family-wise alpha level, by default 0.05. Must be in (0, 1).
        Overrides the ``alpha`` of input ``Results`` objects in the
        output.

    Notes
    -----
    Reserved keys written to ``Results.extra`` (see ``Results``
    docstring for the full schema):

    - ``"correction_method"``
    - ``"original_p_values"`` (dict in multi-effect mode, list in
      scalar-list mode)
    - ``"family_wise_alpha"``
    - ``"n_tests"``

    Applying ``correct()`` twice to the same ``Results`` raises
    ``InvalidDesignError``: the presence of any of the four reserved
    keys in ``extra`` is treated as evidence of a prior correction.
    Apply correction to the original (uncorrected) ``Results`` instead.

    The correction does not touch ``effects``, ``ate``, ``se``, or
    ``ci``. Only ``p_value``, ``alpha``, and ``extra`` are modified
    in the output.

    Future work (v2)
    ----------------
    Benjamini-Yekutieli (``"by"``) for FDR under arbitrary dependence
    is deferred to v2. See ``ROADMAP.md``.

    Examples
    --------
    Multi-effect input (typical from FactorialEstimator):

    >>> from skxperiments.core.results import Results
    >>> r = Results(
    ...     effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1},
    ...     p_value={("A",): 0.01, ("B",): 0.04, ("A", "B"): 0.20},
    ... )
    >>> mtc = MultipleTestingCorrection(method="holm", alpha=0.05)
    >>> corrected = mtc.correct(r)  # doctest: +SKIP
    >>> corrected.p_value  # doctest: +SKIP

    Scalar-list input (typical when comparing experiments):

    >>> r1 = Results(ate=0.5, p_value=0.01)
    >>> r2 = Results(ate=0.3, p_value=0.04)
    >>> r3 = Results(ate=0.1, p_value=0.20)
    >>> corrected = mtc.correct([r1, r2, r3])  # doctest: +SKIP
    """

    _VALID_METHODS = ("bonferroni", "holm", "bh")
    _RESERVED_KEYS = (
        "correction_method",
        "original_p_values",
        "family_wise_alpha",
        "n_tests",
    )

    def __init__(
        self,
        method: str = "holm",
        alpha: float = 0.05,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise InvalidDesignError(
                f"method must be one of {self._VALID_METHODS}, got "
                f"{method!r}."
            )

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
            raise InvalidDesignError(
                f"alpha must be a float, got {type(alpha).__name__}."
            )
        if not (0.0 < float(alpha) < 1.0):
            raise InvalidDesignError(
                f"alpha must be in (0, 1), got {alpha}."
            )

        self.method = method
        self.alpha = float(alpha)

    def correct(
        self,
        results: Results | list[Results],
    ) -> Results | list[Results]:
        """Apply the configured correction to a Results or list of Results.

        Parameters
        ----------
        results : Results or list of Results
            Multi-effect ``Results`` (with ``p_value: dict``) or list
            of scalar ``Results`` (each with ``p_value: float``).

        Returns
        -------
        Results or list of Results
            Same format as input. Corrected p-values; ``alpha``
            overridden by ``self.alpha``; ``extra`` populated with
            family-level metadata.

        Raises
        ------
        InvalidDesignError
            If input format is invalid, p-values are missing, or any
            of the reserved keys are already present in
            ``Results.extra`` (indicating a prior correction).
        """
        if isinstance(results, list):
            return self._correct_list(results)
        return self._correct_multi_effect(results)

    def _correct_multi_effect(self, results: Results) -> Results:
        """Correct a single multi-effect Results."""
        if results.effects is None:
            raise InvalidDesignError(
                "MultipleTestingCorrection on a single Results requires "
                "multi-effect mode (Results.effects). For a list of "
                "scalar Results, pass them as a list."
            )

        if results.p_value is None or not isinstance(results.p_value, dict):
            raise InvalidDesignError(
                "p_value dict is missing or not a dict; cannot correct."
            )

        self._check_no_reserved_keys(results.extra, location="results")

        # Sort keys alphabetically for defensive determinism.
        keys_sorted = sorted(results.p_value.keys())
        p_array = np.array(
            [results.p_value[k] for k in keys_sorted], dtype=float
        )
        m = len(p_array)

        p_corrected = _apply_correction(p_array, self.method, m=m)

        p_value_corrected = {
            k: float(p) for k, p in zip(keys_sorted, p_corrected)
        }

        new_extra = dict(results.extra) if results.extra is not None else {}
        new_extra["correction_method"] = self.method
        new_extra["original_p_values"] = dict(results.p_value)
        new_extra["family_wise_alpha"] = self.alpha
        new_extra["n_tests"] = m

        return Results(
            effects=dict(results.effects),
            p_value=p_value_corrected,
            se=results.se,
            ci=results.ci,
            alpha=self.alpha,
            n_obs=results.n_obs,
            n_treated=results.n_treated,
            n_control=results.n_control,
            estimator_name=results.estimator_name,
            design_name=results.design_name,
            inference_name=results.inference_name,
            extra=new_extra,
        )

    def _correct_list(self, results: list[Results]) -> list[Results]:
        """Correct a list of scalar Results."""
        if len(results) == 0:
            raise InvalidDesignError(
                "MultipleTestingCorrection received an empty list; "
                "nothing to correct."
            )

        for i, r in enumerate(results):
            if not isinstance(r, Results):
                raise InvalidDesignError(
                    f"List element at index {i} is not a Results "
                    f"instance, got {type(r).__name__}."
                )
            if r.ate is None or r.effects is not None:
                raise InvalidDesignError(
                    f"List element at index {i} is not in scalar mode "
                    f"(Results.ate must be set, Results.effects must "
                    f"be None). Mixed lists are not supported."
                )
            if r.p_value is None or not isinstance(r.p_value, (int, float)):
                raise InvalidDesignError(
                    f"List element at index {i} has missing or "
                    f"non-scalar p_value; cannot correct."
                )
            self._check_no_reserved_keys(
                r.extra, location=f"results[{i}]"
            )

        p_array = np.array([r.p_value for r in results], dtype=float)
        m = len(p_array)

        p_corrected = _apply_correction(p_array, self.method, m=m)

        original_p_list = [float(p) for p in p_array]

        out: list[Results] = []
        for i, r in enumerate(results):
            new_extra = dict(r.extra) if r.extra is not None else {}
            new_extra["correction_method"] = self.method
            new_extra["original_p_values"] = list(original_p_list)
            new_extra["family_wise_alpha"] = self.alpha
            new_extra["n_tests"] = m

            out.append(
                Results(
                    ate=r.ate,
                    p_value=float(p_corrected[i]),
                    se=r.se,
                    ci=r.ci,
                    alpha=self.alpha,
                    n_obs=r.n_obs,
                    n_treated=r.n_treated,
                    n_control=r.n_control,
                    estimator_name=r.estimator_name,
                    design_name=r.design_name,
                    inference_name=r.inference_name,
                    extra=new_extra,
                )
            )

        return out

    def _check_no_reserved_keys(
        self,
        extra: dict | None,
        location: str,
    ) -> None:
        """Reject Results whose extra already contains reserved keys.

        Detects an earlier correction (any of the 4 reserved keys
        present) and refuses to apply correction a second time.

        Parameters
        ----------
        extra : dict or None
            The ``Results.extra`` dict to inspect.
        location : str
            Human-readable location string used in the error message
            (e.g., "results" or "results[3]").
        """
        if extra is None:
            return
        present = [k for k in self._RESERVED_KEYS if k in extra]
        if present:
            raise InvalidDesignError(
                f"{location}.extra already contains reserved key(s) "
                f"{present!r}; cannot apply MultipleTestingCorrection "
                f"twice. Apply correction to the original "
                f"(uncorrected) Results."
            )