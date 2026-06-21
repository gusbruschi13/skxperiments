"""Uniform results object for all estimators and inference methods.

Every estimator and inference method in skxperiments returns a Results
object, providing a consistent interface for accessing estimates,
confidence intervals, p-values, and metadata.

A Results instance carries either a single ATE (scalar estimand) or a
dict of named effects (e.g., main effects and interactions from a
factorial design). Exactly one of ``ate`` or ``effects`` must be
provided.
"""

import pandas as pd

from skxperiments.core.exceptions import InvalidDesignError


# Type alias for effect keys: tuples of factor names.
# ("A",) -> main effect of A
# ("A", "B") -> AB interaction
EffectKey = tuple[str, ...]


class Results:
    """Uniform output object for estimators and inference methods.

    Carries either a single scalar ATE or a dict of named effects.
    Exactly one of ``ate`` or ``effects`` must be provided.

    Parameters
    ----------
    ate : float or None, optional
        Point estimate of the Average Treatment Effect, for scalar
        estimands. Mutually exclusive with ``effects``. By default None.
    effects : dict or None, optional
        Mapping of effect-name tuples to point estimates, for
        multi-effect estimands (e.g., FactorialEstimator). Keys are
        tuples of factor names: ``("A",)`` for main effect of A,
        ``("A", "B")`` for AB interaction. Mutually exclusive with
        ``ate``. By default None.
    se : float or dict or None, optional
        Standard error. ``float`` when ``ate`` is set; ``dict`` keyed
        like ``effects`` when ``effects`` is set. By default None.
    ci : tuple or dict or None, optional
        Confidence interval. ``(lower, upper)`` when ``ate`` is set;
        ``dict`` keyed like ``effects`` mapping to ``(lower, upper)``
        tuples when ``effects`` is set. By default None.
    p_value : float or dict or None, optional
        P-value. ``float`` when ``ate`` is set; ``dict`` keyed like
        ``effects`` when ``effects`` is set. By default None.
    alpha : float, optional
        Significance level, by default 0.05.
    n_obs : int or None, optional
        Total number of observations, by default None.
    n_treated : int or None, optional
        Number of treated units, by default None.
    n_control : int or None, optional
        Number of control units, by default None.
    estimator_name : str or None, optional
        Name of the estimator used, by default None.
    design_name : str or None, optional
        Name of the experimental design, by default None.
    inference_name : str or None, optional
        Name of the inference method, by default None.
    extra : dict or None, optional
        Additional metadata as key-value pairs (e.g., n_permutations,
        convergence info). Not the primary result. By default None.
        See Notes for the schema of reserved keys.

    Raises
    ------
    InvalidDesignError
        If neither or both of ``ate`` and ``effects`` are provided,
        or if shape/value validations fail on ci, p_value, or alpha.
    TypeError
        If ``ate`` is provided but is not numeric.

    Notes
    -----
    Reserved keys in ``extra``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    The following keys are reserved by skxperiments components.
    Custom metadata may use any other key.

    *Written by estimators (Phase 3):*

    - ``"inference_mode"`` : str, ``"finite_population"`` or
      ``"superpopulation"``. Documentational metadata propagated by
      ``LinEstimator``. Read by inference classes in Phase 4.
    - ``"theta"`` : float. CUPED adjustment coefficient
      ``Cov(Y, X_pre) / Var(X_pre)``.
    - ``"correlation"`` : float. Pearson correlation between outcome
      and pre-period covariate, written by ``CUPED``.

    *Written by inference classes (Phase 4):*

    - ``"n_permutations"`` : int. Number of permutations used by
      ``RandomizationTest``.
    - ``"null_distribution"`` : np.ndarray. Array of permuted
      statistics under the sharp null, written by
      ``RandomizationTest``. Length equals ``n_permutations``.
    - ``"alternative"`` : str. Alternative hypothesis used by
      ``RandomizationTest``: ``"two-sided"``, ``"greater"``, or
      ``"less"``.

    Examples
    --------
    Scalar estimand:

    >>> r = Results(ate=0.142, se=0.056, p_value=0.011)
    >>> r.is_significant()
    True

    Multi-effect estimand:

    >>> r = Results(
    ...     effects={("A",): 0.5, ("B",): 0.3, ("A", "B"): 0.1},
    ...     p_value={("A",): 0.01, ("B",): 0.04, ("A", "B"): 0.20},
    ... )
    >>> r.is_significant(("A",))
    True

    *Written by inference classes (Phase 4.2):*
    
        - ``"correction_method"`` : str, ``"bonferroni"``, ``"holm"``, or
          ``"bh"``. The multiple-testing correction applied by
          ``MultipleTestingCorrection``.
        - ``"original_p_values"`` : dict or list. The uncorrected p-values
          before applying the correction. Same structure as the corrected
          ``p_value``: dict in multi-effect mode, list in scalar-list mode.
        - ``"family_wise_alpha"`` : float. The family-wise alpha used by
          ``MultipleTestingCorrection``.
        - ``"n_tests"`` : int. The size of the testing family.

    *Written by inference classes (Phase 4.3):*

        - ``"variance_type"`` : str, ``"neyman"`` or
          ``"neyman_stratified"``. The variance estimator used by
          ``NeymanCI``: ``"neyman"`` for CRD (two-sample conservative
          variance) and ``"neyman_stratified"`` for blocked designs
          (size-weighted sum of within-block variances). ``NeymanCI``
          also propagates ``"inference_mode"`` (see Phase 3).
    """

    def __init__(
        self,
        *,
        ate: float | None = None,
        effects: dict[EffectKey, float] | None = None,
        se: float | dict | None = None,
        ci: tuple[float, float] | dict | None = None,
        p_value: float | dict | None = None,
        alpha: float = 0.05,
        n_obs: int | None = None,
        n_treated: int | None = None,
        n_control: int | None = None,
        estimator_name: str | None = None,
        design_name: str | None = None,
        inference_name: str | None = None,
        extra: dict | None = None,
    ) -> None:
        # --- Mutual exclusivity ---
        if ate is None and effects is None:
            raise InvalidDesignError(
                "Results requires exactly one of ate or effects; "
                "both are None."
            )
        if ate is not None and effects is not None:
            raise InvalidDesignError(
                "Results requires exactly one of ate or effects; "
                "both were provided."
            )

        # --- Validate ate (scalar mode) ---
        if ate is not None:
            if not isinstance(ate, (int, float)):
                raise TypeError(
                    f"ate must be int or float, but received "
                    f"{type(ate).__name__}."
                )

        # --- Validate effects (dict mode) ---
        if effects is not None:
            if not isinstance(effects, dict) or len(effects) == 0:
                raise InvalidDesignError(
                    "effects must be a non-empty dict mapping "
                    "tuple[str, ...] to float."
                )
            for key, value in effects.items():
                if not isinstance(key, tuple) or not all(
                    isinstance(k, str) for k in key
                ):
                    raise InvalidDesignError(
                        f"effects keys must be tuples of strings; "
                        f"got key {key!r}."
                    )
                if not isinstance(value, (int, float)):
                    raise InvalidDesignError(
                        f"effects values must be numeric; got "
                        f"{type(value).__name__} for key {key!r}."
                    )

        # --- Validate ci ---
        if ci is not None:
            if effects is None:
                # Scalar mode: ci must be a 2-tuple of numbers.
                if (
                    not isinstance(ci, tuple)
                    or len(ci) != 2
                    or not all(isinstance(b, (int, float)) for b in ci)
                ):
                    raise InvalidDesignError(
                        "ci must be a tuple of two floats (lower, upper) "
                        "in scalar mode."
                    )
                if ci[0] > ci[1]:
                    raise InvalidDesignError(
                        f"ci lower bound ({ci[0]}) must be <= upper "
                        f"bound ({ci[1]})."
                    )
            else:
                # Multi-effect mode: ci must be dict keyed like effects.
                if not isinstance(ci, dict):
                    raise InvalidDesignError(
                        "ci must be a dict in multi-effect mode."
                    )
                for key, value in ci.items():
                    if key not in effects:
                        raise InvalidDesignError(
                            f"ci key {key!r} not in effects."
                        )
                    if (
                        not isinstance(value, tuple)
                        or len(value) != 2
                        or not all(isinstance(b, (int, float)) for b in value)
                    ):
                        raise InvalidDesignError(
                            f"ci[{key!r}] must be a tuple of two floats."
                        )
                    if value[0] > value[1]:
                        raise InvalidDesignError(
                            f"ci[{key!r}] lower bound ({value[0]}) must "
                            f"be <= upper bound ({value[1]})."
                        )

        # --- Validate p_value ---
        if p_value is not None:
            if effects is None:
                if not isinstance(p_value, (int, float)):
                    raise InvalidDesignError(
                        f"p_value must be a float in scalar mode, "
                        f"received {type(p_value).__name__}."
                    )
                if not (0.0 <= p_value <= 1.0):
                    raise InvalidDesignError(
                        f"p_value must be in [0.0, 1.0], received {p_value}."
                    )
            else:
                if not isinstance(p_value, dict):
                    raise InvalidDesignError(
                        "p_value must be a dict in multi-effect mode."
                    )
                for key, value in p_value.items():
                    if key not in effects:
                        raise InvalidDesignError(
                            f"p_value key {key!r} not in effects."
                        )
                    if not isinstance(value, (int, float)):
                        raise InvalidDesignError(
                            f"p_value[{key!r}] must be numeric."
                        )
                    if not (0.0 <= value <= 1.0):
                        raise InvalidDesignError(
                            f"p_value[{key!r}] must be in [0.0, 1.0], "
                            f"received {value}."
                        )

        # --- Validate se (no range constraint, just type) ---
        if se is not None:
            if effects is None:
                if not isinstance(se, (int, float)):
                    raise InvalidDesignError(
                        f"se must be a float in scalar mode, received "
                        f"{type(se).__name__}."
                    )
            else:
                if not isinstance(se, dict):
                    raise InvalidDesignError(
                        "se must be a dict in multi-effect mode."
                    )
                for key, value in se.items():
                    if key not in effects:
                        raise InvalidDesignError(
                            f"se key {key!r} not in effects."
                        )
                    if not isinstance(value, (int, float)):
                        raise InvalidDesignError(
                            f"se[{key!r}] must be numeric."
                        )

        # --- Validate alpha ---
        if not isinstance(alpha, (int, float)):
            raise InvalidDesignError(
                f"alpha must be a float, but received {type(alpha).__name__}."
            )
        if not (0.0 < alpha < 1.0):
            raise InvalidDesignError(
                f"alpha must be in (0.0, 1.0), but received {alpha}."
            )

        # --- Store ---
        self.ate = ate
        self.effects = effects
        self.se = se
        self.ci = ci
        self.p_value = p_value
        self.alpha = alpha
        self.n_obs = n_obs
        self.n_treated = n_treated
        self.n_control = n_control
        self.estimator_name = estimator_name
        self.design_name = design_name
        self.inference_name = inference_name
        self.extra = extra

    @property
    def is_multi_effect(self) -> bool:
        """Whether this Results carries multiple named effects."""
        return self.effects is not None

    def to_dict(self) -> dict:
        """Convert results to a dictionary.

        In scalar mode, includes ``ate`` and other top-level fields.
        In multi-effect mode, includes ``effects`` (and matching ``se``,
        ``ci``, ``p_value`` dicts) instead of ``ate``.

        Returns
        -------
        dict
            All non-None attributes. ``extra`` keys are flattened to
            top level.
        """
        result: dict = {}

        if self.ate is not None:
            result["ate"] = self.ate
        if self.effects is not None:
            result["effects"] = self.effects

        for attr in [
            "se", "ci", "p_value", "alpha",
            "n_obs", "n_treated", "n_control",
            "estimator_name", "design_name", "inference_name",
        ]:
            value = getattr(self, attr)
            if value is not None:
                result[attr] = value

        if self.extra is not None:
            for key, value in self.extra.items():
                result[key] = value

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        In scalar mode, returns a single-row DataFrame.
        In multi-effect mode, returns one row per effect with columns
        ``effect``, ``estimate``, and (when present) ``se``, ``ci_lower``,
        ``ci_upper``, ``p_value``. Metadata columns (n_obs,
        estimator_name, etc.) are repeated across rows.

        Returns
        -------
        pd.DataFrame
        """
        if self.ate is not None:
            return pd.DataFrame([self.to_dict()])

        # Multi-effect: one row per effect.
        rows = []
        for key, estimate in self.effects.items():  # type: ignore[union-attr]
            row: dict = {
                "effect": key,
                "estimate": estimate,
            }
            if self.se is not None:
                row["se"] = self.se.get(key)  # type: ignore[union-attr]
            if self.ci is not None:
                ci_val = self.ci.get(key)  # type: ignore[union-attr]
                if ci_val is not None:
                    row["ci_lower"] = ci_val[0]
                    row["ci_upper"] = ci_val[1]
            if self.p_value is not None:
                row["p_value"] = self.p_value.get(key)  # type: ignore[union-attr]

            for attr in [
                "alpha", "n_obs", "n_treated", "n_control",
                "estimator_name", "design_name", "inference_name",
            ]:
                value = getattr(self, attr)
                if value is not None:
                    row[attr] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Generate a markdown table of results.

        Returns
        -------
        str
        """
        if self.ate is not None:
            return self._to_markdown_scalar()
        return self._to_markdown_multi()

    def _to_markdown_scalar(self) -> str:
        rows: list[tuple[str, str]] = [("ATE", f"{self.ate:.4f}")]

        if self.se is not None:
            rows.append(("SE", f"{self.se:.4f}"))
        if self.ci is not None:
            ci_pct = int((1 - self.alpha) * 100)
            rows.append(
                (f"{ci_pct}% CI", f"[{self.ci[0]:.3f}, {self.ci[1]:.3f}]")
            )
        if self.p_value is not None:
            rows.append(("p-value", f"{self.p_value:.3f}"))
        if self.n_obs is not None:
            rows.append(("N", str(self.n_obs)))
        if self.estimator_name is not None:
            rows.append(("Estimator", self.estimator_name))
        if self.design_name is not None:
            rows.append(("Design", self.design_name))
        if self.inference_name is not None:
            rows.append(("Inference", self.inference_name))

        return self._render_md_table(rows, ("Metric", "Value"))

    def _to_markdown_multi(self) -> str:
        df = self.to_dataframe()
        # Build a markdown table from the DataFrame's effect rows only.
        cols = ["effect", "estimate"]
        if "se" in df.columns:
            cols.append("se")
        if "ci_lower" in df.columns:
            cols.append("ci_lower")
            cols.append("ci_upper")
        if "p_value" in df.columns:
            cols.append("p_value")

        rows: list[tuple[str, ...]] = []
        for _, r in df.iterrows():
            row: list[str] = []
            for c in cols:
                v = r[c]
                if isinstance(v, tuple):
                    row.append(":".join(v))
                elif isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            rows.append(tuple(row))

        return self._render_md_table(rows, tuple(cols))

    @staticmethod
    def _render_md_table(
        rows: list[tuple],
        header: tuple[str, ...],
    ) -> str:
        n_cols = len(header)
        widths = [
            max(
                len(header[i]),
                max((len(r[i]) for r in rows), default=0),
            )
            for i in range(n_cols)
        ]
        sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
        head = (
            "| "
            + " | ".join(f"{header[i]:<{widths[i]}}" for i in range(n_cols))
            + " |"
        )
        body = [
            "| "
            + " | ".join(f"{r[i]:<{widths[i]}}" for i in range(n_cols))
            + " |"
            for r in rows
        ]
        return "\n".join([head, sep] + body)

    def is_significant(self, key: EffectKey | None = None) -> bool:
        """Check whether a result is statistically significant.

        Parameters
        ----------
        key : tuple of str or None, optional
            In scalar mode, must be None.
            In multi-effect mode, the effect key to check. Required.

        Returns
        -------
        bool
            True if p_value (for the given key, if applicable) is not
            None and is strictly less than alpha. False otherwise.

        Raises
        ------
        InvalidDesignError
            If ``key`` is provided in scalar mode, or if ``key`` is
            None in multi-effect mode, or if ``key`` is not in
            ``effects``.
        """
        if self.ate is not None:
            if key is not None:
                raise InvalidDesignError(
                    "is_significant in scalar mode does not accept a key."
                )
            if self.p_value is None:
                return False
            return self.p_value < self.alpha

        # Multi-effect mode.
        if key is None:
            raise InvalidDesignError(
                "is_significant in multi-effect mode requires a key."
            )
        if key not in self.effects:  # type: ignore[union-attr]
            raise InvalidDesignError(
                f"key {key!r} not in effects."
            )
        if self.p_value is None:
            return False
        pv = self.p_value.get(key)  # type: ignore[union-attr]
        if pv is None:
            return False
        return pv < self.alpha

    def summary(self) -> "Results":
        """Print a formatted summary table and return self.

        Returns
        -------
        Results
            Returns self for method chaining.
        """
        lines: list[str] = ["Results", "-------"]

        if self.ate is not None:
            lines.append(f"ATE            {self.ate:.4f}")
            if self.se is not None:
                lines.append(f"SE             {self.se:.4f}")
            if self.ci is not None:
                ci_pct = int((1 - self.alpha) * 100)
                lines.append(
                    f"{ci_pct}% CI         [{self.ci[0]:.3f}, "
                    f"{self.ci[1]:.3f}]"
                )
            if self.p_value is not None:
                lines.append(f"p-value        {self.p_value:.4f}")
                sig = "Yes" if self.is_significant() else "No"
                lines.append(f"Significant    {sig}")
        else:
            lines.append("Effects:")
            for key, value in self.effects.items():  # type: ignore[union-attr]
                key_str = ":".join(key)
                line = f"  {key_str:<20} {value:.4f}"
                if (
                    self.p_value is not None
                    and key in self.p_value  # type: ignore[union-attr]
                ):
                    pv = self.p_value[key]  # type: ignore[union-attr]
                    sig = "*" if pv < self.alpha else " "
                    line += f"  p={pv:.4f} {sig}"
                lines.append(line)

        if self.n_obs is not None:
            n_line = f"N              {self.n_obs}"
            if self.n_treated is not None and self.n_control is not None:
                n_line += (
                    f" ({self.n_treated} treated, {self.n_control} control)"
                )
            lines.append(n_line)

        if self.estimator_name is not None:
            lines.append(f"Estimator      {self.estimator_name}")
        if self.design_name is not None:
            lines.append(f"Design         {self.design_name}")
        if self.inference_name is not None:
            lines.append(f"Inference      {self.inference_name}")

        print("\n".join(lines))
        return self

    def __repr__(self) -> str:
        """Return compact string representation."""
        if self.ate is not None:
            return (
                f"Results(ate={self.ate}, ci={self.ci}, "
                f"p_value={self.p_value})"
            )
        n_eff = len(self.effects)  # type: ignore[arg-type]
        return f"Results(effects={n_eff} keys)"