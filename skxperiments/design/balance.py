"""Covariate balance diagnostics for experimental designs.

Provides check_balance, a standalone function that computes the
standardized mean difference (SMD) between treatment and control
groups for each covariate in an Assignment.
"""

import numpy as np
import pandas as pd

from skxperiments.core.assignment import BaseAssignment
from skxperiments.core.exceptions import (
    InsufficientDataError,
    InvalidDesignError,
)


def check_balance(
    assignment: BaseAssignment,
    covariates: list[str] | None = None,
) -> pd.DataFrame:
    """Compute covariate balance between treatment and control groups.

    For each covariate, returns the mean in each group, the pooled
    standard deviation, and the standardized mean difference (SMD).

    Parameters
    ----------
    assignment : BaseAssignment
        Assignment object produced by a design. Must expose ``data_``
        (DataFrame with treatment column attached) and ``treatment_col_``.
    covariates : list of str or None, optional
        Names of covariates to check. If None, all numeric columns in
        ``assignment.data_`` except the treatment column are used, in
        the order they appear in the DataFrame. Boolean columns count
        as numeric. By default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per covariate and columns:
        ``covariate``, ``mean_treated``, ``mean_control``,
        ``std_pooled``, ``smd``. The index is a default RangeIndex.

    Raises
    ------
    InvalidDesignError
        If a name in ``covariates`` is not a column of ``assignment.data_``,
        or if any selected covariate contains NaN values.
    InsufficientDataError
        If ``covariates`` is None and no numeric columns are available
        after excluding the treatment column.

    Notes
    -----
    The pooled standard deviation follows the convention common in
    the SMD literature for randomized experiments (Austin 2009;
    Stuart 2010):

        std_pooled = sqrt((var_treated + var_control) / 2)

    where each variance is computed with ``ddof=1``. When
    ``std_pooled == 0`` (no within-group variation), the SMD is NaN
    rather than raising an exception.

    The function does not modify ``assignment.data_``.

    References
    ----------
    Austin, P. C. (2009). Balance diagnostics for comparing the
    distribution of baseline covariates between treatment groups in
    propensity-score matched samples. Statistics in Medicine.

    Stuart, E. A. (2010). Matching methods for causal inference:
    A review and a look forward. Statistical Science.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skxperiments.core.assignment import CRDAssignment
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=100),
    ...     "x2": rng.normal(size=100),
    ...     "treatment": rng.integers(0, 2, size=100),
    ... })
    >>> assignment = CRDAssignment(
    ...     data=df, treatment_col="treatment", design=None, seed=42
    ... )
    >>> result = check_balance(assignment)
    >>> set(result.columns) == {
    ...     "covariate", "mean_treated", "mean_control",
    ...     "std_pooled", "smd",
    ... }
    True
    """
    data = assignment.data_
    treatment_col = assignment.treatment_col_

    # Resolve covariate list
    if covariates is None:
        numeric_cols = [
            col
            for col in data.columns
            if col != treatment_col and pd.api.types.is_numeric_dtype(data[col])
        ]
        if len(numeric_cols) == 0:
            raise InsufficientDataError(
                context=(
                    "check_balance with covariates=None "
                    "(no numeric columns available after excluding "
                    f"treatment column '{treatment_col}')"
                ),
                minimum=1,
                received=0,
            )
        selected = numeric_cols
    else:
        missing = [c for c in covariates if c not in data.columns]
        if missing:
            raise InvalidDesignError(
                f"Covariates not found in assignment.data_: {missing}. "
                f"Available columns: {list(data.columns)}."
            )
        selected = list(covariates)

    # Validate no NaN in any selected covariate
    cols_with_nan = [c for c in selected if data[c].isna().any()]
    if cols_with_nan:
        raise InvalidDesignError(
            f"Covariates contain NaN values: {cols_with_nan}. "
            f"check_balance requires complete data; impute or drop NaN "
            f"before calling."
        )

    # Compute group masks
    treatment_values = data[treatment_col].values
    treated_mask = treatment_values == 1
    control_mask = treatment_values == 0

    # Compute statistics per covariate
    rows: list[dict[str, float | str]] = []
    for cov in selected:
        values = data[cov].astype(float).values
        treated_vals = values[treated_mask]
        control_vals = values[control_mask]

        mean_t = float(np.mean(treated_vals))
        mean_c = float(np.mean(control_vals))

        var_t = float(np.var(treated_vals, ddof=1))
        var_c = float(np.var(control_vals, ddof=1))

        std_pooled = float(np.sqrt((var_t + var_c) / 2.0))

        if std_pooled == 0.0:
            smd: float = float("nan")
        else:
            smd = (mean_t - mean_c) / std_pooled

        rows.append(
            {
                "covariate": cov,
                "mean_treated": mean_t,
                "mean_control": mean_c,
                "std_pooled": std_pooled,
                "smd": smd,
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "covariate",
            "mean_treated",
            "mean_control",
            "std_pooled",
            "smd",
        ],
    )
    return result.reset_index(drop=True)