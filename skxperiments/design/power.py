"""Power analysis for two-sample experiments with continuous outcomes.

Provides ``power_analysis``, a standalone function that solves for any
one of (sample size, minimum detectable effect, power) given the other
two, under the normal-approximation framework for the difference of
two means.

Scope (v1):
- Two groups (treated and control), continuous outcome.
- Test of mean difference, two-sided by default.
- Asymptotic normal approximation; valid for large n. Does not use
  the t distribution.
- Designs other than two-arm CRD (blocked, factorial, cluster,
  sequential) are out of scope; binary outcomes are out of scope.
"""

import math
from dataclasses import dataclass

from scipy.stats import norm

from skxperiments.core.exceptions import InvalidDesignError


@dataclass
class PowerResult:
    """Result of a power analysis.

    All fields are populated, including those that were inputs to the
    call. This makes the result self-describing and convenient for
    logging or downstream reuse.

    Attributes
    ----------
    n_total : int
        Total sample size.
    n_treated : int
        Number of units allocated to treatment.
    n_control : int
        Number of units allocated to control. Always satisfies
        ``n_treated + n_control == n_total``.
    mde : float
        Minimum detectable effect (positive by convention).
    power : float
        Statistical power (1 - beta) in (0, 1).
    alpha : float
        Significance level in (0, 1).
    std : float
        Standard deviation of the outcome (assumed equal across groups).
    allocation : float
        Proportion of units allocated to treatment.
    two_sided : bool
        Whether the test is two-sided.
    """

    n_total: int
    n_treated: int
    n_control: int
    mde: float
    power: float
    alpha: float
    std: float
    allocation: float
    two_sided: bool


def power_analysis(
    *,
    n: int | None = None,
    mde: float | None = None,
    power: float | None = None,
    std: float,
    alpha: float = 0.05,
    allocation: float = 0.5,
    two_sided: bool = True,
) -> PowerResult:
    """Solve for sample size, MDE, or power in a two-sample experiment.

    Exactly one of ``n``, ``mde``, ``power`` must be ``None`` — that
    is the quantity to be resolved. The other two must be provided.

    Parameters
    ----------
    n : int or None, optional
        Total sample size. Pass None to solve for n. By default None.
    mde : float or None, optional
        Minimum detectable effect (mean difference). Pass None to
        solve for MDE. By default None.
    power : float or None, optional
        Desired power (1 - beta). Pass None to solve for power. By
        default None.
    std : float
        Standard deviation of the outcome, assumed equal across
        groups. Required.
    alpha : float, optional
        Significance level, by default 0.05.
    allocation : float, optional
        Proportion of units allocated to treatment, in (0, 1), by
        default 0.5.
    two_sided : bool, optional
        Whether the test is two-sided, by default True.

    Returns
    -------
    PowerResult
        Self-describing result with all fields populated.

    Raises
    ------
    InvalidDesignError
        If more than one (or none) of ``n``, ``mde``, ``power`` is
        None; if ``alpha`` or ``allocation`` are not in (0, 1); if
        ``power`` (when provided) is not in (0, 1); if ``std <= 0``;
        if ``mde == 0`` when provided; if ``n`` is not a positive
        integer when provided.

    Notes
    -----
    Under the normal approximation for the difference of two
    independent means with common variance ``std**2``:

        sigma_eff = std * sqrt(1/allocation + 1/(1 - allocation))

    With ``z_alpha = Phi^{-1}(1 - alpha/2)`` for two-sided tests
    (or ``Phi^{-1}(1 - alpha)`` for one-sided) and
    ``z_beta = Phi^{-1}(power)``:

        n_total = ceil(((z_alpha + z_beta) * sigma_eff / mde)**2)
        mde     = (z_alpha + z_beta) * sigma_eff / sqrt(n_total)
        power   = Phi(sqrt(n_total) * |mde| / sigma_eff - z_alpha)

    See Cohen (1988), *Statistical Power Analysis for the Behavioral
    Sciences*, for derivations.

    Examples
    --------
    >>> result = power_analysis(
    ...     mde=0.2, power=0.8, std=1.0, alpha=0.05
    ... )
    >>> result.n_total  # total across both arms (about 393 per arm)
    785
    """
    # --- Validate which target to solve for -----------------------
    targets_none = sum(x is None for x in (n, mde, power))
    if targets_none != 1:
        raise InvalidDesignError(
            f"power_analysis requires exactly one of n, mde, power to be "
            f"None; received {targets_none} None values. "
            f"n={n!r}, mde={mde!r}, power={power!r}."
        )

    # --- Validate other parameters --------------------------------
    if not (0.0 < alpha < 1.0):
        raise InvalidDesignError(
            f"alpha must be in (0, 1), but received {alpha}."
        )

    if not (0.0 < allocation < 1.0):
        raise InvalidDesignError(
            f"allocation must be in (0, 1), but received {allocation}."
        )

    if std <= 0:
        raise InvalidDesignError(
            f"std must be > 0, but received {std}."
        )

    if power is not None and not (0.0 < power < 1.0):
        raise InvalidDesignError(
            f"power must be in (0, 1), but received {power}."
        )

    if mde is not None and mde == 0:
        raise InvalidDesignError(
            "mde must be non-zero when provided; received 0."
        )

    if n is not None:
        if not isinstance(n, int) or n <= 0:
            raise InvalidDesignError(
                f"n must be a positive integer when provided, "
                f"but received {n!r}."
            )

    # --- Compute z-quantiles --------------------------------------
    if two_sided:
        z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    else:
        z_alpha = float(norm.ppf(1.0 - alpha))

    sigma_eff = std * math.sqrt(1.0 / allocation + 1.0 / (1.0 - allocation))

    # --- Solve for the missing target -----------------------------
    if n is None:
        # Solve for n_total given mde, power.
        z_beta = float(norm.ppf(power))
        mde_abs = abs(mde)
        n_total = math.ceil(((z_alpha + z_beta) * sigma_eff / mde_abs) ** 2)
        resolved_mde = abs(mde)
        resolved_power = power

    elif mde is None:
        # Solve for mde given n, power.
        z_beta = float(norm.ppf(power))
        resolved_mde = (z_alpha + z_beta) * sigma_eff / math.sqrt(n)
        # By convention, MDE is reported as positive.
        resolved_mde = abs(resolved_mde)
        n_total = n
        resolved_power = power

    else:
        # Solve for power given n, mde.
        mde_abs = abs(mde)
        z_beta = math.sqrt(n) * mde_abs / sigma_eff - z_alpha
        resolved_power = float(norm.cdf(z_beta))
        n_total = n
        resolved_mde = mde_abs

    # --- Build allocation counts that sum exactly to n_total ------
    n_treated = int(round(n_total * allocation))
    n_control = n_total - n_treated

    return PowerResult(
        n_total=n_total,
        n_treated=n_treated,
        n_control=n_control,
        mde=resolved_mde,
        power=resolved_power,
        alpha=alpha,
        std=std,
        allocation=allocation,
        two_sided=two_sided,
    )