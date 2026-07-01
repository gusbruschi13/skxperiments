"""Generate the versioned synthetic datasets for the use_cases notebooks.

Every dataset is fully reproducible: fixed per-dataset seed, written as a small
CSV under ``examples/use_cases/data/``. The data is *synthetic but realistic*
(seasonality-free, but with correlated pre-period covariates, heterogeneous
baselines and outliers where it matters), and it carries a KNOWN ground truth so
each notebook can show the library recovering the true effect.

Convention used by the notebooks (mirrors the for_starters "science table"
idiom): datasets carry the covariates plus the two potential outcomes ``*_y0``
and ``*_y1``. The notebook randomizes with the library, reads the assigned arm,
and builds the observed outcome as ``where(t == 1, y1, y0)``. The factorial case
instead ships a per-unit ``noise`` column, because the arm (cell) is drawn by the
design and the outcome is a function of the assigned factors plus that noise.

Run from the repo root:

    python examples/use_cases/_generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def _write(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    df.to_csv(path, index=False)
    print(f"wrote {path}  ({len(df)} rows, {len(df.columns)} cols)")


def gen_ecommerce_checkout() -> None:
    """Case 01: e-commerce electronics, user-level checkout A/B.

    Ground truth: revenue lift of +2.5 (currency units) per user. Baseline
    revenue is correlated with pre-period spend (rho ~ 0.6), which is what makes
    CUPED effective. A few high-spend outliers are included.
    """
    rng = np.random.default_rng(101)
    n = 4000

    device = rng.choice(["mobile", "desktop"], size=n, p=[0.65, 0.35])
    sessions_pre = rng.poisson(lam=4.0, size=n) + 1
    spend_pre = np.round(rng.gamma(shape=2.0, scale=30.0, size=n), 2)  # skewed

    tau = 2.5
    noise = rng.normal(0.0, 18.0, size=n)
    y0 = 20.0 + 0.35 * (spend_pre - spend_pre.mean()) + 2.0 * (device == "desktop") + noise
    y0 = np.clip(y0, 0.0, None)
    y1 = y0 + tau

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "device": device,
            "sessions_pre": sessions_pre,
            "spend_pre": spend_pre,
            "revenue_y0": np.round(y0, 2),
            "revenue_y1": np.round(y1, 2),
        }
    )
    _write(df, "ecommerce_checkout.csv")


def gen_fashion_stores() -> None:
    """Case 02: fashion brick-and-mortar, store-level test with blocking.

    Ground truth: sales lift of +0.40 (avg daily sales, in thousands) per store,
    constant across strata. Baseline sales differ strongly by store size
    (PP < P < M < G), which is exactly what blocking removes from the error.
    """
    rng = np.random.default_rng(202)
    sizes = ["PP", "P", "M", "G"]
    size_baseline = {"PP": 3.0, "P": 6.0, "M": 10.0, "G": 16.0}
    regions = ["Sudeste", "Sul", "Nordeste", "Norte"]

    rows = []
    store_id = 1
    for size in sizes:
        n_size = 30  # 120 stores total, balanced across sizes
        base = size_baseline[size]
        foot = rng.normal({"PP": 200, "P": 450, "M": 800, "G": 1500}[size], 80, n_size)
        region = rng.choice(regions, size=n_size)
        tau = 0.40
        y0 = base + 0.002 * (foot - foot.mean()) + rng.normal(0.0, 1.2, n_size)
        y1 = y0 + tau
        for i in range(n_size):
            rows.append(
                {
                    "store_id": store_id,
                    "region": region[i],
                    "store_size": size,
                    "foot_traffic_pre": round(float(foot[i]), 1),
                    "sales_y0": round(float(y0[i]), 3),
                    "sales_y1": round(float(y1[i]), 3),
                }
            )
            store_id += 1

    df = pd.DataFrame(rows)
    _write(df, "fashion_stores.csv")


def gen_fintech_crm() -> None:
    """Case 03: fintech CRM, 2^2 factorial (cashback x send-time).

    The design draws the cells, so we ship the population and a per-unit noise
    column. The notebook computes the outcome from the assigned factors with the
    documented coefficients:

        activations = 30 + 4*A + 2*B + 3*(A*B) + noise      (A, B in {0, 1})

    So the true main effect of A (full high-minus-low) is 4 + 3/2*... see the
    theory doc III; the point is a positive A x B interaction the notebook
    recovers.
    """
    rng = np.random.default_rng(303)
    n_per_cell = 1000
    n = n_per_cell * 4  # 2^2

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "tenure_months": rng.integers(1, 60, size=n),
            "noise": np.round(rng.normal(0.0, 3.0, size=n), 3),
        }
    )
    _write(df, "fintech_crm.csv")


def gen_logistics_dc() -> None:
    """Case 04: logistics, distribution-center level, few large units.

    Ground truth: +1.5 percentage points in the on-time rate. With only 24 DCs a
    single CRD draw is easily imbalanced on throughput/dock_count, so
    re-randomization matters. Baseline on-time rate depends on both covariates.
    """
    rng = np.random.default_rng(404)
    n = 24
    regions = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste"]

    throughput_pre = rng.normal(50_000, 15_000, size=n).round(0)
    dock_count = rng.integers(6, 24, size=n)
    tau = 1.5
    base = (
        88.0
        + 0.00005 * (throughput_pre - throughput_pre.mean())
        + 0.20 * (dock_count - dock_count.mean())
        + rng.normal(0.0, 1.5, size=n)
    )
    y0 = np.clip(base, None, 99.0)
    y1 = np.clip(y0 + tau, None, 99.5)

    df = pd.DataFrame(
        {
            "dc_id": np.arange(1, n + 1),
            "region": rng.choice(regions, size=n),
            "throughput_pre": throughput_pre.astype(int),
            "dock_count": dock_count,
            "on_time_y0": np.round(y0, 3),
            "on_time_y1": np.round(y1, 3),
        }
    )
    _write(df, "logistics_dc.csv")


def gen_streaming_metrics() -> None:
    """Case 05: streaming, user-level, many metrics + guardrails.

    Ground truth by metric (so multiple-testing has something to separate):
      - watch_time: true lift +3.0 min      (primary, should survive correction)
      - sessions:   tiny lift +0.05          (likely not significant)
      - completion: null (0.0)
      - buffering:  guardrail, ~null (+0.02) (should NOT flag as harm)
    """
    rng = np.random.default_rng(505)
    n = 5000
    plan = rng.choice(["basic", "premium"], size=n, p=[0.7, 0.3])

    def arm(base_mean, base_sd, tau, plan_bump=0.0):
        y0 = rng.normal(base_mean, base_sd, size=n) + plan_bump * (plan == "premium")
        return y0, y0 + tau

    wt0, wt1 = arm(42.0, 40.0, tau=3.0, plan_bump=15.0)
    se0, se1 = arm(6.0, 2.0, tau=0.05)
    co0, co1 = arm(0.55, 0.18, tau=0.0)
    bu0, bu1 = arm(4.0, 5.0, tau=0.02)  # guardrail: buffering seconds

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1),
            "plan": plan,
            "watch_time_y0": np.round(np.clip(wt0, 0, None), 3),
            "watch_time_y1": np.round(np.clip(wt1, 0, None), 3),
            "sessions_y0": np.round(np.clip(se0, 0, None), 3),
            "sessions_y1": np.round(np.clip(se1, 0, None), 3),
            "completion_y0": np.round(np.clip(co0, 0, 1), 4),
            "completion_y1": np.round(np.clip(co1, 0, 1), 4),
            "buffering_y0": np.round(np.clip(bu0, 0, None), 3),
            "buffering_y1": np.round(np.clip(bu1, 0, None), 3),
        }
    )
    _write(df, "streaming_metrics.csv")


def main() -> None:
    gen_ecommerce_checkout()
    gen_fashion_stores()
    gen_fintech_crm()
    gen_logistics_dc()
    gen_streaming_metrics()
    print("done.")


if __name__ == "__main__":
    main()
