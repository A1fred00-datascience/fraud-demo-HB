"""Dataset loading, demo generation, and statistical profiling."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = [
    "amount",
    "hour",
    "merchant_risk",
    "distance_km",
    "prev_txn_gap",
    "velocity_1h",
    "card_age_days",
    "is_international",
]
TARGET_COLUMN = "is_fraud"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def generate_demo_dataset(n: int = 1000, fraud_rate: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic fraud dataset with class-conditional distributions.

    Fraud transactions: higher amounts, late-night skew, higher merchant_risk,
    larger distance, faster velocity, more international, shorter card history.
    """
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.4, sigma=0.9, size=n_legit).round(2),
        "hour": np.clip(rng.normal(13, 4.5, n_legit).round().astype(int), 0, 23),
        "merchant_risk": np.clip(rng.beta(2, 8, n_legit) * 100, 0, 100).round(1),
        "distance_km": np.abs(rng.normal(8, 12, n_legit)).round(1),
        "prev_txn_gap": np.abs(rng.normal(720, 480, n_legit)).round(0),
        "velocity_1h": np.clip(rng.poisson(1.2, n_legit), 0, 25),
        "card_age_days": np.clip(rng.normal(900, 350, n_legit).round(0), 30, 3000),
        "is_international": (rng.random(n_legit) < 0.05).astype(int),
        TARGET_COLUMN: 0,
    })

    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.2, sigma=1.1, size=n_fraud).round(2),
        "hour": np.clip(
            np.where(
                rng.random(n_fraud) < 0.55,
                rng.normal(2, 2.5, n_fraud),
                rng.normal(15, 4, n_fraud),
            ).round().astype(int), 0, 23,
        ),
        "merchant_risk": np.clip(rng.beta(5, 3, n_fraud) * 100, 0, 100).round(1),
        "distance_km": np.abs(rng.normal(120, 80, n_fraud)).round(1),
        "prev_txn_gap": np.abs(rng.normal(45, 60, n_fraud)).round(0),
        "velocity_1h": np.clip(rng.poisson(5.5, n_fraud), 0, 25),
        "card_age_days": np.clip(rng.normal(280, 200, n_fraud).round(0), 1, 3000),
        "is_international": (rng.random(n_fraud) < 0.35).astype(int),
        TARGET_COLUMN: 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def load_csv(file_obj) -> pd.DataFrame:
    """Read an uploaded CSV. Validates required columns."""
    if hasattr(file_obj, "read"):
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_csv(io.StringIO(file_obj))
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Expected columns: {ALL_COLUMNS}"
        )
    return df[ALL_COLUMNS].copy()


def profile_dataset(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Dict[str, Any]:
    """Per-feature, per-class summary stats. This is what we send to Claude.

    No raw rows are exposed — only aggregate distribution statistics.
    """
    profile: Dict[str, Any] = {
        "n_total": int(len(df)),
        "n_fraud": int(df[target].sum()),
        "n_legit": int(len(df) - df[target].sum()),
        "fraud_rate": round(float(df[target].mean()), 4),
        "features": {},
    }

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        per_class: Dict[str, Dict[str, float]] = {}
        for cls_label, cls_val in (("legit", 0), ("fraud", 1)):
            sub = df.loc[df[target] == cls_val, col]
            if len(sub) == 0:
                continue
            per_class[cls_label] = {
                "mean": round(float(sub.mean()), 4),
                "std": round(float(sub.std() if len(sub) > 1 else 0.0), 4),
                "min": round(float(sub.min()), 4),
                "p25": round(float(sub.quantile(0.25)), 4),
                "median": round(float(sub.median()), 4),
                "p75": round(float(sub.quantile(0.75)), 4),
                "max": round(float(sub.max()), 4),
            }
        profile["features"][col] = per_class

    return profile


def kpi_counts(df: pd.DataFrame, target: str = TARGET_COLUMN) -> Dict[str, int]:
    return {
        "total": int(len(df)),
        "fraud": int(df[target].sum()),
        "legit": int(len(df) - df[target].sum()),
        "features": int(len([c for c in df.columns if c != target])),
    }
