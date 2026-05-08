"""Statistical validation of synthetic vs real data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from .data_utils import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class FeatureKS:
    feature: str
    ks_stat: float
    p_value: float
    fidelity: str  # "high" | "medium" | "low"


def _bucket(ks_stat: float) -> str:
    if ks_stat < 0.15:
        return "high"
    if ks_stat < 0.30:
        return "medium"
    return "low"


def ks_test_features(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    features: List[str] = None,
) -> List[FeatureKS]:
    """Two-sample Kolmogorov-Smirnov test per feature."""
    features = features or [c for c in FEATURE_COLUMNS if c in real_df.columns and c in synthetic_df.columns]
    out: List[FeatureKS] = []
    for col in features:
        real_vals = real_df[col].dropna().to_numpy(dtype=float)
        synth_vals = synthetic_df[col].dropna().to_numpy(dtype=float)
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        ks = stats.ks_2samp(real_vals, synth_vals)
        out.append(
            FeatureKS(
                feature=col,
                ks_stat=float(ks.statistic),
                p_value=float(ks.pvalue),
                fidelity=_bucket(float(ks.statistic)),
            )
        )
    return out


def fidelity_summary(results: List[FeatureKS]) -> Dict[str, float]:
    if not results:
        return {"avg_ks": 0.0, "high": 0, "medium": 0, "low": 0, "n": 0}
    counts = {"high": 0, "medium": 0, "low": 0}
    for r in results:
        counts[r.fidelity] += 1
    return {
        "avg_ks": float(np.mean([r.ks_stat for r in results])),
        "high": counts["high"],
        "medium": counts["medium"],
        "low": counts["low"],
        "n": len(results),
    }


def correlation_matrix(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    features = features or [c for c in FEATURE_COLUMNS if c in df.columns]
    return df[features].corr().round(3)


def correlation_delta(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> float:
    """Mean absolute difference between real and synthetic correlation matrices."""
    a = correlation_matrix(real_df).to_numpy()
    b = correlation_matrix(synthetic_df).to_numpy()
    if a.shape != b.shape:
        return float("nan")
    return float(np.nanmean(np.abs(a - b)))
