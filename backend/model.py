"""ML training & evaluation across real / synthetic / combined regimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .data_utils import FEATURE_COLUMNS, TARGET_COLUMN

REGIMES = ["real_only", "synthetic_only", "combined"]
REGIME_LABELS = {
    "real_only": "Real only",
    "synthetic_only": "Synthetic only",
    "combined": "Real + Synthetic",
}


@dataclass
class RegimeResult:
    regime: str
    label: str
    auc: float
    precision: float
    recall: float
    f1: float
    fpr: List[float] = field(default_factory=list)
    tpr: List[float] = field(default_factory=list)
    n_train: int = 0


@dataclass
class ComparisonResult:
    regimes: List[RegimeResult]
    best_regime: str
    test_size: int
    insight: str


def _train_eval(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 7,
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
    if len(np.unique(y_train)) < 2:
        return float("nan"), 0.0, 0.0, 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    clf = GradientBoostingClassifier(
        n_estimators=120, max_depth=3, learning_rate=0.08, random_state=seed
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, proba))
    prec = float(precision_score(y_test, pred, zero_division=0))
    rec = float(recall_score(y_test, pred, zero_division=0))
    f1 = float(f1_score(y_test, pred, zero_division=0))
    fpr, tpr, _ = roc_curve(y_test, proba)
    return auc, prec, rec, f1, fpr, tpr


def compare_models(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 7,
) -> ComparisonResult:
    """Train 3 GBMs on real / synthetic / combined; evaluate on the same real holdout."""
    feats = [c for c in FEATURE_COLUMNS if c in real_df.columns]
    X_real = real_df[feats]
    y_real = real_df[TARGET_COLUMN].astype(int)

    X_real_train, X_test, y_real_train, y_test = train_test_split(
        X_real, y_real, test_size=test_size, random_state=seed,
        stratify=y_real if y_real.nunique() > 1 else None,
    )

    X_synth = synthetic_df[feats]
    y_synth = synthetic_df[TARGET_COLUMN].astype(int)

    X_combined = pd.concat([X_real_train, X_synth], ignore_index=True)
    y_combined = pd.concat([y_real_train, y_synth], ignore_index=True)

    results: List[RegimeResult] = []
    for regime, X_tr, y_tr in [
        ("real_only", X_real_train, y_real_train),
        ("synthetic_only", X_synth, y_synth),
        ("combined", X_combined, y_combined),
    ]:
        auc, prec, rec, f1, fpr, tpr = _train_eval(X_tr, y_tr, X_test, y_test, seed)
        results.append(RegimeResult(
            regime=regime, label=REGIME_LABELS[regime],
            auc=auc, precision=prec, recall=rec, f1=f1,
            fpr=fpr.tolist(), tpr=tpr.tolist(),
            n_train=len(X_tr),
        ))

    best = max(results, key=lambda r: (r.auc if not np.isnan(r.auc) else -1))
    real_auc = next(r.auc for r in results if r.regime == "real_only")
    combined_auc = next(r.auc for r in results if r.regime == "combined")
    if real_auc and not np.isnan(real_auc) and real_auc > 0:
        lift = (combined_auc - real_auc) / real_auc * 100
        if lift > 0.5:
            insight = (
                f"Adding synthetic data lifted AUC by **+{lift:.2f}%** over real-only "
                f"(real {real_auc:.3f} → combined {combined_auc:.3f})."
            )
        elif lift < -0.5:
            insight = (
                f"Combined training underperformed real-only by {abs(lift):.2f}% "
                f"— synthetic data may need more fidelity tuning."
            )
        else:
            insight = (
                f"Combined performance is on par with real-only "
                f"(Δ {lift:+.2f}%) — synthetic data is augmenting without harm."
            )
    else:
        insight = "Real-only baseline did not converge; cannot compute lift."

    return ComparisonResult(
        regimes=results,
        best_regime=best.regime,
        test_size=len(X_test),
        insight=insight,
    )


def metrics_long_form(comparison: ComparisonResult) -> pd.DataFrame:
    rows: List[Dict] = []
    for r in comparison.regimes:
        for metric_name, val in [("AUC", r.auc), ("Precision", r.precision),
                                 ("Recall", r.recall), ("F1", r.f1)]:
            rows.append({"regime": r.label, "metric": metric_name, "value": val})
    return pd.DataFrame(rows)
