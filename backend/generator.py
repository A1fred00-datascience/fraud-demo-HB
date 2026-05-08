"""Synthetic fraud data generation via Claude API.

Two providers are supported: the Anthropic API (default) and AWS Bedrock.
The Bedrock path is shown side-by-side so the migration is a one-line swap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import anthropic
import numpy as np
import pandas as pd

from .data_utils import FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"

INPUT_TOKEN_COST_PER_M = 1.00
OUTPUT_TOKEN_COST_PER_M = 5.00

SYSTEM_PROMPT = """You are a synthetic fraud transaction data generator.
You receive a statistical profile of real banking fraud data and produce
synthetic transactions whose distribution matches that profile per class.

Output rules:
- Return ONLY valid JSON. No prose, no markdown fences, no commentary.
- Top-level shape: {"records": [ { ...row... }, ... ]}
- Each record MUST contain every feature key listed in the user message.
- `is_fraud` must be 0 (legit) or 1 (fraud).
- Numeric values must respect the min/max bounds and rough mean/std of the
  matching class profile. Do not copy the means literally — sample plausibly.
- `is_international` is 0 or 1.
- `hour` is an integer 0-23. `velocity_1h` is a non-negative integer.
- All other numeric features are floats rounded to 2 decimals.
- Produce exactly the requested record counts per class."""


@dataclass
class GenerationResult:
    df: pd.DataFrame
    input_tokens: int = 0
    output_tokens: int = 0
    batches: int = 0
    cost_usd: float = 0.0
    warnings: List[str] = field(default_factory=list)
    provider: str = "anthropic"


class SyntheticGenerator:
    """Wraps the LLM call. Default provider is `anthropic`; `bedrock` available."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        provider: str = "anthropic",
        aws_region: str = "us-east-1",
    ):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider.")
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "local":
            # Free, no-API fallback: samples from the same statistical profile
            # using truncated Gaussians per class. Use this for demos / when
            # the API is offline or not provisioned.
            self.client = None
            self.model = "local-sampler"
        elif provider == "bedrock":
            # ── AWS Bedrock path (drop-in alternative) ─────────────────────────
            # The wire format is identical to the Anthropic API; only the
            # transport changes. Uncomment when migrating to Bedrock:
            #
            # import boto3
            # self.client = boto3.client("bedrock-runtime", region_name=aws_region)
            # self.model = DEFAULT_BEDROCK_MODEL_ID
            # ───────────────────────────────────────────────────────────────────
            raise NotImplementedError(
                "Bedrock provider is wired but commented out. "
                "Uncomment the boto3 block in backend/generator.py to enable."
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # ─── public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        profile: Dict[str, Any],
        n_records: int,
        fraud_ratio: float,
        on_progress: Optional[Callable[[float, str], None]] = None,
        batch_size: int = 100,
    ) -> GenerationResult:
        if self.provider == "local":
            return self._generate_local(profile, n_records, fraud_ratio, on_progress)

        n_fraud = int(round(n_records * fraud_ratio))
        n_legit = n_records - n_fraud

        all_rows: List[Dict[str, Any]] = []
        result = GenerationResult(df=pd.DataFrame(), provider=self.provider)

        remaining_fraud = n_fraud
        remaining_legit = n_legit
        total = n_records
        produced = 0

        while remaining_fraud + remaining_legit > 0:
            batch_total = min(batch_size, remaining_fraud + remaining_legit)
            batch_fraud_share = remaining_fraud / max(remaining_fraud + remaining_legit, 1)
            batch_fraud = int(round(batch_total * batch_fraud_share))
            batch_legit = batch_total - batch_fraud
            batch_fraud = min(batch_fraud, remaining_fraud)
            batch_legit = min(batch_legit, remaining_legit)

            if on_progress:
                on_progress(
                    produced / total,
                    f"Generating batch {result.batches + 1} · "
                    f"{batch_fraud} fraud + {batch_legit} legit",
                )

            text, in_tok, out_tok = self._call_model(profile, batch_fraud, batch_legit)
            rows = self._parse_records(text)

            for r in rows:
                cleaned = self._clean_row(r)
                if cleaned is not None:
                    all_rows.append(cleaned)

            result.input_tokens += in_tok
            result.output_tokens += out_tok
            result.batches += 1
            remaining_fraud -= batch_fraud
            remaining_legit -= batch_legit
            produced = total - (remaining_fraud + remaining_legit)

        if on_progress:
            on_progress(1.0, f"Done · {len(all_rows)} records")

        result.df = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS + [TARGET_COLUMN])
        result.cost_usd = (
            result.input_tokens / 1_000_000 * INPUT_TOKEN_COST_PER_M
            + result.output_tokens / 1_000_000 * OUTPUT_TOKEN_COST_PER_M
        )
        if len(result.df) < n_records:
            result.warnings.append(
                f"Model returned {len(result.df)} of {n_records} requested rows."
            )
        return result

    # ─── local (free) sampler ──────────────────────────────────────────────────

    def _generate_local(
        self,
        profile: Dict[str, Any],
        n_records: int,
        fraud_ratio: float,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> GenerationResult:
        """Sample synthetic records from the per-class profile using numpy.

        Truncated Gaussians per feature, with type-aware post-processing for
        integer/binary fields. Zero API cost; same downstream surface so KS
        validation and ML comparison work identically.
        """
        rng = np.random.default_rng()
        n_fraud = int(round(n_records * fraud_ratio))
        n_legit = n_records - n_fraud

        if on_progress:
            on_progress(0.0, "Sampling locally · no API call")

        rows: List[Dict[str, Any]] = []
        feature_names = list(profile.get("features", {}).keys())

        for cls_label, cls_val, n in (("legit", 0, n_legit), ("fraud", 1, n_fraud)):
            for _ in range(n):
                row: Dict[str, Any] = {}
                for feat in feature_names:
                    cls_stats = profile["features"][feat].get(cls_label)
                    if cls_stats is None:
                        continue
                    mean = cls_stats["mean"]
                    std = max(cls_stats["std"], 1e-3)
                    lo, hi = cls_stats["min"], cls_stats["max"]
                    val = float(np.clip(rng.normal(mean, std), lo, hi))
                    row[feat] = val
                row[TARGET_COLUMN] = cls_val
                rows.append(row)

        df = pd.DataFrame(rows, columns=feature_names + [TARGET_COLUMN])
        if "hour" in df.columns:
            df["hour"] = df["hour"].round().clip(0, 23).astype(int)
        if "velocity_1h" in df.columns:
            df["velocity_1h"] = df["velocity_1h"].round().clip(lower=0).astype(int)
        if "is_international" in df.columns:
            df["is_international"] = (df["is_international"] >= 0.5).astype(int)
        for col in df.columns:
            if col in ("hour", "velocity_1h", "is_international", TARGET_COLUMN):
                continue
            df[col] = df[col].round(2)

        df = df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)

        if on_progress:
            on_progress(1.0, f"Done · {len(df)} records sampled locally")

        return GenerationResult(
            df=df, input_tokens=0, output_tokens=0,
            batches=1, cost_usd=0.0, provider="local",
        )

    # ─── internals ─────────────────────────────────────────────────────────────

    def _build_user_prompt(self, profile: Dict[str, Any], n_fraud: int, n_legit: int) -> str:
        feature_keys = list(profile.get("features", {}).keys())
        return (
            f"Generate exactly {n_fraud} fraud (is_fraud=1) and {n_legit} legit "
            f"(is_fraud=0) synthetic transactions.\n\n"
            f"Required feature keys (every record must include all of these):\n"
            f"{json.dumps(feature_keys + [TARGET_COLUMN])}\n\n"
            f"Statistical profile (per feature, per class):\n"
            f"{json.dumps(profile['features'], indent=2)}\n\n"
            f"Return ONLY: {{\"records\": [...]}}"
        )

    def _call_model(self, profile: Dict[str, Any], n_fraud: int, n_legit: int):
        user_prompt = self._build_user_prompt(profile, n_fraud, n_legit)

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            return text, resp.usage.input_tokens, resp.usage.output_tokens

        # ── Bedrock equivalent (uncomment to use) ─────────────────────────────
        # import json as _json
        # body = _json.dumps({
        #     "anthropic_version": "bedrock-2023-05-31",
        #     "max_tokens": 8192,
        #     "system": SYSTEM_PROMPT,
        #     "messages": [{"role": "user", "content": user_prompt}],
        # })
        # resp = self.client.invoke_model(modelId=self.model, body=body)
        # payload = _json.loads(resp["body"].read())
        # text = "".join(b["text"] for b in payload["content"] if b["type"] == "text")
        # usage = payload.get("usage", {})
        # return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        # ──────────────────────────────────────────────────────────────────────

        raise RuntimeError("Unreachable: provider not wired.")

    @staticmethod
    def _parse_records(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
            return data.get("records", []) if isinstance(data, dict) else data
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                return data.get("records", []) if isinstance(data, dict) else []
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _clean_row(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            out = {
                "amount": float(r["amount"]),
                "hour": int(r["hour"]),
                "merchant_risk": float(r["merchant_risk"]),
                "distance_km": float(r["distance_km"]),
                "prev_txn_gap": float(r["prev_txn_gap"]),
                "velocity_1h": int(r["velocity_1h"]),
                "card_age_days": float(r["card_age_days"]),
                "is_international": int(r["is_international"]),
                TARGET_COLUMN: int(r[TARGET_COLUMN]),
            }
            out["hour"] = max(0, min(23, out["hour"]))
            out["is_international"] = 1 if out["is_international"] else 0
            out[TARGET_COLUMN] = 1 if out[TARGET_COLUMN] else 0
            out["velocity_1h"] = max(0, out["velocity_1h"])
            return out
        except (KeyError, TypeError, ValueError):
            return None
