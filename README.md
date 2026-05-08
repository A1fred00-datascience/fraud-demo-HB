# SynthGuard

AI-powered synthetic fraud data generator. Loads a real fraud dataset, sends only its **statistical profile** (means, std, percentiles per class) to Claude — never raw rows — and generates synthetic transactions matching the distribution. Validates fidelity with KS tests and proves uplift by training a `GradientBoostingClassifier` under three regimes (real / synthetic / combined) on the same real holdout.

Built for a conference demo. AWS Bedrock-ready.

---

## Quick start (local)

```bash
cd fraud_synth_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

streamlit run frontend/app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`), click **Load demo dataset** on tab 01, then walk through tabs 02 → 04.

---

## Project structure

```
fraud_synth_demo/
├── backend/
│   ├── data_utils.py    # demo dataset, profiling, CSV loader
│   ├── generator.py     # Claude API + Bedrock side-by-side
│   ├── validator.py     # KS tests, correlation deltas
│   └── model.py         # 3-way GBM comparison
├── frontend/
│   └── app.py           # Streamlit UI (4 tabs)
├── .streamlit/
│   ├── config.toml      # dark fintech theme
│   └── secrets.toml.example
├── requirements.txt
├── .env.example
└── README.md
```

All Claude calls live in `backend/generator.py`. The frontend imports `SyntheticGenerator` and never embeds API logic.

---

## Demo flow

1. **01 · Upload Data** — click **Load demo dataset** for a self-contained 1,000-row dataset with 8 features and a 15% fraud rate. Or upload your own CSV.
2. **02 · Generate Synthetic** — pick record count and fraud ratio in the sidebar. The backend sends Claude only the per-class statistical profile.
3. **03 · Validate** — KS tests per feature (green < 0.15, amber < 0.30, red ≥ 0.30), distribution overlays, correlation heatmaps.
4. **04 · Model Comparison** — trains three GBMs (real / synthetic / combined), evaluates on the same 25% real holdout, shows ROC curves and the lift from adding synthetic data.

---

## AWS Bedrock migration

`backend/generator.py` ships with the Bedrock variant commented out side-by-side with the Anthropic API path. To switch:

1. Uncomment the `boto3` block in `SyntheticGenerator.__init__` and `_call_model`.
2. Set `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` in `.env`.
3. Pass `provider="bedrock"` when constructing `SyntheticGenerator`.

The wire format (`anthropic_version`, `messages`, `system`) is identical — only the transport changes.

---

## Deploy to Streamlit Cloud

1. Push this directory to a GitHub repo (do not commit `.env`).
2. Go to <https://share.streamlit.io>, click **New app**, connect the repo.
3. Set the **main file** to `frontend/app.py`.
4. Under **Advanced settings → Secrets**, paste:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```

5. Click **Deploy**. The free tier runs the demo end-to-end.

---

## Cost note

Default model is `claude-haiku-4-5-20251001`. The sidebar shows session token usage and an estimated cost based on $1.00 / M input and $5.00 / M output tokens. Generating 200 records typically costs well under one cent.
