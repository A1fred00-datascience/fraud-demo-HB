"""SynthGuard — AI-powered synthetic fraud data generator (Streamlit UI)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Make `backend/` importable when launched as `streamlit run frontend/app.py`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import data_utils, model as ml_model, validator
from backend.generator import (
    DEFAULT_MODEL,
    INPUT_TOKEN_COST_PER_M,
    OUTPUT_TOKEN_COST_PER_M,
    SyntheticGenerator,
)

load_dotenv()

# ─── Page config & theme ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="SynthGuard · AI Fraud Data Generator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Management Solutions brand palette — navy + gold
ACCENT = "#c9a437"        # gold
ACCENT_DIM = "#97791d"
NAVY = "#152543"          # navy (matches Management Solutions lettering)
BG = "#0a1530"            # deep navy background
SURFACE = "#142244"
SURFACE_2 = "#1c2c54"
BORDER = "#2a3d6b"
TEXT = "#f0e9d6"          # warm cream — pairs with gold
MUTED = "#8595b7"
RED = "#e07b6e"
AMBER = "#e8c971"         # cream-gold for medium fidelity
BLUE = "#6aa9ff"          # secondary chart accent

PLOTLY_TEMPLATE = "plotly_dark"
PLOT_LAYOUT = dict(
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="DM Mono, ui-monospace, monospace", color=TEXT, size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    colorway=[ACCENT, BLUE, AMBER, RED, "#b08cff", "#4cd0d6"],
)



def inject_css() -> None:
    accent_rgb = "201, 164, 55"      # ACCENT #c9a437
    red_rgb = "224, 123, 110"        # RED   #e07b6e
    amber_rgb = "232, 201, 113"      # AMBER #e8c971
    blue_rgb = "106, 169, 255"       # BLUE  #6aa9ff

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

        /* App shell · navy gradient with subtle gold halo */
        .stApp {{
            background:
                radial-gradient(900px 500px at 90% -10%, rgba({accent_rgb},0.08), transparent 60%),
                radial-gradient(700px 400px at -10% 100%, rgba({blue_rgb},0.06), transparent 60%),
                {BG};
            color: {TEXT};
            font-family: 'DM Mono', ui-monospace, monospace;
        }}
        [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
            background: transparent;
        }}
        [data-testid="stHeader"] {{ background: transparent; height: 0; }}
        #MainMenu, footer {{ visibility: hidden; }}

        h1, h2, h3, h4, h5, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
            font-family: 'Syne', sans-serif;
            letter-spacing: -0.015em;
            color: {TEXT};
        }}
        h1 {{ font-weight: 800; }}
        h2 {{ font-weight: 700; }}
        h3 {{ font-weight: 600; font-size: 1.4rem; }}

        .block-container {{
            padding-top: 1.6rem;
            padding-bottom: 4rem;
            max-width: 1440px;
        }}

        /* Header card */
        .sg-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 26px;
            border: 1px solid {BORDER};
            border-radius: 16px;
            background:
                radial-gradient(400px 200px at 100% 0%, rgba({accent_rgb},0.10), transparent 70%),
                linear-gradient(135deg, {SURFACE} 0%, {SURFACE_2} 100%);
            margin-bottom: 24px;
            box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
        }}
        .sg-header .brand {{ display: flex; align-items: center; gap: 18px; }}
        .sg-eyebrow {{
            font-family: 'DM Mono', monospace; font-size: 10px;
            color: {ACCENT}; letter-spacing: 0.20em; text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .sg-title {{
            font-family: 'Syne', sans-serif; font-weight: 800; font-size: 32px;
            color: {TEXT}; line-height: 1.0; letter-spacing: -0.02em;
        }}
        .sg-subtitle {{
            font-family: 'DM Mono', monospace; font-size: 11px;
            color: {MUTED}; letter-spacing: 0.1em; text-transform: uppercase;
            margin-top: 6px;
        }}
        .sg-pill {{
            display: inline-flex; align-items: center; gap: 8px;
            padding: 7px 14px; border-radius: 999px;
            background: rgba({accent_rgb}, 0.10);
            border: 1px solid rgba({accent_rgb}, 0.40);
            color: {ACCENT}; font-size: 11px;
            font-family: 'DM Mono', monospace; letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .sg-dot {{
            width: 6px; height: 6px; border-radius: 50%; background: {ACCENT};
            box-shadow: 0 0 10px {ACCENT};
        }}

        /* KPI cards */
        .sg-card {{
            background: linear-gradient(180deg, {SURFACE} 0%, {SURFACE_2} 100%);
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 20px 22px;
            min-height: 110px;
            transition: border-color 120ms ease, transform 120ms ease;
        }}
        .sg-card:hover {{ border-color: rgba({accent_rgb}, 0.35); transform: translateY(-1px); }}
        .sg-card-header {{
            font-family: 'DM Mono', monospace;
            color: {MUTED}; font-size: 10px;
            letter-spacing: 0.14em; text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .sg-card-value {{
            font-family: 'Syne', sans-serif; font-weight: 700;
            font-size: 34px; color: {TEXT}; line-height: 1;
        }}
        .sg-card-foot {{
            font-family: 'DM Mono', monospace;
            color: {MUTED}; font-size: 11px; margin-top: 8px;
        }}
        .sg-card.accent {{
            border-color: {ACCENT};
            box-shadow: 0 0 0 1px rgba({accent_rgb},0.20), 0 8px 24px rgba({accent_rgb},0.08);
        }}
        .sg-card.accent .sg-card-value {{ color: {ACCENT}; }}
        .sg-card.warn   {{ border-color: rgba({amber_rgb},0.40); }}
        .sg-card.warn   .sg-card-value {{ color: {AMBER}; }}
        .sg-card.danger {{ border-color: rgba({red_rgb},0.40); }}
        .sg-card.danger .sg-card-value {{ color: {RED}; }}

        .sg-badge {{
            display: inline-block; padding: 3px 10px; border-radius: 6px;
            font-size: 10px; font-family: 'DM Mono', monospace;
            letter-spacing: 0.10em; text-transform: uppercase;
        }}
        .sg-badge.ok   {{ background: rgba({accent_rgb},0.14); color: {ACCENT}; border:1px solid rgba({accent_rgb},0.40);}}
        .sg-badge.warn {{ background: rgba({amber_rgb},0.14); color: {AMBER}; border:1px solid rgba({amber_rgb},0.40);}}
        .sg-badge.bad  {{ background: rgba({red_rgb},0.14);   color: {RED};   border:1px solid rgba({red_rgb},0.40);}}

        /* Insight callout */
        .sg-insight {{
            background:
                radial-gradient(400px 120px at 0% 0%, rgba({accent_rgb},0.18), transparent 70%),
                linear-gradient(135deg, rgba({accent_rgb},0.08), rgba({accent_rgb},0.02));
            border: 1px solid rgba({accent_rgb},0.45);
            border-left: 4px solid {ACCENT};
            padding: 18px 22px;
            border-radius: 12px;
            font-size: 14px; color: {TEXT};
        }}
        .sg-insight .label {{
            color: {ACCENT}; font-family: 'DM Mono', monospace;
            font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
            margin-bottom: 8px;
        }}

        /* Section heading */
        .sg-section {{
            font-family: 'DM Mono', monospace;
            color: {MUTED}; font-size: 11px;
            letter-spacing: 0.16em; text-transform: uppercase;
            margin: 22px 0 12px 0;
            display: flex; align-items: center; gap: 10px;
        }}
        .sg-section::before {{
            content: ""; display: inline-block; width: 16px; height: 1px;
            background: {ACCENT};
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            background: {ACCENT} !important; color: {BG} !important;
            border: 0 !important;
            font-family: 'DM Mono', monospace !important; font-weight: 500 !important;
            letter-spacing: 0.08em !important; text-transform: uppercase !important;
            padding: 11px 20px !important; border-radius: 8px !important;
            font-size: 12px !important;
            box-shadow: 0 4px 14px rgba({accent_rgb},0.25);
            transition: all 140ms ease !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background: {AMBER} !important; color: {BG} !important;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba({accent_rgb},0.40);
        }}
        .stButton > button:disabled {{
            background: {SURFACE_2} !important; color: {MUTED} !important;
            box-shadow: none !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px; background: transparent;
            border-bottom: 1px solid {BORDER};
            padding-bottom: 0;
        }}
        .stTabs [data-baseweb="tab"] {{
            background: transparent !important; color: {MUTED} !important;
            font-family: 'DM Mono', monospace !important; font-size: 12px !important;
            letter-spacing: 0.10em !important; text-transform: uppercase !important;
            padding: 14px 22px !important;
            border-bottom: 2px solid transparent !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ACCENT} !important;
            border-bottom-color: {ACCENT} !important;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {SURFACE} 0%, {BG} 100%) !important;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            font-family: 'Syne', sans-serif;
            color: {TEXT};
        }}
        section[data-testid="stSidebar"] hr {{ border-color: {BORDER}; margin: 14px 0; }}

        /* Form widgets */
        .stTextInput input, .stNumberInput input,
        .stSelectbox > div > div, [data-baseweb="select"] > div {{
            background: {SURFACE_2} !important;
            color: {TEXT} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        .stSlider [data-baseweb="slider"] [role="slider"] {{
            background: {ACCENT} !important;
            border: 2px solid {ACCENT} !important;
            box-shadow: 0 0 0 4px rgba({accent_rgb},0.20);
        }}
        .stSlider [data-baseweb="slider"] > div > div {{
            background: {ACCENT} !important;
        }}

        /* File uploader */
        [data-testid="stFileUploader"] {{
            background: {SURFACE}; border: 1.5px dashed {BORDER};
            border-radius: 12px; padding: 4px;
        }}
        [data-testid="stFileUploader"] section {{
            background: transparent !important;
            border-radius: 10px;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: {SURFACE_2} !important;
            border: 1.5px dashed rgba({accent_rgb},0.35) !important;
            border-radius: 10px !important;
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {ACCENT} !important;
        }}

        /* Dataframes */
        [data-testid="stDataFrame"] {{
            background: {SURFACE} !important;
            border: 1px solid {BORDER}; border-radius: 12px;
            overflow: hidden;
        }}
        [data-testid="stDataFrame"] [role="grid"] {{
            background: {SURFACE} !important;
        }}

        /* Progress bar */
        .stProgress > div > div > div {{
            background: linear-gradient(90deg, {ACCENT}, {AMBER}) !important;
        }}

        /* Alerts */
        [data-testid="stAlert"] {{ border-radius: 10px; }}

        /* Markdown text */
        [data-testid="stMarkdownContainer"] p {{ color: {TEXT}; }}
        [data-testid="stMarkdownContainer"] code {{
            background: {SURFACE_2}; color: {ACCENT};
            padding: 2px 6px; border-radius: 4px;
            font-family: 'DM Mono', monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─── Session state init ───────────────────────────────────────────────────────

def init_state() -> None:
    defaults = {
        "real_df": None,
        "synthetic_df": None,
        "profile": None,
        "gen_meta": None,         # GenerationResult metadata
        "comparison": None,
        "tokens_used_in": 0,
        "tokens_used_out": 0,
        "cost_usd": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── Reusable components ──────────────────────────────────────────────────────

def header() -> None:
    st.markdown(
        f"""
        <div class="sg-header">
          <div class="brand">
            <div>
              <div class="sg-eyebrow">by Management Solutions</div>
              <div class="sg-title">SynthGuard</div>
              <div class="sg-subtitle">synthetic fraud data · model uplift studio</div>
            </div>
          </div>
          <div class="sg-pill"><span class="sg-dot"></span>Claude · {DEFAULT_MODEL}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, foot: str = "", variant: str = "") -> str:
    cls = f"sg-card {variant}".strip()
    return (
        f"<div class='{cls}'>"
        f"<div class='sg-card-header'>{label}</div>"
        f"<div class='sg-card-value'>{value}</div>"
        + (f"<div class='sg-card-foot'>{foot}</div>" if foot else "")
        + "</div>"
    )


def section(title: str) -> None:
    st.markdown(f"<div class='sg-section'>{title}</div>", unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
    fig.update_layout(template=PLOTLY_TEMPLATE, **PLOT_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(family="Syne", size=15, color=TEXT)))
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def get_api_key() -> Optional[str]:
    env_key = os.getenv("ANTHROPIC_API_KEY")
    try:
        secrets_key = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        secrets_key = None
    return env_key or secrets_key or st.session_state.get("api_key_override") or None


def sidebar() -> None:
    with st.sidebar:
        st.markdown("## Configuration")

        st.markdown("**API key**")
        env_key = os.getenv("ANTHROPIC_API_KEY")
        try:
            secrets_key = st.secrets.get("ANTHROPIC_API_KEY")  # type: ignore[attr-defined]
        except Exception:
            secrets_key = None

        if env_key or secrets_key:
            source = ".env" if env_key else "secrets.toml"
            st.markdown(
                f"<span class='sg-badge ok'>Loaded from {source}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.session_state["api_key_override"] = st.text_input(
                "ANTHROPIC_API_KEY",
                type="password",
                value=st.session_state.get("api_key_override", ""),
                help="Falls back to .env or secrets.toml when set.",
                label_visibility="collapsed",
                placeholder="sk-ant-...",
            )

        st.markdown("---")
        st.markdown("## Generation")

        st.session_state["provider"] = st.radio(
            "Generator",
            options=["Claude API", "Local sampling (free)"],
            index=1 if st.session_state.get("provider") == "Local sampling (free)" else 0,
            help="Local sampling uses statistical sampling from the dataset profile — "
                 "no API call, no cost. Ideal for offline demos.",
        )

        st.session_state["n_records"] = st.slider(
            "Synthetic records", 50, 500, st.session_state.get("n_records", 200), step=50
        )
        fraud_pct = st.slider(
            "Fraud ratio (%)", 5, 50, st.session_state.get("fraud_ratio_pct", 25), step=5
        )
        st.session_state["fraud_ratio_pct"] = fraud_pct
        st.session_state["fraud_ratio"] = fraud_pct / 100.0
        st.session_state["model_choice"] = st.selectbox(
            "Model",
            options=[
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-6",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown("## Cost (CloudWatch-style)")
        in_tok = st.session_state.get("tokens_used_in", 0)
        out_tok = st.session_state.get("tokens_used_out", 0)
        cost = st.session_state.get("cost_usd", 0.0)
        st.markdown(
            f"""
            <div class='sg-card'>
              <div class='sg-card-header'>Session usage</div>
              <div style='display:flex;justify-content:space-between;margin-top:4px;'>
                <span style='color:{MUTED};font-size:12px;'>Input tok</span>
                <span style='font-family:DM Mono;font-weight:500;'>{in_tok:,}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:2px;'>
                <span style='color:{MUTED};font-size:12px;'>Output tok</span>
                <span style='font-family:DM Mono;font-weight:500;'>{out_tok:,}</span>
              </div>
              <div style='display:flex;justify-content:space-between;margin-top:8px;border-top:1px solid {BORDER};padding-top:8px;'>
                <span style='color:{ACCENT};font-size:12px;'>Est. cost</span>
                <span style='font-family:DM Mono;color:{ACCENT};font-weight:500;'>${cost:.4f}</span>
              </div>
              <div class='sg-card-foot'>
                ${INPUT_TOKEN_COST_PER_M:.2f}/M in · ${OUTPUT_TOKEN_COST_PER_M:.2f}/M out
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Reset session", key="reset_session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─── Tab 01 — Upload Data ─────────────────────────────────────────────────────

def tab_upload() -> None:
    st.markdown("### 01 · Source dataset")
    st.markdown(
        f"<p style='color:{MUTED};margin-top:-8px;'>Load a CSV with required columns, "
        "or fire up the demo dataset for a self-contained run. A ready-to-upload sample "
        "lives at <code>sample_data/fraud_sample.csv</code>.</p>",
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([2, 1])
    with col_l:
        uploaded = st.file_uploader(
            "Upload fraud dataset (CSV)", type=["csv"], label_visibility="collapsed"
        )
        if uploaded is not None:
            try:
                df = data_utils.load_csv(uploaded)
                st.session_state["real_df"] = df
                st.session_state["profile"] = data_utils.profile_dataset(df)
                st.success(f"Loaded {len(df):,} rows from {uploaded.name}.")
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
    with col_r:
        if st.button("⚡ Load demo dataset", width="stretch"):
            df = data_utils.generate_demo_dataset(n=1000, fraud_rate=0.15)
            st.session_state["real_df"] = df
            st.session_state["profile"] = data_utils.profile_dataset(df)
            st.toast("Demo dataset loaded · 1,000 transactions", icon="✅")

    df = st.session_state.get("real_df")
    if df is None:
        st.info("Awaiting data. Use the demo button to start.")
        return

    counts = data_utils.kpi_counts(df)
    cols = st.columns(4)
    cards = [
        kpi_card("Total records", f"{counts['total']:,}"),
        kpi_card("Fraud", f"{counts['fraud']:,}",
                 foot=f"{counts['fraud']/counts['total']*100:.1f}% of total", variant="danger"),
        kpi_card("Legit", f"{counts['legit']:,}",
                 foot=f"{counts['legit']/counts['total']*100:.1f}% of total"),
        kpi_card("Features", f"{counts['features']}"),
    ]
    for col, html in zip(cols, cards):
        col.markdown(html, unsafe_allow_html=True)

    section("Sample · first 10 rows")
    st.dataframe(df.head(10), width="stretch", hide_index=True)

    section("Feature distribution · real vs fraud overlay")
    feature_options = data_utils.FEATURE_COLUMNS
    selected = st.selectbox("Feature", feature_options, key="upload_feat", label_visibility="collapsed")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df.loc[df[data_utils.TARGET_COLUMN] == 0, selected],
        name="Legit", opacity=0.75, marker_color=ACCENT, nbinsx=40,
    ))
    fig.add_trace(go.Histogram(
        x=df.loc[df[data_utils.TARGET_COLUMN] == 1, selected],
        name="Fraud", opacity=0.75, marker_color=RED, nbinsx=40,
    ))
    fig.update_layout(barmode="overlay", bargap=0.02)
    st.plotly_chart(style_fig(fig, f"{selected} · class overlay"), width="stretch")


# ─── Tab 02 — Generate Synthetic ──────────────────────────────────────────────

def tab_generate() -> None:
    st.markdown("### 02 · Synthesize with Claude")

    df = st.session_state.get("real_df")
    if df is None:
        st.info("Load a dataset on tab 01 first.")
        return

    n_records = st.session_state.get("n_records", 200)
    fraud_ratio = st.session_state.get("fraud_ratio", 0.25)
    n_fraud = int(round(n_records * fraud_ratio))
    n_legit = n_records - n_fraud

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Plan · total", f"{n_records}"), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Plan · fraud", f"{n_fraud}",
                              foot=f"{fraud_ratio*100:.0f}% ratio", variant="danger"),
                     unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Plan · legit", f"{n_legit}"), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Model", st.session_state.get("model_choice", DEFAULT_MODEL).split("-")[1].upper()),
                     unsafe_allow_html=True)

    section("Generation")
    use_local = st.session_state.get("provider") == "Local sampling (free)"
    api_key = get_api_key()
    can_run = use_local or bool(api_key)
    if not can_run:
        st.warning("Set ANTHROPIC_API_KEY in .env, secrets.toml, or the sidebar — "
                   "or switch the sidebar **Generator** to **Local sampling (free)** "
                   "to run without an API call.")
    if use_local:
        st.info("Local sampling mode · no API call, no cost. Toggle to **Claude API** in the sidebar to use Claude.")

    if st.button("⚡ Generate synthetic dataset", disabled=not can_run):
        progress = st.progress(0.0, text="Initializing…")
        status = st.empty()

        def on_progress(pct: float, msg: str) -> None:
            progress.progress(min(pct, 1.0), text=msg)
            status.markdown(f"<div style='color:{MUTED};font-size:12px;'>{msg}</div>",
                            unsafe_allow_html=True)

        try:
            gen = SyntheticGenerator(
                api_key=api_key if not use_local else None,
                model=st.session_state.get("model_choice", DEFAULT_MODEL),
                provider="local" if use_local else "anthropic",
            )
            result = gen.generate(
                profile=st.session_state["profile"],
                n_records=n_records,
                fraud_ratio=fraud_ratio,
                on_progress=on_progress,
                batch_size=100,
            )
        except Exception as e:
            progress.empty()
            st.error(f"Generation failed: {e}")
            return

        progress.empty()
        st.session_state["synthetic_df"] = result.df
        st.session_state["gen_meta"] = result
        st.session_state["tokens_used_in"] += result.input_tokens
        st.session_state["tokens_used_out"] += result.output_tokens
        st.session_state["cost_usd"] += result.cost_usd

        if result.warnings:
            for w in result.warnings:
                st.warning(w)
        st.success(
            f"Generated {len(result.df):,} synthetic records in {result.batches} batch(es) · "
            f"{result.input_tokens + result.output_tokens:,} tokens · ${result.cost_usd:.4f}"
        )

    synth = st.session_state.get("synthetic_df")
    if synth is None or synth.empty:
        st.caption("Synthetic dataset will appear here.")
        return

    section("Synthetic data preview")
    st.dataframe(synth.head(15), width="stretch", hide_index=True)

    csv = synth.to_csv(index=False).encode()
    st.download_button("⬇ Download synthetic.csv", data=csv,
                       file_name="synthetic_fraud.csv", mime="text/csv")


# ─── Tab 03 — Validate ────────────────────────────────────────────────────────

def tab_validate() -> None:
    st.markdown("### 03 · Statistical validation")

    real = st.session_state.get("real_df")
    synth = st.session_state.get("synthetic_df")
    if real is None or synth is None or synth.empty:
        st.info("Generate a synthetic dataset on tab 02 first.")
        return

    ks_results = validator.ks_test_features(real, synth)
    summary = validator.fidelity_summary(ks_results)
    corr_delta = validator.correlation_delta(real, synth)

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Avg KS statistic",
                              f"{summary['avg_ks']:.3f}",
                              foot=f"across {summary['n']} features",
                              variant="accent" if summary['avg_ks'] < 0.15 else
                                      "warn" if summary['avg_ks'] < 0.30 else "danger"),
                     unsafe_allow_html=True)
    cols[1].markdown(kpi_card("High fidelity", f"{summary['high']}",
                              foot="KS < 0.15", variant="accent"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Medium", f"{summary['medium']}",
                              foot="0.15 ≤ KS < 0.30", variant="warn"), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("Low", f"{summary['low']}",
                              foot="KS ≥ 0.30",
                              variant="danger" if summary['low'] > 0 else ""),
                     unsafe_allow_html=True)

    section("KS statistic per feature · lower is better")
    df_ks = pd.DataFrame([{"feature": r.feature, "ks": r.ks_stat,
                           "fidelity": r.fidelity, "p": r.p_value} for r in ks_results])
    color_map = {"high": ACCENT, "medium": AMBER, "low": RED}
    fig = px.bar(df_ks, x="feature", y="ks", color="fidelity",
                 color_discrete_map=color_map,
                 hover_data={"p": ":.3f", "ks": ":.3f"})
    fig.add_hline(y=0.15, line_dash="dash", line_color=ACCENT,
                  annotation_text="0.15", annotation_position="top right",
                  annotation_font=dict(color=ACCENT))
    fig.add_hline(y=0.30, line_dash="dash", line_color=RED,
                  annotation_text="0.30", annotation_position="top right",
                  annotation_font=dict(color=RED))
    fig.update_yaxes(range=[0, max(0.5, df_ks["ks"].max() * 1.15)])
    st.plotly_chart(style_fig(fig), width="stretch")

    section("Distribution overlay")
    feat = st.selectbox("Feature", data_utils.FEATURE_COLUMNS,
                        key="validate_feat", label_visibility="collapsed")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=real[feat], name="Real",
                                opacity=0.75, marker_color=ACCENT, nbinsx=40,
                                histnorm="probability density"))
    fig2.add_trace(go.Histogram(x=synth[feat], name="Synthetic",
                                opacity=0.75, marker_color=BLUE, nbinsx=40,
                                histnorm="probability density"))
    fig2.update_layout(barmode="overlay", bargap=0.02)
    st.plotly_chart(style_fig(fig2, f"{feat} · real vs synthetic"), width="stretch")

    section(f"Correlation matrices · mean Δ = {corr_delta:.3f}")
    c1, c2 = st.columns(2)
    real_corr = validator.correlation_matrix(real)
    synth_corr = validator.correlation_matrix(synth)
    gold_scale = [[0.0, "#3a2a55"], [0.5, SURFACE_2], [1.0, ACCENT]]
    blue_scale = [[0.0, "#3a2a55"], [0.5, SURFACE_2], [1.0, BLUE]]
    fig_a = px.imshow(real_corr, color_continuous_scale=gold_scale,
                      zmin=-1, zmax=1, aspect="auto", title="Real")
    fig_b = px.imshow(synth_corr, color_continuous_scale=blue_scale,
                      zmin=-1, zmax=1, aspect="auto", title="Synthetic")
    c1.plotly_chart(style_fig(fig_a), width="stretch")
    c2.plotly_chart(style_fig(fig_b), width="stretch")


# ─── Tab 04 — Model Comparison ────────────────────────────────────────────────

def tab_model() -> None:
    st.markdown("### 04 · Model comparison · the money slide")

    real = st.session_state.get("real_df")
    synth = st.session_state.get("synthetic_df")
    if real is None or synth is None or synth.empty:
        st.info("Need both real and synthetic datasets — load tabs 01 & 02 first.")
        return

    if st.button("▶ Train & compare models"):
        with st.spinner("Training 3 GradientBoostingClassifiers · evaluating on holdout…"):
            comparison = ml_model.compare_models(real, synth, test_size=0.25)
            st.session_state["comparison"] = comparison

    comparison = st.session_state.get("comparison")
    if comparison is None:
        st.caption("Click the button to run the head-to-head.")
        return

    section(f"AUC-ROC · evaluated on {comparison.test_size} held-out real transactions")
    cols = st.columns(3)
    for col, r in zip(cols, comparison.regimes):
        is_best = r.regime == comparison.best_regime
        variant = "accent" if is_best else ""
        label = ("★ " if is_best else "") + r.label
        col.markdown(
            kpi_card(label, f"{r.auc:.3f}",
                     foot=f"trained on {r.n_train:,} rows", variant=variant),
            unsafe_allow_html=True,
        )

    section("ROC curves")
    fig = go.Figure()
    palette = {"real_only": ACCENT, "synthetic_only": BLUE, "combined": AMBER}
    for r in comparison.regimes:
        is_best = r.regime == comparison.best_regime
        fig.add_trace(go.Scatter(
            x=r.fpr, y=r.tpr, mode="lines", name=f"{r.label} (AUC {r.auc:.3f})",
            line=dict(color=palette.get(r.regime, ACCENT),
                      width=3 if is_best else 2),
        ))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color=BORDER, dash="dash"), showlegend=False))
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1])
    st.plotly_chart(style_fig(fig), width="stretch")

    section("Metric breakdown")
    long_df = ml_model.metrics_long_form(comparison)
    fig2 = px.bar(long_df, x="metric", y="value", color="regime", barmode="group",
                  color_discrete_map={
                      "Real only": ACCENT,
                      "Synthetic only": BLUE,
                      "Real + Synthetic": AMBER,
                  })
    fig2.update_yaxes(range=[0, 1.05])
    st.plotly_chart(style_fig(fig2), width="stretch")

    st.markdown(
        f"""
        <div class='sg-insight'>
          <div class='label'>Key insight</div>
          {comparison.insight}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    init_state()
    header()
    sidebar()

    t1, t2, t3, t4 = st.tabs([
        "01 · Upload Data",
        "02 · Generate Synthetic",
        "03 · Validate",
        "04 · Model Comparison",
    ])
    with t1:
        tab_upload()
    with t2:
        tab_generate()
    with t3:
        tab_validate()
    with t4:
        tab_model()


if __name__ == "__main__":
    main()
