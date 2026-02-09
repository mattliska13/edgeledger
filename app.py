# app.py  (Premium v2 wrapper + AI Assist; drop-in around your existing modules)
import os
import time
from datetime import datetime
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Optional OpenAI (AI Assist layer). App still runs without it.
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================================================
# Page + Premium Theme (SAFE: UI only)
# =========================================================
st.set_page_config(page_title="EdgeLedger", layout="wide", initial_sidebar_state="expanded")

PREMIUM_CSS = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

:root{
  --bg0:#070b14; --bg1:#0b1220; --bg2:#0f172a;
  --card:#0b1220; --stroke: rgba(148,163,184,0.18);
  --text:#e5e7eb; --muted:#94a3b8; --good:#22c55e; --bad:#ef4444;
}

html, body, [class*="css"]  { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1400px; }
@media (min-width: 1400px) { .block-container { max-width: 1700px; } }

section[data-testid="stSidebar"] {
  background: radial-gradient(1200px 800px at 10% 10%, rgba(59,130,246,0.12), transparent 50%),
              linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.big-title { font-size: 1.9rem; font-weight: 950; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.subtle { color: var(--muted); font-size: 0.95rem; margin-bottom: 0.35rem; }

.card {
  background: radial-gradient(900px 500px at 10% 0%, rgba(99,102,241,0.10), transparent 55%),
              linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 14px 16px;
  margin-bottom: 12px;
}
.pill { display:inline-block; padding:0.20rem 0.60rem; border-radius:999px;
  background:rgba(255,255,255,0.08); margin-right:0.4rem; font-size:0.85rem; }
.small {font-size:0.85rem; color:var(--muted);}
hr { border: none; border-top: 1px solid var(--stroke); margin: 10px 0; }

div[data-testid="stDataFrame"] { width: 100%; }
div[data-testid="stDataFrame"] > div { overflow-x: auto !important; }

@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .big-title { font-size: 1.35rem; }
  .subtle { font-size: 0.85rem; }
  .card { padding: 10px 10px; border-radius: 14px; }
  .stMarkdown p, .stCaption { font-size: 0.9rem; }
  canvas, svg, img { max-width: 100% !important; height: auto !important; }
  div[role="radiogroup"] label { padding: 10px 10px !important; margin: 6px 0 !important; border-radius: 12px !important; }
  div[role="radiogroup"] label p { font-size: 1.05rem !important; font-weight: 700 !important; }
}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# =========================================================
# Keys (Secrets -> Env -> Session override)
# =========================================================
def get_key(name: str, default: str = "") -> str:
    if name in st.session_state and str(st.session_state[name]).strip():
        return str(st.session_state[name]).strip()
    if hasattr(st, "secrets") and name in st.secrets:
        v = str(st.secrets.get(name, "")).strip()
        if v:
            return v
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

ODDS_API_KEY = get_key("ODDS_API_KEY", "")
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "") or get_key("DATAGOLF_KEY", "")

# Optional AI key
OPENAI_API_KEY = get_key("OPENAI_API_KEY", "")

# =========================================================
# HTTP (shared)
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/2.0 (streamlit)"})

def safe_get(url: str, params: dict | None = None, timeout: int = 25):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        ok = 200 <= r.status_code < 300
        try:
            payload = r.json()
        except Exception:
            payload = r.text
        return ok, r.status_code, payload, r.url
    except Exception as e:
        return False, 0, {"error": str(e)}, url

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# =========================================================
# AI Assist (SAFE: optional + isolated)
# Uses Structured Outputs so the model returns stable JSON :contentReference[oaicite:2]{index=2}
# =========================================================
AI_ENABLED_DEFAULT = False

def ai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

AI_JSON_SCHEMA = {
    "name": "edgeledger_row_enrichment",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {"type": "string"},
            "confidence": {"type": "number"},      # 0..1
            "reasoning_bullets": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6
            },
            "trend_flags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 6
            },
            "edge_adjustment": {"type": "number"}  # small, e.g. -0.01..+0.01
        },
        "required": ["label","confidence","reasoning_bullets","trend_flags","edge_adjustment"]
    }
}

@st.cache_data(ttl=60 * 15, show_spinner=False)
def ai_enrich_rows_cached(rows_json: str) -> list[dict]:
    """
    Input: JSON string of rows
    Output: list of enrichments aligned to input length
    """
    cli = ai_client()
    if cli is None:
        return []

    rows = json.loads(rows_json)
    # Keep payload light (avoid token bloat)
    prompt_rows = []
    for r in rows:
        prompt_rows.append({
            "Sport": r.get("Sport",""),
            "Market": r.get("Market",""),
            "Event": r.get("Event",""),
            "Selection": r.get("Selection",""),
            "Line": r.get("Line",""),
            "BestPrice": r.get("BestPrice",""),
            "YourProb": float(r.get("YourProb", np.nan)) if r.get("YourProb") is not None else None,
            "ImpliedBest": float(r.get("ImpliedBest", np.nan)) if r.get("ImpliedBest") is not None else None,
            "Edge": float(r.get("Edge", np.nan)) if r.get("Edge") is not None else None,
        })

    # One call for a batch (cheap + consistent)
    # IMPORTANT: This is "assist", not replacing your math.
    resp = cli.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": (
                "You are a sports betting analytics assistant. "
                "Given rows with probabilities and prices, produce concise enrichment for each row: "
                "1) short label, 2) confidence 0..1, 3) 3-6 bullets, 4) trend flags, "
                "5) tiny edge adjustment between -0.01 and +0.01 ONLY when strongly justified.\n\n"
                f"ROWS:\n{json.dumps(prompt_rows)}"
            )
        }],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "edgeledger_enrichment_batch",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": AI_JSON_SCHEMA["schema"],
                            }
                        },
                        "required": ["items"]
                    }
                }
            }
        },
    )
    # SDK returns parsed JSON in output_text for json_schema format
    out = json.loads(resp.output_text)
    return out.get("items", []) if isinstance(out, dict) else []

def apply_ai_enrichment(df_best: pd.DataFrame, sport: str, market: str, max_rows: int = 25) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    rows = []
    for _, r in df_best.head(max_rows).iterrows():
        rows.append({
            "Sport": sport,
            "Market": market,
            "Event": r.get("Event",""),
            "Selection": r.get("Outcome", r.get("Selection","")),
            "Line": r.get("LineBucket", r.get("Line","")),
            "BestPrice": r.get("BestPrice",""),
            "YourProb": r.get("YourProb", np.nan),
            "ImpliedBest": r.get("ImpliedBest", np.nan),
            "Edge": r.get("Edge", np.nan),
        })
    enrich = ai_enrich_rows_cached(json.dumps(rows))

    out = df_best.copy()
    if not enrich:
        return out

    # Map enrichments back onto first max_rows
    add_cols = ["AI_Label","AI_Conf","AI_Flags","AI_Notes","AI_EdgeAdj"]
    for c in add_cols:
        if c not in out.columns:
            out[c] = ""

    for i, e in enumerate(enrich[:max_rows]):
        out.loc[out.index[i], "AI_Label"] = e.get("label","")
        out.loc[out.index[i], "AI_Conf"] = float(e.get("confidence", 0.0))
        out.loc[out.index[i], "AI_Flags"] = ", ".join(e.get("trend_flags", []) or [])
        out.loc[out.index[i], "AI_Notes"] = " • ".join(e.get("reasoning_bullets", []) or [])
        out.loc[out.index[i], "AI_EdgeAdj"] = float(e.get("edge_adjustment", 0.0))

    # Optional: apply adjustment as a separate column (do NOT overwrite core Edge unless you explicitly want it)
    out["Edge_AI"] = pd.to_numeric(out.get("Edge", np.nan), errors="coerce")
    out["Edge_AI"] = out["Edge_AI"] + pd.to_numeric(out["AI_EdgeAdj"], errors="coerce").fillna(0.0)
    return out

# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Edge = YourProb − ImpliedProb(best price)</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
show_non_value = st.sidebar.checkbox("Show non-value rows (Edge ≤ 0)", value=False)

mode = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA", "Tracker"], index=0)

with st.sidebar.expander("API Keys (session-only override)", expanded=False):
    st.caption("If Secrets aren’t set, paste keys here (session-only).")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY / DATAGOLF_API_KEY", value=DATAGOLF_API_KEY or "", type="password")
    oa_in = st.text_input("OPENAI_API_KEY (optional AI Assist)", value=OPENAI_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()
    if oa_in.strip():
        st.session_state["OPENAI_API_KEY"] = oa_in.strip()
        OPENAI_API_KEY = oa_in.strip()

with st.sidebar.expander("AI Assist (optional)", expanded=False):
    ai_on = st.toggle("Enable AI Assist (labels + trend notes)", value=AI_ENABLED_DEFAULT)
    st.caption("AI Assist never replaces your math. It adds notes + optional tiny Edge_AI adjustment.")

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Ranked by **Edge = YourProb − ImpliedProb(best price)**. "
    "DK/FD only. Modules run independently. AI Assist is optional and non-destructive."
)

# =========================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# PASTE YOUR EXISTING (WORKING) CORE MODULES HERE
# - tracker functions + file I/O
# - odds math + normalizers
# - The Odds API fetchers (v4) :contentReference[oaicite:3]{index=3}
# - build_game_lines_board / build_props_board
# - PGA DataGolf module
# - render_tracker()
#
# IMPORTANT: Do not change your existing logic.
# Only ensure these functions exist:
#   - build_game_lines_board(sport, bet_type) -> (df_best, err)
#   - build_props_board(sport, prop_label, max_events_scan) -> (df_best, err)
#   - build_pga_board() -> (out, err)
#   - render_tracker()
# and these dicts exist:
#   - SPORT_KEYS_LINES, GAME_MARKETS
#   - SPORT_KEYS_PROPS, PROP_MARKETS
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =========================================================

# -------------------------
# PLACEHOLDER GUARDS
# -------------------------
if "build_game_lines_board" not in globals():
    st.error("Paste your existing core logic where indicated (build_game_lines_board missing).")
    st.stop()

# =========================================================
# Premium Render helpers (SAFE)
# =========================================================
def card_open():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def bar_prob(df, label_col, prob_col_percent, title):
    if df.empty:
        return
    fig = plt.figure()
    vals = pd.to_numeric(df[prob_col_percent], errors="coerce").fillna(0.0).values
    labs = df[label_col].astype(str).values
    plt.barh(labs, vals)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(100))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# MAIN
# =========================================================
if mode == "Tracker":
    render_tracker()
    st.stop()

if mode == "Game Lines":
    card_open()

    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_bettype")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="gl_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_top25")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No game lines available."))
        card_close()
        st.stop()

    # Optional AI overlay (does NOT break core columns)
    if ai_on:
        df_best = apply_ai_enrichment(df_best, sport=sport, market=bet_type, max_rows=25)

    st.subheader(f"{sport} — {bet_type} (DK/FD) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per game per market. Ranked by Edge. AI Assist adds notes only.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    base_cols = ["Event", "Outcome"]
    if "LineBucket" in top.columns and top["LineBucket"].notna().any():
        base_cols += ["LineBucket"]
    base_cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]

    if ai_on:
        base_cols += ["AI_Label", "AI_Conf", "AI_Flags", "AI_Notes", "Edge_AI"]

    base_cols = [c for c in base_cols if c in top.columns]
    st.dataframe(top[base_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    if "YourProb%" in chart.columns:
        bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    if "Implied%" in chart.columns:
        bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = base_cols  # same set
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    card_close()

elif mode == "Player Props":
    card_open()

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_prop")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="pp_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_top25")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 14, 8, key="pp_scan")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No props returned for DK/FD on scanned events."))
        card_close()
        st.stop()

    if ai_on:
        df_best = apply_ai_enrichment(df_best, sport=sport, market=prop_label, max_rows=25)

    st.subheader(f"{sport} — Player Props ({prop_label}) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per player per market per game. Ranked by Edge. AI Assist adds notes only.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"]
    if "LineBucket" in top.columns and top["LineBucket"].notna().any():
        cols += ["LineBucket"]
    cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    if ai_on:
        cols += ["AI_Label", "AI_Conf", "AI_Flags", "AI_Notes", "Edge_AI"]
    cols = [c for c in cols if c in top.columns]

    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    if "YourProb%" in chart.columns:
        bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    if "Implied%" in chart.columns:
        bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        st.dataframe(snap[cols], use_container_width=True, hide_index=True)

    card_close()

else:
    card_open()

    if not DATAGOLF_API_KEY.strip():
        st.warning('Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY). PGA is hidden until then.')
        card_close()
        st.stop()

    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")
    st.caption("Top picks for Win / Top-10 + One-and-Done using DataGolf model probabilities + SG splits + fit/history/form proxies.")

    out, err = build_pga_board()
    if isinstance(out, dict) and "winners" in out:
        winners = out["winners"]
        top10s = out["top10s"]
        oad = out["oad"]

        st.markdown("### 🏆 Best Win Picks (Top 10)")
        st.dataframe(winners, use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Best Top-10 Picks (Top 10)")
        st.dataframe(top10s, use_container_width=True, hide_index=True)

        st.markdown("### 🧳 Best One-and-Done Options (Top 7)")
        st.dataframe(oad, use_container_width=True, hide_index=True)

    else:
        st.warning(err.get("error", "No PGA data available right now."))

    card_close()
