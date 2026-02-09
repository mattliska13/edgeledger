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

import os
import time
import re
from html import unescape as _html_unescape
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================================================
# Page + Responsive Theme
# =========================================================
st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1200px; }
@media (min-width: 1200px) { .block-container { max-width: 1600px; } }

.big-title { font-size: 1.9rem; font-weight: 900; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.subtle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.35rem; }

.card { background: #0b1220; border: 1px solid rgba(148,163,184,0.18); border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; }
.pill { display:inline-block; padding:0.18rem 0.55rem; border-radius:999px; background:rgba(255,255,255,0.08); margin-right:0.4rem; font-size:0.85rem; }
.small {font-size:0.85rem; color:#94a3b8;}
hr { border: none; border-top: 1px solid rgba(148,163,184,0.18); margin: 10px 0; }

div[data-testid="stDataFrame"] { width: 100%; }
div[data-testid="stDataFrame"] > div { overflow-x: auto !important; }

/* Mobile improvements: bigger radio + spacing */
@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .big-title { font-size: 1.35rem; }
  .subtle { font-size: 0.85rem; }
  .card { padding: 10px 10px; border-radius: 14px; }
  .stMarkdown p, .stCaption { font-size: 0.9rem; }
  canvas, svg, img { max-width: 100% !important; height: auto !important; }

  /* radio buttons: bigger tap targets */
  div[role="radiogroup"] label {
    padding: 10px 10px !important;
    margin: 6px 0 !important;
    border-radius: 12px !important;
  }
  div[role="radiogroup"] label p {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
  }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# Tracker (does not change odds/PGA logic)
# =========================================================
TRACK_FILE = "tracker.csv"

TRACK_COLUMNS = [
    "LoggedAt", "Mode", "Sport", "Market", "Event", "Selection", "Line",
    "BestBook", "BestPrice", "YourProb", "Implied", "Edge", "EV",
    "Status", "Result"
]


def _load_tracker() -> pd.DataFrame:
    if "tracker_df" in st.session_state and isinstance(st.session_state["tracker_df"], pd.DataFrame):
        return st.session_state["tracker_df"].copy()

    if os.path.exists(TRACK_FILE):
        try:
            df = pd.read_csv(TRACK_FILE)
        except Exception:
            df = pd.DataFrame(columns=TRACK_COLUMNS)
    else:
        df = pd.DataFrame(columns=TRACK_COLUMNS)

    for c in TRACK_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan

    st.session_state["tracker_df"] = df.copy()
    return df


def _save_tracker(df: pd.DataFrame):
    st.session_state["tracker_df"] = df.copy()
    try:
        df.to_csv(TRACK_FILE, index=False)
    except Exception:
        pass


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _windows(df: pd.DataFrame):
    d = df.copy()
    d["LoggedAt_dt"] = _to_dt(d["LoggedAt"])
    d = d.dropna(subset=["LoggedAt_dt"])
    now = pd.Timestamp.now()
    today = now.normalize()

    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)

    return {
        "Today": d[d["LoggedAt_dt"] >= today],
        "This Week": d[d["LoggedAt_dt"] >= week_start],
        "This Month": d[d["LoggedAt_dt"] >= month_start],
        "This Year": d[d["LoggedAt_dt"] >= year_start],
    }


def _summary_pick_rate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Window", "Mode", "Picks", "Graded", "Wins", "Losses", "Pushes", "HitRate%"])
    x = df.copy()
    x["Picks"] = 1
    x["Graded"] = (x["Status"] == "Graded").astype(int)
    x["Wins"] = (x["Result"] == "W").astype(int)
    x["Losses"] = (x["Result"] == "L").astype(int)
    x["Pushes"] = (x["Result"] == "P").astype(int)

    agg = x.groupby(["Mode"], dropna=False).agg(
        Picks=("Picks", "sum"),
        Graded=("Graded", "sum"),
        Wins=("Wins", "sum"),
        Losses=("Losses", "sum"),
        Pushes=("Pushes", "sum"),
    ).reset_index()

    denom = (agg["Wins"] + agg["Losses"] + agg["Pushes"]).replace(0, np.nan)
    agg["HitRate%"] = ((agg["Wins"] / denom) * 100.0).round(1).fillna(0.0)

    agg.insert(0, "Window", label)
    return agg.sort_values(["Picks", "HitRate%"], ascending=False)


def tracker_log_rows(rows: List[Dict]):
    df = _load_tracker()
    add = pd.DataFrame(rows)
    for c in TRACK_COLUMNS:
        if c not in add.columns:
            add[c] = np.nan
    add = add[TRACK_COLUMNS].copy()
    df = pd.concat([df, add], ignore_index=True)
    _save_tracker(df)
    return df


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
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "")
if not DATAGOLF_API_KEY:
    DATAGOLF_API_KEY = get_key("DATAGOLF_KEY", "")

# =========================================================
# HTTP
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/1.0 (streamlit)"})

UFC_SESSION = requests.Session()
UFC_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})


def safe_get(url: str, params: dict, timeout: int = 25):
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


def is_list_of_dicts(x):
    return isinstance(x, list) and (len(x) == 0 or isinstance(x[0], dict))


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================================================
# Odds math
# =========================================================
def american_to_implied(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def clamp01(x):
    return np.clip(x, 0.001, 0.999)


def pct01_to_100(series01):
    return (pd.to_numeric(series01, errors="coerce") * 100.0).round(2)


def line_bucket_half_point(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    return round(v * 2.0) / 2.0


# =========================================================
# API Config (The Odds API)
# =========================================================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # ONLY these

SPORT_KEYS_LINES = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}
SPORT_KEYS_PROPS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

GAME_MARKETS = {
    "Moneyline": "h2h",
    "Spreads": "spreads",
    "Totals": "totals",
}

PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>EDGE = YourProb − ImpliedProb(best price)</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
show_non_value = st.sidebar.checkbox("Show non-value rows (Edge ≤ 0)", value=False)

# ✅ Added UFC to radio buttons (no impact to other modules)
mode = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA", "UFC", "Tracker"], index=0)

with st.sidebar.expander("API Keys (session-only override)", expanded=False):
    st.caption("If Secrets aren’t set, paste keys here (session-only).")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY / DATAGOLF_API_KEY", value=DATAGOLF_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Ranked by **Edge = YourProb − ImpliedProb(best price)**. "
    "**Strict contradiction removal**: only one side per game/market (and one side per player/market). "
    "DK/FD only. Game lines / props / PGA / UFC run independently."
)

if not ODDS_API_KEY.strip() and mode not in ["Tracker", "UFC", "PGA"]:
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar expander.')
    st.stop()

# =========================================================
# Caching (daily)
# =========================================================
@st.cache_data(ttl=60 * 60 * 24)
def fetch_game_lines(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": ",".join(GAME_MARKETS.values()),
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}


@st.cache_data(ttl=60 * 60 * 24)
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}


@st.cache_data(ttl=60 * 60 * 24)
def fetch_event_odds(sport_key: str, event_id: str, market_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": market_key,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}


# =========================================================
# Normalizers
# =========================================================
def normalize_lines(raw):
    rows = []
    if not is_list_of_dicts(raw):
        return pd.DataFrame()

    for ev in raw:
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}"
        commence = ev.get("commence_time")

        for bm in (ev.get("bookmakers", []) or []):
            book = bm.get("title") or bm.get("key")
            for mk in (bm.get("markets", []) or []):
                mkey = mk.get("key")
                for out in (mk.get("outcomes", []) or []):
                    line = out.get("point")
                    rows.append({
                        "Scope": "GameLine",
                        "Event": matchup,
                        "Commence": commence,
                        "Market": mkey,
                        "Outcome": out.get("name"),
                        "Line": line,
                        "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                        "Price": out.get("price"),
                        "Book": book,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Outcome", "Price"])


def normalize_props(event_payload):
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    matchup = f"{away} @ {home}"

    for bm in (event_payload.get("bookmakers", []) or []):
        book = bm.get("title") or bm.get("key")
        for mk in (bm.get("markets", []) or []):
            mkey = mk.get("key")
            for out in (mk.get("outcomes", []) or []):
                player = out.get("name")
                side = out.get("description")  # Over/Under usually; may be blank
                line = out.get("point")
                price = out.get("price")

                if player is None or price is None:
                    continue

                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": str(side) if side is not None else "",
                    "Line": line,
                    "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                    "Price": price,
                    "Book": book
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Player", "Price"])


# =========================================================
# Core Best-Bet Logic (Implied vs YourProb)
# =========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    return out


def compute_no_vig_within_book_two_way(df: pd.DataFrame, group_cols_book: list) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    sums = out.groupby(group_cols_book)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out


def estimate_your_prob(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
    """
    YourProb:
    - two-way: no-vig within each book, avg across books
    - one-way: market-consensus avg implied across books (enables value by shopping)
    """
    if df.empty:
        return df.copy()

    out = add_implied(df)
    side_col = "Outcome" if "Outcome" in out.columns else "Side"
    n_sides = out.groupby(key_cols)[side_col].transform(lambda s: s.nunique(dropna=True))
    out["_n_sides"] = pd.to_numeric(n_sides, errors="coerce").fillna(1)

    out = compute_no_vig_within_book_two_way(out, group_cols_book=book_cols)

    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")

    out["YourProb"] = np.where(out["_n_sides"] >= 2, nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    out = out.drop(columns=["_n_sides"], errors="ignore")
    return out


def best_price_and_edge(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Price"])

    idx = out.groupby(group_cols_best)["Price"].idxmax()
    best = out.loc[idx].copy()

    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook"})
    best["ImpliedBest"] = best["BestPrice"].apply(american_to_implied)
    best["ImpliedBest"] = clamp01(pd.to_numeric(best["ImpliedBest"], errors="coerce").fillna(0.5))

    best["Edge"] = (pd.to_numeric(best["YourProb"], errors="coerce") - pd.to_numeric(best["ImpliedBest"], errors="coerce"))
    best["EV"] = (best["Edge"] * 100.0)
    return best


def strict_no_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    STRICT contradiction removal:
    Keep exactly ONE row per contradiction group (max Edge), ignoring line differences.
    """
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    return out.loc[idx].sort_values("Edge", ascending=False)


def filter_value(df_best: pd.DataFrame, show_non_value: bool) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    out = out.sort_values("Edge", ascending=False)
    if show_non_value:
        return out
    return out[out["Edge"] > 0]


def decorate_probs(df_best: pd.DataFrame) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["YourProb%"] = pct01_to_100(out["YourProb"])
    out["Implied%"] = pct01_to_100(out["ImpliedBest"])
    out["Edge%"] = pct01_to_100(out["Edge"])
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").round(2)
    return out


# =========================================================
# Boards
# =========================================================
def build_game_lines_board(sport: str, bet_type: str):
    sport_key = SPORT_KEYS_LINES[sport]
    market_key = GAME_MARKETS[bet_type]

    res = fetch_game_lines(sport_key)
    if debug:
        st.json({"endpoint": "odds(game_lines)", "status": res["status"], "url": res["url"], "params": res["params"]})

    if not res["ok"] or not is_list_of_dicts(res["payload"]):
        return pd.DataFrame(), {"error": "Game lines API failed", "payload": res["payload"], "status": res["status"]}

    df = normalize_lines(res["payload"])
    if df.empty:
        return pd.DataFrame(), {"error": "No normalized game lines"}

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        return pd.DataFrame(), {"error": "No rows for selected market"}

    key_cols = ["Event", "Market", "Outcome"]
    book_cols = ["Event", "Market", "Book"]
    best_cols = ["Event", "Market", "Outcome"]

    if market_key in ["spreads", "totals"]:
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    contradiction_cols = ["Event", "Market"]
    df_best = strict_no_contradictions(df_best, contradiction_cols=contradiction_cols)

    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)
    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}


def build_props_board(sport: str, prop_label: str, max_events_scan: int = 8):
    sport_key = SPORT_KEYS_PROPS[sport]
    market_key = PROP_MARKETS[prop_label]

    ev_res = fetch_events(sport_key)
    if debug:
        st.json({"endpoint": "events", "status": ev_res["status"], "url": ev_res["url"], "params": ev_res["params"]})

    if not ev_res["ok"] or not is_list_of_dicts(ev_res["payload"]):
        return pd.DataFrame(), {"error": "Events API failed", "payload": ev_res["payload"], "status": ev_res["status"]}

    event_ids = [e.get("id") for e in ev_res["payload"] if isinstance(e, dict) and e.get("id")]
    event_ids = event_ids[: int(max_events_scan)]
    if not event_ids:
        return pd.DataFrame(), {"error": "No upcoming events"}

    all_rows = []
    call_log = []
    for eid in event_ids:
        r = fetch_event_odds(sport_key, eid, market_key)
        call_log.append({"event_id": eid, "market": market_key, "status": r["status"], "ok": r["ok"]})
        if not r["ok"] or not isinstance(r["payload"], dict):
            continue
        dfp = normalize_props(r["payload"])
        if not dfp.empty:
            all_rows.append(dfp)
        time.sleep(0.04)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {"error": "No props returned for DK/FD on scanned events (or market not posted yet).", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    best_cols = ["Event", "Market", "Player", "Side"]

    if "LineBucket" in df.columns and df["LineBucket"].notna().any():
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    contradiction_cols = ["Event", "Market", "Player"]
    df_best = strict_no_contradictions(df_best, contradiction_cols=contradiction_cols)

    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)
    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}


# =========================================================
# Charts
# =========================================================
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
# PGA — Advanced DataGolf Module (independent)
# =========================================================
DG_HOST = "https://feeds.datagolf.com"


def _dg_get(path: str, params: dict):
    url = f"{DG_HOST}{path}"
    ok, status, payload, final_url = safe_get(url, params=params, timeout=30)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}


def _dg_find_rows(payload):
    if isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], dict)):
        return payload, {}

    if not isinstance(payload, dict):
        return [], {}

    meta = {}
    for k in ["event_name", "last_updated", "models_available", "tour"]:
        if k in payload:
            meta[k] = payload.get(k)

    for key in ["data", "predictions", "preds", "players", "rows"]:
        if key in payload and isinstance(payload[key], list):
            rows = payload[key]
            if len(rows) == 0 or isinstance(rows[0], dict):
                return rows, meta

    prefer = ["baseline_history_fit", "baseline", "sg_total", "default"]
    for p in prefer:
        if p in payload and isinstance(payload[p], list):
            rows = payload[p]
            if len(rows) == 0 or isinstance(rows[0], dict):
                meta["model_used"] = p
                return rows, meta

    for k, v in payload.items():
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            meta["model_used"] = k
            return v, meta

    return [], meta


def _normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split()).lower()


def _first_col(df: pd.DataFrame, candidates: list):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_pga_board():
    if not DATAGOLF_API_KEY.strip():
        return pd.DataFrame(), {"error": 'Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY). PGA is hidden until then.'}

    pre_params = {
        "tour": "pga",
        "add_position": 10,
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    decomp_params = {"tour": "pga", "file_format": "json", "key": DATAGOLF_API_KEY}
    skill_params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}

    pre = _dg_get("/preds/pre-tournament", pre_params)
    dec = _dg_get("/preds/player-decompositions", decomp_params)
    skl = _dg_get("/preds/skill-ratings", skill_params)

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]},
            "dg_decomp": {"ok": dec["ok"], "status": dec["status"], "url": dec["url"]},
            "dg_skill": {"ok": skl["ok"], "status": skl["status"], "url": skl["url"]},
        })

    if not pre["ok"]:
        return pd.DataFrame(), {"error": "DataGolf pre-tournament call failed", "status": pre["status"], "payload": pre["payload"]}

    pre_rows, meta = _dg_find_rows(pre["payload"])
    if not pre_rows:
        return pd.DataFrame(), {"error": "No PGA prediction rows returned from DataGolf (parsed none).", "dg_meta": meta}

    df_pre = pd.DataFrame(pre_rows)
    name_col = _first_col(df_pre, ["player_name", "name", "golfer", "player"])
    if not name_col:
        return pd.DataFrame(), {"error": "Could not find player name column in DataGolf pre-tournament payload."}

    win_col = None
    top10_col = None
    for c in df_pre.columns:
        lc = str(c).lower()
        if win_col is None and "win" in lc:
            win_col = c
        if top10_col is None and ("top" in lc and "10" in lc):
            top10_col = c

    if not win_col:
        return pd.DataFrame(), {"error": "Could not locate win probability column in DataGolf pre-tournament data."}

    df_pre["Player"] = df_pre[name_col].astype(str)
    df_pre["PlayerKey"] = df_pre["Player"].apply(_normalize_name)
    df_pre["WinProb"] = pd.to_numeric(df_pre[win_col], errors="coerce") / 100.0
    df_pre["Top10Prob"] = pd.to_numeric(df_pre[top10_col], errors="coerce") / 100.0 if top10_col else np.nan

    df_skill = pd.DataFrame()
    if skl["ok"]:
        rows_s, _ = _dg_find_rows(skl["payload"])
        if rows_s:
            df_skill = pd.DataFrame(rows_s)
            nm = _first_col(df_skill, ["player_name", "name", "player"])
            if nm:
                df_skill["PlayerKey"] = df_skill[nm].astype(str).apply(_normalize_name)

    df_dec = pd.DataFrame()
    if dec["ok"]:
        rows_d, _ = _dg_find_rows(dec["payload"])
        if rows_d:
            df_dec = pd.DataFrame(rows_d)
            nm = _first_col(df_dec, ["player_name", "name", "player"])
            if nm:
                df_dec["PlayerKey"] = df_dec[nm].astype(str).apply(_normalize_name)

    df = df_pre[["Player", "PlayerKey", "WinProb", "Top10Prob"]].copy()

    if not df_skill.empty:
        rating_col = _first_col(df_skill, ["skill", "rating", "sg_total", "value"])
        if rating_col:
            tmp = df_skill[["PlayerKey", rating_col]].copy().rename(columns={rating_col: "SkillRating"})
            df = df.merge(tmp, on="PlayerKey", how="left")

    if not df_dec.empty:
        col_map_candidates = {
            "SG_T2G": ["sg_t2g", "sg_tee_to_green", "t2g", "tee_to_green"],
            "SG_Putt": ["sg_putt", "sg_putting", "putt", "putting"],
            "BogeyAvoid": ["bogey_avoid", "bogey_avoidance"],
            "CourseFit": ["course_fit", "fit"],
            "CourseHistory": ["course_history", "history"],
            "RecentForm": ["recent_form", "form", "current_form", "recent"],
        }
        tmp = df_dec.copy()
        keep_cols = ["PlayerKey"]
        ren = {}
        for out_name, cands in col_map_candidates.items():
            c = _first_col(tmp, cands)
            if c:
                keep_cols.append(c)
                ren[c] = out_name
        if len(keep_cols) > 1:
            tmp = tmp[keep_cols].rename(columns=ren)
            df = df.merge(tmp, on="PlayerKey", how="left")

    for c in ["SkillRating", "SG_T2G", "SG_Putt", "BogeyAvoid", "CourseFit", "CourseHistory", "RecentForm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    def z(x):
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return pd.Series(np.zeros(len(x)), index=x.index)
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if sd == 0 or np.isnan(sd):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - mu) / sd

    df["z_win"] = z(df["WinProb"])
    df["z_top10"] = z(df["Top10Prob"]) if df["Top10Prob"].notna().any() else 0.0
    df["z_skill"] = z(df["SkillRating"]) if "SkillRating" in df.columns else 0.0
    df["z_t2g"] = z(df["SG_T2G"]) if "SG_T2G" in df.columns else 0.0
    df["z_putt"] = z(df["SG_Putt"]) if "SG_Putt" in df.columns else 0.0
    df["z_bogey"] = z(df["BogeyAvoid"]) if "BogeyAvoid" in df.columns else 0.0
    df["z_fit"] = z(df["CourseFit"]) if "CourseFit" in df.columns else 0.0
    df["z_hist"] = z(df["CourseHistory"]) if "CourseHistory" in df.columns else 0.0
    df["z_form"] = z(df["RecentForm"]) if "RecentForm" in df.columns else 0.0

    df["WinScore"] = (
        0.55 * df["z_win"] +
        0.12 * df["z_t2g"] +
        0.08 * df["z_putt"] +
        0.08 * df["z_skill"] +
        0.06 * df["z_fit"] +
        0.06 * df["z_hist"] +
        0.05 * df["z_bogey"]
    )

    df["Top10Score"] = (
        0.55 * df["z_top10"] +
        0.12 * df["z_t2g"] +
        0.08 * df["z_putt"] +
        0.07 * df["z_skill"] +
        0.06 * df["z_fit"] +
        0.06 * df["z_hist"] +
        0.06 * df["z_bogey"]
    )

    df["OADScore"] = (
        0.55 * df["Top10Score"] +
        0.25 * df["WinScore"] +
        0.20 * df["z_t2g"]
    )

    df["Win%"] = pct01_to_100(df["WinProb"])
    df["Top10%"] = pct01_to_100(df["Top10Prob"]) if df["Top10Prob"].notna().any() else np.nan

    winners = df.sort_values("WinScore", ascending=False).head(10).copy()
    top10s = df.sort_values("Top10Score", ascending=False).head(10).copy()
    oad = df.sort_values("OADScore", ascending=False).head(7).copy()

    return {"winners": winners, "top10s": top10s, "oad": oad, "meta": meta}, {}


# =========================================================
# UFC Module (UFCStats-first, OddsAPI fallback)  ✅ isolated
# =========================================================
UFCSTATS_BASE = "https://ufcstats.com"
UFC_UPCOMING_URL = f"{UFCSTATS_BASE}/statistics/events/upcoming?page=all"

# Odds API MMA fallback (DK/FD moneylines)
MMA_SPORT_KEY = "mma_mixed_martial_arts"
MMA_MARKET_H2H = "h2h"


def _strip(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.utcnow()


@st.cache_data(ttl=60 * 10)
def ufc_fetch_text(url: str) -> Dict:
    """
    UFCStats fetch using UFC_SESSION (browser-ish headers).
    If UFCStats blocks Streamlit Cloud, r.status_code often 403/503 with Cloudflare text.
    """
    try:
        r = UFC_SESSION.get(url, timeout=20, allow_redirects=True)
        txt = r.text if isinstance(r.text, str) else ""
        return {"ok": 200 <= r.status_code < 300, "status": r.status_code, "url": r.url, "text": txt[:2_500_000]}
    except Exception as e:
        return {"ok": False, "status": 0, "url": url, "error": str(e), "text": ""}


def _ufcstats_blocked(html_text: str, status: int) -> bool:
    if status in (403, 429, 503):
        return True
    t = (html_text or "").lower()
    # common block pages
    if "cloudflare" in t or "attention required" in t or "access denied" in t:
        return True
    if "checking your browser" in t or "ddos" in t:
        return True
    return False


def _parse_event_rows(html_text: str) -> List[Dict]:
    out = []
    for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S):
        row = m.group(1)

        link_m = re.search(r"href\s*=\s*['\"]([^'\"]*event-details[^'\"]*)['\"]", row, flags=re.I)
        if not link_m:
            continue

        url = link_m.group(1)
        if url.startswith("/"):
            url = UFCSTATS_BASE + url
        url = url.replace("http://", "https://")

        title_m = re.search(r"href\s*=\s*['\"][^'\"]*event-details[^'\"]*['\"]\s*>(.*?)</a>", row, flags=re.I | re.S)
        title = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", title_m.group(1))) if title_m else "")

        date_m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", row)
        dt = None
        if date_m:
            dt = pd.to_datetime(date_m.group(1), errors="coerce")

        out.append({"title": title or "UFC Event", "date": dt, "url": url})

    seen = set()
    uniq = []
    for r in out:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        uniq.append(r)
    return uniq


def _parse_fight_links(event_html: str) -> List[str]:
    links = re.findall(r"href\s*=\s*['\"]([^'\"]*fight-details[^'\"]*)['\"]", event_html, flags=re.I)
    out = []
    for u in links:
        if u.startswith("/"):
            u = UFCSTATS_BASE + u
        if u.startswith("http"):
            u = u.replace("http://", "https://")
            out.append(u)

    seen = set()
    uniq = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq


def ufc_list_upcoming_events(upcoming_html: str) -> List[Dict]:
    events = _parse_event_rows(upcoming_html)
    if not events:
        return []

    cut = _utc_now() - pd.Timedelta(hours=8)
    cleaned = []
    for e in events:
        d = e.get("date")
        if isinstance(d, pd.Timestamp) and pd.notna(d):
            if d >= cut.normalize() - pd.Timedelta(days=2):
                cleaned.append(e)
        else:
            cleaned.append(e)

    def sort_key(x):
        d = x.get("date")
        if isinstance(d, pd.Timestamp) and pd.notna(d):
            return d.value
        return pd.Timestamp.max.value

    return sorted(cleaned, key=sort_key)


def _num_from_pct(s: str) -> float:
    s = (s or "").strip().replace("%", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def _num(s: str) -> float:
    try:
        return float((s or "").strip())
    except Exception:
        return np.nan


@st.cache_data(ttl=60 * 60 * 6)
def ufc_fetch_fighter(fighter_url: str) -> Dict:
    r = ufc_fetch_text(fighter_url)
    if not r.get("ok"):
        return {"ok": False, "url": fighter_url, "error": r.get("error", ""), "status": r.get("status", 0)}

    html = r.get("text", "")

    name = ""
    m = re.search(r"<span[^>]*class=['\"]b-content__title-highlight['\"][^>]*>(.*?)</span>", html, flags=re.I | re.S)
    if m:
        name = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", m.group(1))))

    rec = ""
    m = re.search(r"Record:\s*</span>\s*<span[^>]*>(.*?)</span>", html, flags=re.I | re.S)
    if m:
        rec = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", m.group(1))))

    bio = {}
    for li in re.findall(r"<li[^>]*class=['\"]b-list__box-list-item['\"][^>]*>(.*?)</li>", html, flags=re.I | re.S):
        txt = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", li)))
        if ":" in txt:
            k, v = txt.split(":", 1)
            bio[_strip(k)] = _strip(v)

    stats = {}
    for box in re.findall(r"<li[^>]*class=['\"]b-list__box-list-item_type_block['\"][^>]*>(.*?)</li>", html, flags=re.I | re.S):
        txt = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", box)))
        if ":" in txt:
            k, v = txt.split(":", 1)
            stats[_strip(k)] = _strip(v)

    def parse_reach(x):
        x = (x or "").replace('"', "").strip()
        try:
            return float(x)
        except Exception:
            return np.nan

    def parse_height(x):
        x = (x or "").strip()
        m2 = re.search(r"(\d+)\s*'\s*(\d+)", x)
        if not m2:
            return np.nan
        ft = float(m2.group(1))
        inch = float(m2.group(2))
        return ft * 12.0 + inch

    def parse_dob_to_age(dob):
        dt = pd.to_datetime(dob, errors="coerce")
        if pd.isna(dt):
            return np.nan
        return float((_utc_now() - dt).days / 365.25)

    reach = parse_reach(bio.get("Reach", ""))
    height = parse_height(bio.get("Height", ""))
    stance = bio.get("STANCE", bio.get("Stance", ""))
    age = parse_dob_to_age(bio.get("DOB", ""))

    slpm = _num(stats.get("SLpM", ""))
    sapm = _num(stats.get("SApM", ""))
    str_acc = _num_from_pct(stats.get("Str. Acc.", stats.get("Str. Acc", "")))
    str_def = _num_from_pct(stats.get("Str. Def", stats.get("Str. Def.", "")))
    td_avg = _num(stats.get("TD Avg.", stats.get("TD Avg", "")))
    td_acc = _num_from_pct(stats.get("TD Acc.", stats.get("TD Acc", "")))
    td_def = _num_from_pct(stats.get("TD Def.", stats.get("TD Def", "")))
    sub_avg = _num(stats.get("Sub. Avg.", stats.get("Sub. Avg", "")))

    w = l = d = np.nan
    mrec = re.search(r"(\d+)\s*-\s*(\d+)(?:\s*-\s*(\d+))?", rec)
    if mrec:
        w = float(mrec.group(1))
        l = float(mrec.group(2))
        d = float(mrec.group(3)) if mrec.group(3) else 0.0

    win_pct = np.nan
    if not np.isnan(w) and not np.isnan(l):
        denom = w + l + (d if not np.isnan(d) else 0.0)
        if denom > 0:
            win_pct = w / denom

    return {
        "ok": True,
        "url": fighter_url,
        "name": name,
        "record": rec,
        "W": w, "L": l, "D": d, "WinPct": win_pct,
        "Age": age,
        "Reach": reach,
        "HeightIn": height,
        "Stance": stance,
        "SLpM": slpm,
        "SApM": sapm,
        "StrAcc": str_acc,
        "StrDef": str_def,
        "TDAvg": td_avg,
        "TDAcc": td_acc,
        "TDDef": td_def,
        "SubAvg": sub_avg,
    }


def _parse_fight_fighters(fight_html: str) -> Optional[Tuple[str, str]]:
    links = re.findall(r"href\s*=\s*['\"]([^'\"]*fighter-details[^'\"]*)['\"]", fight_html, flags=re.I)
    if len(links) < 2:
        return None
    a, b = links[0], links[1]
    if a.startswith("/"):
        a = UFCSTATS_BASE + a
    if b.startswith("/"):
        b = UFCSTATS_BASE + b
    a = a.replace("http://", "https://")
    b = b.replace("http://", "https://")
    return a, b


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except Exception:
        return 0.5


def ufc_predict_pair(f1: Dict, f2: Dict) -> Dict:
    def nz(v, default=0.0):
        try:
            if pd.isna(v):
                return default
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            return default

    age_diff = nz(f1.get("Age")) - nz(f2.get("Age"))
    reach_diff = nz(f1.get("Reach")) - nz(f2.get("Reach"))
    winpct_diff = nz(f1.get("WinPct")) - nz(f2.get("WinPct"))

    strike_1 = nz(f1.get("SLpM")) - nz(f1.get("SApM"))
    strike_2 = nz(f2.get("SLpM")) - nz(f2.get("SApM"))
    strike_diff = strike_1 - strike_2
    acc_def_diff = (nz(f1.get("StrAcc")) + nz(f1.get("StrDef"))) - (nz(f2.get("StrAcc")) + nz(f2.get("StrDef")))

    grap_1 = (0.9 * nz(f1.get("TDAvg")) + 0.03 * nz(f1.get("TDAcc")) + 0.03 * nz(f1.get("TDDef")) + 0.6 * nz(f1.get("SubAvg")))
    grap_2 = (0.9 * nz(f2.get("TDAvg")) + 0.03 * nz(f2.get("TDAcc")) + 0.03 * nz(f2.get("TDDef")) + 0.6 * nz(f2.get("SubAvg")))
    grap_diff = grap_1 - grap_2

    stance_bonus = 0.0
    s1 = (f1.get("Stance") or "").lower()
    s2 = (f2.get("Stance") or "").lower()
    if s1 and s2 and s1 != s2:
        stance_bonus = 0.05

    score = (
        0.55 * strike_diff +
        0.20 * grap_diff +
        0.18 * winpct_diff +
        0.07 * (reach_diff / 10.0) +
        -0.05 * (age_diff / 10.0) +
        0.05 * (acc_def_diff / 100.0) +
        stance_bonus
    )

    pA = _sigmoid(score)
    pick = f1.get("name") if pA >= 0.5 else f2.get("name")
    prob = float(max(0.01, min(0.99, pA if pick == f1.get("name") else 1.0 - pA)))

    # Finish/Decision proxy
    finish_potential_A = 0.6 * max(0.0, nz(f1.get("SLpM")) * (nz(f1.get("StrAcc")) / 100.0)) + 0.4 * max(
        0.0, nz(f1.get("TDAvg")) * (nz(f1.get("TDAcc")) / 100.0) + nz(f1.get("SubAvg"))
    )
    finish_potential_B = 0.6 * max(0.0, nz(f2.get("SLpM")) * (nz(f2.get("StrAcc")) / 100.0)) + 0.4 * max(
        0.0, nz(f2.get("TDAvg")) * (nz(f2.get("TDAcc")) / 100.0) + nz(f2.get("SubAvg"))
    )
    durability_A = (nz(f1.get("StrDef")) + nz(f1.get("TDDef"))) / 200.0
    durability_B = (nz(f2.get("StrDef")) + nz(f2.get("TDDef"))) / 200.0

    method = "Decision"
    if pick == f1.get("name"):
        if (finish_potential_A - finish_potential_B) > 0.35 and durability_B < 0.55:
            method = "ITD (Proxy)"
    else:
        if (finish_potential_B - finish_potential_A) > 0.35 and durability_A < 0.55:
            method = "ITD (Proxy)"

    return {
        "FighterA": f1.get("name", ""),
        "FighterB": f2.get("name", ""),
        "Pick": pick,
        "Prob": round(prob, 3),
        "Method": method,
        "ScoreDiff(A-B)": round(float(score), 3),
    }


@st.cache_data(ttl=60 * 30)
def ufc_build_picks_for_event(event_url: str) -> Dict:
    ev = ufc_fetch_text(event_url)
    if not ev.get("ok"):
        return {"ok": False, "error": "Could not load UFC event page.", "status": ev.get("status"), "url": ev.get("url")}

    txt = ev.get("text", "")
    if _ufcstats_blocked(txt, int(ev.get("status", 0) or 0)):
        return {"ok": False, "error": "UFCStats blocked (Cloudflare/Access denied).", "status": ev.get("status"), "url": ev.get("url")}

    fight_links = _parse_fight_links(txt)
    if not fight_links:
        return {"ok": False, "error": "No fights parsed on this event page (HTML changed or blocked).", "event_url": ev.get("url")}

    picks = []
    for fu in fight_links:
        fr = ufc_fetch_text(fu)
        if not fr.get("ok"):
            continue
        fh = fr.get("text", "")
        pair = _parse_fight_fighters(fh)
        if not pair:
            continue

        f1u, f2u = pair
        f1 = ufc_fetch_fighter(f1u)
        f2 = ufc_fetch_fighter(f2u)
        if not f1.get("ok") or not f2.get("ok"):
            continue

        pred = ufc_predict_pair(f1, f2)
        pred["FightURL"] = fu.replace("http://", "https://")
        picks.append(pred)
        time.sleep(0.02)

    if not picks:
        return {"ok": False, "error": "Fight list parsed but picks empty (fighter pages failed or markup changed).", "event_url": ev.get("url")}

    df = pd.DataFrame(picks).sort_values("Prob", ascending=False)
    return {"ok": True, "df": df}


# ---------- Odds API fallback (DK/FD H2H) ----------
@st.cache_data(ttl=60 * 10)
def mma_fetch_h2h_odds():
    if not ODDS_API_KEY.strip():
        return {"ok": False, "error": "Missing ODDS_API_KEY for MMA fallback."}

    url = f"{ODDS_HOST}/sports/{MMA_SPORT_KEY}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": MMA_MARKET_H2H,
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS,  # DK/FD only
    }
    ok, status, payload, final_url = safe_get(url, params=params, timeout=25)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}


def mma_normalize_h2h(payload) -> pd.DataFrame:
    if not is_list_of_dicts(payload):
        return pd.DataFrame()

    rows = []
    for ev in payload:
        home = ev.get("home_team")  # fighter A
        away = ev.get("away_team")  # fighter B
        matchup = f"{away} vs {home}" if home and away else (ev.get("id") or "Fight")
        commence = ev.get("commence_time")

        for bm in (ev.get("bookmakers", []) or []):
            book = bm.get("title") or bm.get("key")
            for mk in (bm.get("markets", []) or []):
                if mk.get("key") != "h2h":
                    continue
                for out in (mk.get("outcomes", []) or []):
                    rows.append({
                        "Event": matchup,
                        "Commence": commence,
                        "Fighter": out.get("name"),
                        "Price": out.get("price"),
                        "Book": book,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Fighter", "Price", "Book"])


def mma_build_consensus_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build consensus win probabilities from DK/FD, no-vig within each book, then average.
    Then pick the higher probability fighter per fight.
    """
    if df.empty:
        return df

    d = df.copy()
    d["Implied"] = d["Price"].apply(american_to_implied)
    d["Implied"] = clamp01(pd.to_numeric(d["Implied"], errors="coerce").fillna(0.5))

    # group by fight+book to remove vig
    sums = d.groupby(["Event", "Book"])["Implied"].transform("sum")
    d["NoVigProb"] = np.where(sums > 0, d["Implied"] / sums, np.nan)

    # avg across books
    d["ConsensusProb"] = d.groupby(["Event", "Fighter"])["NoVigProb"].transform("mean")
    out = d.groupby(["Event", "Fighter"], as_index=False).agg(
        ConsensusProb=("ConsensusProb", "mean"),
        BestPrice=("Price", "max"),
    )

    # select winner per event
    idx = out.groupby("Event")["ConsensusProb"].idxmax()
    picks = out.loc[idx].copy()
    picks["ConsensusProb%"] = (picks["ConsensusProb"] * 100.0).round(1)
    picks = picks.sort_values("ConsensusProb", ascending=False)
    return picks


def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC — Fight Picks")
    st.caption(
        "UFCStats-first for deep metrics (age/reach/stance + striking + takedowns). "
        "If UFCStats blocks the host, we fall back to DK/FD moneyline consensus via The Odds API."
    )

    # ---- Try UFCStats upcoming ----
    up = ufc_fetch_text(UFC_UPCOMING_URL)
    blocked = _ufcstats_blocked(up.get("text", ""), int(up.get("status", 0) or 0))

    if debug:
        st.json({"ufc_upcoming": {"ok": up.get("ok"), "status": up.get("status"), "url": up.get("url"), "blocked": blocked}})

    # If UFCStats works, proceed with the deep-metrics flow
    if up.get("ok") and not blocked:
        events = ufc_list_upcoming_events(up.get("text", ""))
        if not events:
            st.warning("No UFC events found on UFCStats (markup changed). Trying Odds API fallback…")
        else:
            labels = []
            for e in events[:20]:
                d = e.get("date")
                ds = d.strftime("%b %d, %Y") if isinstance(d, pd.Timestamp) and pd.notna(d) else ""
                labels.append(f"{e.get('title','UFC Event')} — {ds}".strip(" —"))

            selected = st.selectbox("Select event", labels, index=0, key="ufc_event_select")
            sel_event = events[labels.index(selected)]
            st.caption(f"Event URL: {sel_event.get('url')}")

            if st.button("Build UFC Picks (Deep Metrics)", key="ufc_build_btn"):
                with st.spinner("Loading UFCStats event & building picks..."):
                    out = ufc_build_picks_for_event(sel_event.get("url", ""))
                if not out.get("ok"):
                    st.warning(out.get("error", "Could not build UFC picks for this event. Trying Odds API fallback…"))
                else:
                    df = out["df"]
                    st.success(f"Built {len(df)} picks ✅")
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    return

    # ---- Odds API fallback ----
    st.info("UFCStats is blocked/down from this host. Showing DK/FD moneyline consensus picks (fallback).")

    fb = mma_fetch_h2h_odds()
    if debug:
        st.json({"mma_fallback": {"ok": fb.get("ok"), "status": fb.get("status"), "url": fb.get("url"), "params": fb.get("params")}})

    if not fb.get("ok"):
        st.warning(fb.get("error", "Could not load MMA odds fallback."))
        if debug:
            st.json(fb)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = mma_normalize_h2h(fb.get("payload"))
    if df.empty:
        st.warning("No MMA fights returned from The Odds API right now (DK/FD may not have posted lines yet).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    picks = mma_build_consensus_picks(df)
    if picks.empty:
        st.warning("Could not compute consensus picks from the returned odds payload.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader("✅ DK/FD Consensus Picks (No-Vig)")
    st.caption("These picks use no-vig probabilities within each book, averaged across DK/FD. Deep metrics are unavailable when UFCStats is blocked.")
    st.dataframe(
        picks[["Event", "Fighter", "ConsensusProb%", "BestPrice"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Tracker Page UI
# =========================================================
def render_tracker():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 Tracker — Pick Rate + Hit Rate")
    st.caption("Log picks from any section, then grade them (W/L/P/N/A). Summaries auto-calc by window.")

    df = _load_tracker()

    win_map = _windows(df)
    tables = []
    for label, sub in win_map.items():
        s = _summary_pick_rate(sub, label)
        if not s.empty:
            tables.append(s)

    if tables:
        summary = pd.concat(tables, ignore_index=True)
        st.markdown("### Summary (Today / Week / Month / Year)")
        st.dataframe(
            summary[["Window", "Mode", "Picks", "Graded", "Wins", "Losses", "Pushes", "HitRate%"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No tracked picks yet. Log picks from Game Lines / Props / PGA.")

    st.markdown("### Grade Picks")
    st.caption("Set Status=Graded and Result=W/L/P/N/A. (Leaving Pending means not counted in hit rate.)")

    if df.empty:
        st.info("Tracker is empty.")
    else:
        df["Status"] = df["Status"].fillna("Pending")
        df["Result"] = df["Result"].fillna("")
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", key="tracker_editor")
        if st.button("Save Tracker"):
            _save_tracker(edited)
            st.success("Saved.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================
if mode == "Tracker":
    render_tracker()
    st.stop()

if mode == "UFC":
    render_ufc()
    st.stop()

if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_bettype")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="gl_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_top25")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No game lines available."))
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per game per market (even across different lines). Ranked by Edge.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    if st.button("Log these Top Picks to Tracker"):
        rows = []
        for _, r in top.iterrows():
            rows.append({
                "LoggedAt": datetime.now().isoformat(),
                "Mode": "Game Lines",
                "Sport": sport,
                "Market": bet_type,
                "Event": r.get("Event", ""),
                "Selection": r.get("Outcome", ""),
                "Line": r.get("LineBucket", ""),
                "BestBook": r.get("BestBook", ""),
                "BestPrice": r.get("BestPrice", ""),
                "YourProb": r.get("YourProb", np.nan),
                "Implied": r.get("ImpliedBest", np.nan),
                "Edge": r.get("Edge", np.nan),
                "EV": r.get("EV", np.nan),
                "Status": "Pending",
                "Result": "",
            })
        tracker_log_rows(rows)
        st.success("Logged to Tracker ✅ (go to Tracker tab to grade/results).")

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_prop")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="pp_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_top25")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 14, 8, key="pp_scan")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No props returned for DK/FD on scanned events."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) — STRICT no-contradictions")
    st.caption("Strict rule: only ONE pick per player per market per game (no Over+Under, no different lines). Ranked by Edge.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    if st.button("Log these Top Picks to Tracker", key="log_props"):
        rows = []
        for _, r in top.iterrows():
            sel = f"{r.get('Player', '')} {r.get('Side', '')}".strip()
            rows.append({
                "LoggedAt": datetime.now().isoformat(),
                "Mode": "Player Props",
                "Sport": sport,
                "Market": prop_label,
                "Event": r.get("Event", ""),
                "Selection": sel,
                "Line": r.get("LineBucket", ""),
                "BestBook": r.get("BestBook", ""),
                "BestPrice": r.get("BestPrice", ""),
                "YourProb": r.get("YourProb", np.nan),
                "Implied": r.get("ImpliedBest", np.nan),
                "Edge": r.get("Edge", np.nan),
                "EV": r.get("EV", np.nan),
                "Status": "Pending",
                "Result": "",
            })
        tracker_log_rows(rows)
        st.success("Logged to Tracker ✅ (go to Tracker tab to grade/results).")

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (sorted by Edge)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if not DATAGOLF_API_KEY.strip():
        st.warning('Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY). PGA is hidden until then.')
        st.stop()

    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")
    st.caption("Top picks for Win / Top-10 + One-and-Done using DataGolf model probabilities + SG splits + fit/history/form proxies.")

    out, err = build_pga_board()
    if isinstance(out, dict) and "winners" in out:
        if debug:
            st.json({"dg_meta": out.get("meta", {})})

        winners = out["winners"]
        top10s = out["top10s"]
        oad = out["oad"]

        st.markdown("### 🏆 Best Win Picks (Top 10)")
        show_cols = [c for c in ["Player", "Win%", "Top10%", "SG_T2G", "SG_Putt", "BogeyAvoid", "CourseFit", "CourseHistory", "RecentForm", "SkillRating", "WinScore"] if c in winners.columns]
        st.dataframe(winners[show_cols], use_container_width=True, hide_index=True)

        st.markdown("### 🎯 Best Top-10 Picks (Top 10)")
        show_cols2 = [c for c in ["Player", "Top10%", "Win%", "SG_T2G", "SG_Putt", "BogeyAvoid", "CourseFit", "CourseHistory", "RecentForm", "SkillRating", "Top10Score"] if c in top10s.columns]
        st.dataframe(top10s[show_cols2], use_container_width=True, hide_index=True)

        st.markdown("### 🧳 Best One-and-Done Options (Top 7)")
        show_cols3 = [c for c in ["Player", "Top10%", "Win%", "SG_T2G", "SG_Putt", "BogeyAvoid", "CourseFit", "CourseHistory", "RecentForm", "SkillRating", "OADScore"] if c in oad.columns]
        st.dataframe(oad[show_cols3], use_container_width=True, hide_index=True)

        if st.button("Log PGA Top Picks to Tracker"):
            rows = []
            for _, r in winners.head(10).iterrows():
                rows.append({
                    "LoggedAt": datetime.now().isoformat(),
                    "Mode": "PGA",
                    "Sport": "PGA",
                    "Market": "Win",
                    "Event": out.get("meta", {}).get("event_name", ""),
                    "Selection": r.get("Player", ""),
                    "Line": "",
                    "BestBook": "",
                    "BestPrice": "",
                    "YourProb": pd.to_numeric(r.get("WinProb", np.nan), errors="coerce"),
                    "Implied": np.nan,
                    "Edge": np.nan,
                    "EV": np.nan,
                    "Status": "Pending",
                    "Result": "",
                })
            for _, r in top10s.head(10).iterrows():
                rows.append({
                    "LoggedAt": datetime.now().isoformat(),
                    "Mode": "PGA",
                    "Sport": "PGA",
                    "Market": "Top 10",
                    "Event": out.get("meta", {}).get("event_name", ""),
                    "Selection": r.get("Player", ""),
                    "Line": "",
                    "BestBook": "",
                    "BestPrice": "",
                    "YourProb": pd.to_numeric(r.get("Top10Prob", np.nan), errors="coerce"),
                    "Implied": np.nan,
                    "Edge": np.nan,
                    "EV": np.nan,
                    "Status": "Pending",
                    "Result": "",
                })
            tracker_log_rows(rows)
            st.success("Logged PGA picks to Tracker ✅")

    else:
        st.warning(err.get("error", "No PGA data available right now."))
        if debug:
            st.json(err)

    st.markdown("</div>", unsafe_allow_html=True)


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
