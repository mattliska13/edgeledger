# app.py — EdgeLedger (Game Lines / Player Props / PGA / Tracker) + UFC Picks (ESPN-first)
# NOTE: UFC module is added in a “no-risk” way: new functions + new sidebar mode + new main branch.
# It does NOT modify existing odds/PGA/tracker logic.

import os
import time
import json
import math
import re
from datetime import datetime, timezone

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
        return pd.DataFrame(columns=["Window","Mode","Picks","Graded","Wins","Losses","Pushes","HitRate%"])
    x = df.copy()
    x["Picks"] = 1
    x["Graded"] = (x["Status"] == "Graded").astype(int)
    x["Wins"] = (x["Result"] == "W").astype(int)
    x["Losses"] = (x["Result"] == "L").astype(int)
    x["Pushes"] = (x["Result"] == "P").astype(int)

    agg = x.groupby(["Mode"], dropna=False).agg(
        Picks=("Picks", "sum"),
        Graded=("Graded","sum"),
        Wins=("Wins","sum"),
        Losses=("Losses","sum"),
        Pushes=("Pushes","sum"),
    ).reset_index()

    denom = (agg["Wins"] + agg["Losses"] + agg["Pushes"]).replace(0, np.nan)
    agg["HitRate%"] = ((agg["Wins"] / denom) * 100.0).round(1).fillna(0.0)

    agg.insert(0, "Window", label)
    return agg.sort_values(["Picks","HitRate%"], ascending=False)

def tracker_log_rows(rows: list[dict]):
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
# HTTP (shared)
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/1.0 (streamlit)"})

def safe_get(url: str, params: dict | None = None, timeout: int = 25):
    """Safe GET returning (ok, status, payload_json_or_text, final_url)."""
    try:
        r = SESSION.get(url, params=params or None, timeout=timeout)
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
BOOKMAKERS = "draftkings,fanduel"   # ONLY these

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

# ✅ Add UFC mode (new) without touching others
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
    "DK/FD only. Game lines / props / PGA run independently."
)

if not ODDS_API_KEY.strip() and mode not in ["Tracker", "UFC"]:
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
    model_choice = None
    for p in prefer:
        if p in payload and isinstance(payload[p], list):
            model_choice = p
            break

    if model_choice and isinstance(payload.get(model_choice), list):
        rows = payload[model_choice]
        if len(rows) == 0 or isinstance(rows[0], dict):
            meta["model_used"] = model_choice
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
# UFC — ESPN-first Picks Module (NO CSV, no lxml, no UFCStats dependency)
# =========================================================
ESPN_UFC_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
ESPN_CORE_ATHLETE = "https://sports.core.api.espn.com/v2/sports/mma/leagues/ufc/athletes/{athlete_id}"

def _utc_now():
    return datetime.now(timezone.utc)

def _parse_iso(dt_str: str):
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None

def _num(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        if s == "" or s.lower() in ["nan", "none", "-"]:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except Exception:
        return 0.5

def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

@st.cache_data(ttl=60 * 5)
def ufc_fetch_scoreboard():
    ok, status, payload, url = safe_get(ESPN_UFC_SCOREBOARD, params=None, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": url}

@st.cache_data(ttl=60 * 60 * 24)
def ufc_fetch_athlete(athlete_id: str):
    url = ESPN_CORE_ATHLETE.format(athlete_id=str(athlete_id))
    ok, status, payload, final_url = safe_get(url, params=None, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def _extract_record_summary(ath_payload: dict):
    # ESPN core athlete often includes: "records": [{"summary":"16-3-0"}], sometimes nested.
    rec = {"summary": "", "wins": np.nan, "losses": np.nan, "draws": np.nan}
    try:
        records = ath_payload.get("records", []) or []
        for r in records:
            if isinstance(r, dict) and r.get("type") in ["total", "overall", "all"] or r.get("name") in ["overall", "total"]:
                summ = r.get("summary") or ""
                if summ:
                    rec["summary"] = summ
                    break
        if not rec["summary"] and records and isinstance(records[0], dict):
            rec["summary"] = records[0].get("summary") or ""
        if rec["summary"]:
            m = re.match(r"^\s*(\d+)\s*-\s*(\d+)(?:\s*-\s*(\d+))?\s*$", rec["summary"])
            if m:
                rec["wins"] = float(m.group(1))
                rec["losses"] = float(m.group(2))
                rec["draws"] = float(m.group(3)) if m.group(3) else 0.0
    except Exception:
        pass
    return rec

def _extract_bio_stats(ath_payload: dict):
    # Try multiple common keys without assuming structure is stable.
    # We keep this defensive; missing fields simply become NaN.
    out = {
        "age": np.nan,
        "reach_in": np.nan,
        "height_in": np.nan,
        "stance": "",
        "rank": np.nan,
        "wins": np.nan,
        "losses": np.nan,
        "win_pct": np.nan,
        "inside_dist_rate": np.nan,  # proxy if breakdown exists
        "decision_rate": np.nan,     # proxy if breakdown exists
        "slpm": np.nan, "str_acc": np.nan, "sapm": np.nan, "str_def": np.nan,
        "td_avg": np.nan, "td_acc": np.nan, "td_def": np.nan, "sub_avg": np.nan,
        "last5_win_rate": np.nan,
    }

    if not isinstance(ath_payload, dict):
        return out

    # Age
    try:
        dob = ath_payload.get("dateOfBirth") or ath_payload.get("birthDate")
        if dob:
            d = _parse_iso(str(dob))
            if d:
                now = _utc_now()
                out["age"] = (now.date() - d.date()).days / 365.25
    except Exception:
        pass

    # Stance (sometimes "stance" or "style")
    for k in ["stance", "style", "handedness"]:
        v = ath_payload.get(k)
        if isinstance(v, str) and v.strip():
            out["stance"] = v.strip()
            break

    # Height / Reach (often embedded in "displayHeight"/"displayWeight" strings or "height"/"reach")
    # Try numeric keys first
    for k in ["reach", "reachInches", "reach_in", "reachIn"]:
        v = ath_payload.get(k)
        if v is not None:
            out["reach_in"] = _num(v)
            break

    # Some ESPN payloads store measurements in "statistics" or "measurements"
    try:
        meas = ath_payload.get("measurements") or ath_payload.get("measurement") or {}
        if isinstance(meas, dict):
            if not np.isfinite(out["reach_in"]) and meas.get("reach"):
                out["reach_in"] = _num(meas.get("reach"))
            if meas.get("height"):
                out["height_in"] = _num(meas.get("height"))
    except Exception:
        pass

    # Rank (some payloads have "rank" or "ranking")
    for k in ["rank", "ranking"]:
        v = ath_payload.get(k)
        if v is not None:
            out["rank"] = _num(v)
            break

    # Record
    rec = _extract_record_summary(ath_payload)
    out["wins"], out["losses"] = rec["wins"], rec["losses"]
    if np.isfinite(out["wins"]) and np.isfinite(out["losses"]) and (out["wins"] + out["losses"]) > 0:
        out["win_pct"] = out["wins"] / (out["wins"] + out["losses"])

    # Stats blocks: often in ath_payload["statistics"] with categories / splits
    # We search for common labels defensively.
    def harvest_stats(node):
        if not isinstance(node, (dict, list)):
            return
        if isinstance(node, list):
            for it in node:
                harvest_stats(it)
            return
        # dict
        for key, val in node.items():
            # If this looks like a label/value record
            if key in ["name", "label"] and isinstance(val, str):
                nm = val.lower()
                # value might be in "value", "displayValue"
                v = node.get("value", node.get("displayValue", node.get("display", None)))
                if v is None:
                    continue
                if "sig. str" in nm and ("landed" in nm or "per min" in nm or "slpm" in nm):
                    out["slpm"] = _num(v)
                elif ("sig. str" in nm or "sig str" in nm) and ("acc" in nm or "accuracy" in nm):
                    out["str_acc"] = _num(v) / 100.0 if _num(v) > 1 else _num(v)
                elif ("sig. str" in nm or "sig str" in nm) and ("absorbed" in nm or "sapm" in nm or "against" in nm):
                    out["sapm"] = _num(v)
                elif ("sig. str" in nm or "sig str" in nm) and ("def" in nm or "defense" in nm):
                    out["str_def"] = _num(v) / 100.0 if _num(v) > 1 else _num(v)
                elif ("takedown" in nm) and ("avg" in nm or "per" in nm):
                    out["td_avg"] = _num(v)
                elif ("takedown" in nm) and ("acc" in nm or "accuracy" in nm):
                    out["td_acc"] = _num(v) / 100.0 if _num(v) > 1 else _num(v)
                elif ("takedown" in nm) and ("def" in nm or "defense" in nm):
                    out["td_def"] = _num(v) / 100.0 if _num(v) > 1 else _num(v)
                elif ("sub" in nm or "submission" in nm) and ("avg" in nm or "per" in nm):
                    out["sub_avg"] = _num(v)

            # recurse
            if isinstance(val, (dict, list)):
                harvest_stats(val)

    try:
        harvest_stats(ath_payload.get("statistics"))
        harvest_stats(ath_payload.get("stats"))
        harvest_stats(ath_payload.get("splits"))
    except Exception:
        pass

    # Finish vs decision proxies if breakdown exists (very optional)
    try:
        # look for dicts like {"winsByKnockout": X, ...}
        flat = json.dumps(ath_payload)
        # If any explicit keys exist, attempt best-effort extraction:
        def get_first_number_for_key(k):
            m = re.search(rf'"{re.escape(k)}"\s*:\s*([0-9]+)', flat)
            return float(m.group(1)) if m else np.nan

        w_ko = get_first_number_for_key("winsByKnockout")
        w_sub = get_first_number_for_key("winsBySubmission")
        w_dec = get_first_number_for_key("winsByDecision")
        w_total = out["wins"]

        if np.isfinite(w_total) and w_total > 0:
            if np.isfinite(w_ko) or np.isfinite(w_sub):
                fin = (0 if not np.isfinite(w_ko) else w_ko) + (0 if not np.isfinite(w_sub) else w_sub)
                out["inside_dist_rate"] = fin / w_total
            if np.isfinite(w_dec):
                out["decision_rate"] = w_dec / w_total
    except Exception:
        pass

    # last5_win_rate: ESPN core sometimes has recent events; keep defensive.
    try:
        # Search for "recent" fights with result flags
        wins = 0
        tot = 0
        recent = ath_payload.get("recentEvents") or ath_payload.get("recent") or []
        if isinstance(recent, list):
            for ev in recent[:5]:
                if not isinstance(ev, dict):
                    continue
                # Look for "winner": true/false etc.
                res = ev.get("winner")
                if res is True:
                    wins += 1
                    tot += 1
                elif res is False:
                    tot += 1
        if tot > 0:
            out["last5_win_rate"] = wins / tot
    except Exception:
        pass

    return out

def ufc_list_upcoming_events(scoreboard_payload: dict):
    """
    Uses ESPN UFC scoreboard JSON to list upcoming (scheduled) events.
    Returns list of dict: {id, name, date_iso, comps_count}
    """
    events = []
    if not isinstance(scoreboard_payload, dict):
        return events

    for ev in scoreboard_payload.get("events", []) or []:
        if not isinstance(ev, dict):
            continue
        ev_id = str(ev.get("id") or "").strip()
        nm = (ev.get("name") or ev.get("shortName") or "").replace("\n", " ").strip()
        dt = ev.get("date")
        comps = ev.get("competitions", []) or []
        if not ev_id or not dt:
            continue
        d = _parse_iso(str(dt))
        if not d:
            continue
        # keep future-ish events (or today)
        if d >= (_utc_now() - pd.Timedelta(hours=8)).to_pytimedelta():
            events.append({
                "id": ev_id,
                "name": nm or f"UFC Event {ev_id}",
                "date": str(dt),
                "competitions": comps,
                "comps_count": len(comps),
            })

    # sort by date
    events.sort(key=lambda x: x["date"])
    return events

def ufc_build_fight_table(event_obj: dict):
    """
    Builds a fight-level dataframe:
    - pulls competitors from event_obj["competitions"]
    - fetches athlete bios/stats from ESPN core athlete endpoint
    - computes model score & pick
    """
    comps = event_obj.get("competitions", []) or []
    if not comps:
        return pd.DataFrame(), {"error": "No competitions found on selected event payload."}

    fights = []
    diag = {"athlete_fetch": []}

    # limit fights to avoid hangs if ESPN returns something unexpected
    for c in comps[:18]:
        if not isinstance(c, dict):
            continue
        competitors = c.get("competitors", []) or []
        if len(competitors) != 2:
            continue

        def comp_ath_id(comp):
            # competitor id is athlete id
            if isinstance(comp, dict):
                cid = comp.get("id")
                if cid:
                    return str(cid)
                ath = comp.get("athlete") or {}
                if isinstance(ath, dict) and ath.get("id"):
                    return str(ath.get("id"))
            return ""

        a_id = comp_ath_id(competitors[0])
        b_id = comp_ath_id(competitors[1])
        if not a_id or not b_id:
            continue

        # Names
        def comp_name(comp):
            if not isinstance(comp, dict):
                return ""
            ath = comp.get("athlete") or {}
            if isinstance(ath, dict):
                return (ath.get("displayName") or ath.get("fullName") or ath.get("shortName") or "").strip()
            return ""

        a_name = comp_name(competitors[0])
        b_name = comp_name(competitors[1])

        # Weight class (type abbreviation often on competition["type"]["abbreviation"])
        wc = ""
        try:
            t = c.get("type") or {}
            if isinstance(t, dict):
                wc = (t.get("abbreviation") or t.get("name") or "").strip()
        except Exception:
            wc = ""

        # Fetch athlete payloads (cached)
        a_res = ufc_fetch_athlete(a_id)
        b_res = ufc_fetch_athlete(b_id)
        diag["athlete_fetch"].append({"a": a_id, "a_ok": a_res["ok"], "a_status": a_res["status"],
                                      "b": b_id, "b_ok": b_res["ok"], "b_status": b_res["status"]})

        if not a_res["ok"] or not isinstance(a_res["payload"], dict):
            continue
        if not b_res["ok"] or not isinstance(b_res["payload"], dict):
            continue

        a_stats = _extract_bio_stats(a_res["payload"])
        b_stats = _extract_bio_stats(b_res["payload"])

        fights.append({
            "WeightClass": wc,
            "A": a_name or f"Athlete {a_id}",
            "B": b_name or f"Athlete {b_id}",
            "A_id": a_id,
            "B_id": b_id,
            **{f"A_{k}": v for k, v in a_stats.items()},
            **{f"B_{k}": v for k, v in b_stats.items()},
        })

    if not fights:
        return pd.DataFrame(), {"error": "No fights parsed from selected event competitions."}

    df = pd.DataFrame(fights)

    # Model: z-score across fighters within card
    # Build feature deltas (A - B)
    def delta(col):
        return pd.to_numeric(df[f"A_{col}"], errors="coerce") - pd.to_numeric(df[f"B_{col}"], errors="coerce")

    # Features
    df["d_win_pct"] = delta("win_pct")
    df["d_last5"] = delta("last5_win_rate")
    df["d_age"] = delta("age")             # younger better => we will NEGATE age delta
    df["d_reach"] = delta("reach_in")
    df["d_slpm"] = delta("slpm")
    df["d_sapm"] = delta("sapm")           # lower is better => negate
    df["d_str_acc"] = delta("str_acc")
    df["d_str_def"] = delta("str_def")
    df["d_td_avg"] = delta("td_avg")
    df["d_td_acc"] = delta("td_acc")
    df["d_td_def"] = delta("td_def")
    df["d_sub_avg"] = delta("sub_avg")
    df["d_inside"] = delta("inside_dist_rate")
    df["d_dec"] = delta("decision_rate")

    # Rank: smaller number is better; if present, use -(A_rank - B_rank)
    df["d_rank"] = -(pd.to_numeric(df["A_rank"], errors="coerce") - pd.to_numeric(df["B_rank"], errors="coerce"))

    # Direction fixes
    df["d_age"] = -df["d_age"]     # younger advantage
    df["d_sapm"] = -df["d_sapm"]   # lower absorbed is better

    # Z-score each delta column (robust to missing)
    feat_cols = [
        "d_win_pct","d_last5","d_age","d_reach",
        "d_slpm","d_sapm","d_str_acc","d_str_def",
        "d_td_avg","d_td_acc","d_td_def","d_sub_avg",
        "d_inside","d_dec","d_rank"
    ]
    for c in feat_cols:
        df[f"z_{c}"] = _zscore(df[c])

    # Weights (tuned for general signal balance; missing fields contribute ~0 via zscore fallback)
    W = {
        "z_d_win_pct": 0.26,
        "z_d_last5": 0.08,
        "z_d_rank": 0.06,
        "z_d_age": 0.05,
        "z_d_reach": 0.05,
        "z_d_slpm": 0.10,
        "z_d_sapm": 0.08,
        "z_d_str_acc": 0.05,
        "z_d_str_def": 0.05,
        "z_d_td_avg": 0.08,
        "z_d_td_acc": 0.04,
        "z_d_td_def": 0.06,
        "z_d_sub_avg": 0.03,
        "z_d_inside": 0.02,
        "z_d_dec": 0.01,
    }

    score = np.zeros(len(df))
    for k, w in W.items():
        if k in df.columns:
            score += w * pd.to_numeric(df[k], errors="coerce").fillna(0.0).values

    df["ModelScore_A_minus_B"] = score

    # Convert to win probability style confidence
    # Scale factor keeps typical differences in a reasonable range.
    df["Conf_A"] = df["ModelScore_A_minus_B"].apply(lambda x: float(_sigmoid(2.25 * float(x))))
    df["Pick"] = np.where(df["ModelScore_A_minus_B"] >= 0, df["A"], df["B"])
    df["PickProb"] = np.where(df["ModelScore_A_minus_B"] >= 0, df["Conf_A"], 1.0 - df["Conf_A"])

    # Explain: top edges by absolute z contribution
    contrib_cols = list(W.keys())
    def top_factors(row, n=3):
        pairs = []
        for k, w in W.items():
            v = row.get(k, 0.0)
            if pd.isna(v):
                v = 0.0
            pairs.append((k.replace("z_d_", ""), float(w) * float(v)))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        out = []
        for name, val in pairs[:n]:
            out.append(f"{name}:{val:+.2f}")
        return ", ".join(out)

    df["TopFactors"] = df.apply(top_factors, axis=1)

    # Clean display fields
    def fmt_pct(x):
        if not np.isfinite(x):
            return ""
        return f"{100.0*float(x):.1f}%"

    df["PickProb%"] = df["PickProb"].apply(fmt_pct)
    return df, diag

def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC Picks — ESPN-first model (no CSV)")
    st.caption(
        "Pulls UFC card from ESPN scoreboard → competitors/athletes → builds a stats-based model. "
        "Uses age/reach/stance/record + striking + takedowns; finish/decision are best-effort proxies when available."
    )

    sb = ufc_fetch_scoreboard()
    if debug:
        st.json({"ufc_scoreboard": {"ok": sb["ok"], "status": sb["status"], "url": sb["url"]}})

    if not sb["ok"] or not isinstance(sb["payload"], dict):
        st.error("Could not load UFC scoreboard from ESPN.")
        if debug:
            st.json(sb)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    events = ufc_list_upcoming_events(sb["payload"])
    if not events:
        st.warning("No UFC events found on ESPN scoreboard right now.")
        if debug:
            st.json({"payload_keys": list(sb["payload"].keys())})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Dropdown
    labels = [f"{e['name']} — {e['date'][:10]}" for e in events]
    idx_default = 0
    selected = st.selectbox("Select event", options=list(range(len(events))), format_func=lambda i: labels[i], index=idx_default)

    ev = events[int(selected)]
    st.markdown(
        f"<span class='pill'>Event ID: {ev['id']}</span>"
        f"<span class='pill'>Fights listed: {ev['comps_count']}</span>",
        unsafe_allow_html=True
    )

    # Build picks
    with st.spinner("Building UFC model picks…"):
        df, diag = ufc_build_fight_table(ev)

    if df.empty:
        st.warning("Could not build UFC picks for this event.")
        if debug:
            st.json({"event": {"id": ev["id"], "name": ev["name"], "date": ev["date"]}, "diag": diag})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Display
    show = df.copy()
    show["Fight"] = show["A"].astype(str) + " vs " + show["B"].astype(str)
    show = show.sort_values("PickProb", ascending=False)

    cols = [
        "WeightClass", "Fight", "Pick", "PickProb%", "TopFactors",
        "A_age","B_age","A_reach_in","B_reach_in",
        "A_win_pct","B_win_pct","A_last5_win_rate","B_last5_win_rate",
        "A_slpm","B_slpm","A_sapm","B_sapm",
        "A_td_avg","B_td_avg","A_td_def","B_td_def",
    ]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)

    if st.toggle("Show raw model table (debug)", value=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    if debug:
        st.json({"ufc_diag": diag})

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
            summary[["Window","Mode","Picks","Graded","Wins","Losses","Pushes","HitRate%"]],
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
                "BestBook": r.get("BestBook",""),
                "BestPrice": r.get("BestPrice",""),
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
            sel = f"{r.get('Player','')} {r.get('Side','')}".strip()
            rows.append({
                "LoggedAt": datetime.now().isoformat(),
                "Mode": "Player Props",
                "Sport": sport,
                "Market": prop_label,
                "Event": r.get("Event", ""),
                "Selection": sel,
                "Line": r.get("LineBucket", ""),
                "BestBook": r.get("BestBook",""),
                "BestPrice": r.get("BestPrice",""),
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
