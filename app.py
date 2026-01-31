# app.py — EdgeLedger (Game Lines / Props / PGA / Tracker) + UFC (ESPN) module
# ✅ UFC fixes: uses ESPN Scoreboard (upcoming), tries BOTH competition_id + event_id for summary,
#    plus scoreboard-provided href fallbacks. No lxml, no csv, no impact to other modules.

import os
import time
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

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

def tracker_log_rows(rows: List[Dict[str, Any]]):
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

def safe_get(url: str, params: Dict[str, Any], timeout: int = 25):
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

# ✅ Mode includes UFC
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

def compute_no_vig_within_book_two_way(df: pd.DataFrame, group_cols_book: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    sums = out.groupby(group_cols_book)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out

def estimate_your_prob(df: pd.DataFrame, key_cols: List[str], book_cols: List[str]) -> pd.DataFrame:
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

def best_price_and_edge(df: pd.DataFrame, group_cols_best: List[str]) -> pd.DataFrame:
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

def strict_no_contradictions(df_best: pd.DataFrame, contradiction_cols: List[str]) -> pd.DataFrame:
    """
    STRICT contradiction removal:
    Keep exactly ONE row per contradiction group (max Edge), ignoring line differences.
    This eliminates contradictions within DK, within FD, and across DK+FD simultaneously.
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
    decomp_params = {
        "tour": "pga",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    skill_params = {
        "display": "value",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }

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
# UFC — ESPN Module (FIXED)
# =========================================================
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/summary"
ESPN_ATHLETE = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/athletes/{athlete_id}"

def _safe_float(x):
    try:
        if x is None or x == "":
            return np.nan
        return float(str(x).replace("%","").strip())
    except Exception:
        return np.nan

def _parse_inches(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    txt = str(s).strip()
    if txt.isdigit():
        return float(txt)
    try:
        if "'" in txt:
            parts = txt.replace('"','').split("'")
            ft = int(parts[0].strip())
            inch = int(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 0
            return float(ft * 12 + inch)
    except Exception:
        pass
    return np.nan

def _calc_age(dob_iso):
    try:
        if not dob_iso:
            return np.nan
        dob = pd.to_datetime(dob_iso, errors="coerce")
        if pd.isna(dob):
            return np.nan
        today = pd.Timestamp.now(tz=None).normalize()
        yrs = (today - dob.normalize()).days / 365.25
        return float(yrs)
    except Exception:
        return np.nan

@st.cache_data(ttl=60 * 30)
def fetch_ufc_events_next_days(lookahead_days: int = 21):
    """
    Find the next UFC card within lookahead.
    Captures scoreboard event_id + competition_id and any href links for fallback.
    """
    events = []
    checked = []

    base = pd.Timestamp.now().date()

    for i in range(int(lookahead_days) + 1):
        d = base + timedelta(days=i)
        datestr = d.strftime("%Y%m%d")

        ok, status, payload, url = safe_get(ESPN_SCOREBOARD, params={"dates": datestr}, timeout=10)
        checked.append({"dates": datestr, "ok": ok, "status": status})

        if not ok or not isinstance(payload, dict):
            continue

        evs = payload.get("events", []) or []
        for e in evs:
            if not isinstance(e, dict):
                continue

            event_id = e.get("id")
            name = e.get("name") or e.get("shortName") or "UFC Event"
            dt = e.get("date") or ""

            comp_id = ""
            comps = e.get("competitions")
            if isinstance(comps, list) and comps:
                if isinstance(comps[0], dict) and comps[0].get("id"):
                    comp_id = str(comps[0].get("id"))

            hrefs = []
            links = e.get("links")
            if isinstance(links, list):
                for lk in links:
                    if isinstance(lk, dict):
                        href = lk.get("href")
                        if href:
                            hrefs.append(str(href))

            if event_id:
                events.append({
                    "event_id": str(event_id),
                    "competition_id": comp_id,
                    "name": str(name),
                    "date": str(dt),
                    "hrefs": hrefs,
                })

        if events:
            break

        time.sleep(0.05)

    return {"events": events, "checked": checked}

@st.cache_data(ttl=60 * 30)
def fetch_ufc_event_summary(event_id: str, competition_id: str = "", hrefs: Optional[List[str]] = None):
    """
    ESPN UFC summary endpoint may expect competition_id rather than scoreboard event_id.
    We attempt:
      1) summary?event=<competition_id>
      2) summary?event=<event_id>
      3) any scoreboard href that looks like an ESPN API endpoint
    """
    attempts = []

    def _try(u: str, params: Optional[Dict[str, Any]] = None):
        ok, status, payload, final_url = safe_get(u, params=params or {}, timeout=12)
        attempts.append({"ok": ok, "status": status, "url": final_url})
        return ok, status, payload, final_url

    if competition_id:
        ok, status, payload, final_url = _try(ESPN_SUMMARY, params={"event": competition_id})
        if ok and isinstance(payload, dict):
            return {"ok": True, "status": status, "payload": payload, "url": final_url, "attempts": attempts}

    ok, status, payload, final_url = _try(ESPN_SUMMARY, params={"event": event_id})
    if ok and isinstance(payload, dict):
        return {"ok": True, "status": status, "payload": payload, "url": final_url, "attempts": attempts}

    hrefs = hrefs or []
    for h in hrefs:
        if not isinstance(h, str):
            continue
        if ("site.api.espn.com" in h) or ("site.web.api.espn.com" in h):
            ok2, status2, payload2, final_url2 = _try(h, params={})
            if ok2 and isinstance(payload2, dict):
                return {"ok": True, "status": status2, "payload": payload2, "url": final_url2, "attempts": attempts}

    return {"ok": False, "status": status, "payload": payload, "url": final_url, "attempts": attempts}

@st.cache_data(ttl=60 * 60 * 12)
def fetch_ufc_athlete(athlete_id: str):
    url = ESPN_ATHLETE.format(athlete_id=athlete_id)
    ok, status, payload, final_url = safe_get(url, params={}, timeout=12)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def _extract_record(rec_str: str):
    if not isinstance(rec_str, str):
        return (np.nan, np.nan, np.nan, np.nan)
    parts = [p for p in rec_str.replace("(", "").replace(")", "").replace(" ", "").split("-") if p != ""]
    try:
        w = int(parts[0]) if len(parts) > 0 else np.nan
        l = int(parts[1]) if len(parts) > 1 else np.nan
        d = int(parts[2]) if len(parts) > 2 else 0
        if np.isnan(w) or np.isnan(l):
            return (np.nan, np.nan, np.nan, np.nan)
        tot = w + l + (d if not np.isnan(d) else 0)
        winp = (w / tot) if tot > 0 else np.nan
        return (w, l, d, winp)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

def _get_stat(payload: dict, keys: List[str]):
    if not isinstance(payload, dict):
        return np.nan

    for k in keys:
        if k in payload:
            return payload.get(k)

    bio = payload.get("bio", {})
    if isinstance(bio, dict):
        for k in keys:
            if k in bio:
                return bio.get(k)

    stats = payload.get("statistics")
    if isinstance(stats, list):
        for blk in stats:
            if not isinstance(blk, dict):
                continue
            for item in (blk.get("statistics") or []):
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name","")).lower()
                abbr = str(item.get("abbreviation","")).lower()
                val = item.get("value")
                for k in keys:
                    kk = str(k).lower()
                    if kk in name or kk == abbr:
                        return val

    return np.nan

def _athlete_features(ath_payload: dict):
    name = ath_payload.get("displayName") or ath_payload.get("fullName") or ath_payload.get("name") or ""
    dob = _get_stat(ath_payload, ["dateOfBirth"])
    age = _calc_age(dob)

    reach_raw = _get_stat(ath_payload, ["reach", "reachInches"])
    height_raw = _get_stat(ath_payload, ["height", "heightInches"])
    reach_in = _parse_inches(reach_raw)
    height_in = _parse_inches(height_raw)

    stance = _get_stat(ath_payload, ["stance"])
    stance = str(stance) if stance is not None and stance is not np.nan else ""

    record_str = _get_stat(ath_payload, ["record"])
    if isinstance(record_str, dict):
        record_str = record_str.get("summary") or record_str.get("displayValue") or ""
    record_str = str(record_str) if record_str is not None and record_str is not np.nan else ""
    w, l, d, winp = _extract_record(record_str)

    rank = _safe_float(_get_stat(ath_payload, ["rank", "ranking"]))

    slpm = _safe_float(_get_stat(ath_payload, ["sigStrLandedPerMin", "slpm"]))
    sapm = _safe_float(_get_stat(ath_payload, ["sigStrAbsorbedPerMin", "sapm"]))
    str_def = _safe_float(_get_stat(ath_payload, ["sigStrDefense", "str_def"]))
    str_acc = _safe_float(_get_stat(ath_payload, ["sigStrAccuracy", "str_acc"]))

    td_avg = _safe_float(_get_stat(ath_payload, ["takedownsAvg", "takedownAvg", "td_avg"]))
    td_acc = _safe_float(_get_stat(ath_payload, ["takedownAccuracy", "takedownPct", "td_acc"]))
    td_def = _safe_float(_get_stat(ath_payload, ["takedownDefense", "td_def"]))

    itd = _safe_float(_get_stat(ath_payload, ["insideTheDistance", "itd", "finishRate"]))
    dec = _safe_float(_get_stat(ath_payload, ["decisionRate", "decisionPct", "dec"]))

    return {
        "Name": name,
        "Age": age,
        "ReachIn": reach_in,
        "HeightIn": height_in,
        "Stance": stance,
        "W": w, "L": l, "D": d, "WinPct": winp,
        "Rank": rank,
        "SLpM": slpm,
        "SApM": sapm,
        "StrDef": str_def,
        "StrAcc": str_acc,
        "TDAvg": td_avg,
        "TDAcc": td_acc,
        "TDDef": td_def,
        "ITD": itd,
        "DecRate": dec,
        "RecordStr": record_str,
    }

def _pick_model_row(a: dict, b: dict):
    d = {}
    for k in ["Age", "ReachIn", "HeightIn", "WinPct", "Rank", "SLpM", "SApM", "StrDef", "StrAcc",
              "TDAvg", "TDAcc", "TDDef", "ITD", "DecRate"]:
        d[k] = _safe_float(a.get(k)) - _safe_float(b.get(k))

    stance_a = (a.get("Stance") or "").lower()
    stance_b = (b.get("Stance") or "").lower()
    stance_bonus = 0.0
    if stance_a and stance_b and stance_a != stance_b:
        stance_bonus = 0.15
    if ("south" in stance_a and "orth" in stance_b) or ("orth" in stance_a and "south" in stance_b):
        stance_bonus = 0.25

    rank_adv = np.nan
    if not np.isnan(_safe_float(a.get("Rank"))) and not np.isnan(_safe_float(b.get("Rank"))):
        rank_adv = (_safe_float(b.get("Rank")) - _safe_float(a.get("Rank")))

    score = 0.0
    weight_sum = 0.0

    def add(term, w):
        nonlocal score, weight_sum
        if term is None or np.isnan(term):
            return
        score += w * float(term)
        weight_sum += abs(w)

    add(-d["Age"], 0.35)
    add(d["ReachIn"] / 4.0, 0.30)
    add(d["HeightIn"] / 6.0, 0.08)
    add(d["WinPct"], 0.90)
    add(rank_adv / 10.0 if rank_adv is not np.nan else np.nan, 0.55)

    add(d["SLpM"] / 2.0, 0.45)
    add(-d["SApM"] / 2.0, 0.35)
    add(d["StrDef"] / 20.0, 0.22)
    add(d["StrAcc"] / 20.0, 0.10)

    add(d["TDAvg"] / 2.0, 0.25)
    add(d["TDAcc"] / 20.0, 0.12)
    add(d["TDDef"] / 20.0, 0.25)

    add(d["ITD"] / 25.0, 0.18)
    add(d["DecRate"] / 25.0, 0.08)

    add(stance_bonus, 0.15)

    if weight_sum == 0:
        wp_a = _safe_float(a.get("WinPct"))
        wp_b = _safe_float(b.get("WinPct"))
        if not np.isnan(wp_a) and not np.isnan(wp_b) and (wp_a + wp_b) > 0:
            p = wp_a / (wp_a + wp_b)
        else:
            p = 0.50
        return float(np.clip(p, 0.05, 0.95)), np.nan, 0.0

    raw = score / max(0.6, (weight_sum / 3.2))
    p = 1.0 / (1.0 + math.exp(-raw))
    p = float(np.clip(p, 0.05, 0.95))

    itd_a = _safe_float(a.get("ITD"))
    itd_b = _safe_float(b.get("ITD"))
    fin = np.nan
    if not np.isnan(itd_a) and not np.isnan(itd_b):
        base_fin = (itd_a + itd_b) / 200.0
        fin = base_fin + (np.clip((itd_a - itd_b) / 100.0, -0.15, 0.15) * (p - 0.5) * 2.0)
        fin = float(np.clip(fin, 0.10, 0.85))

    return p, fin, float(raw)

def _extract_fights_recursive(obj, fights_out):
    if isinstance(obj, dict):
        if "competitors" in obj and isinstance(obj["competitors"], list) and len(obj["competitors"]) >= 2:
            comps = obj["competitors"]
            a = comps[0] if isinstance(comps[0], dict) else None
            b = comps[1] if isinstance(comps[1], dict) else None
            a_ath = a.get("athlete") if isinstance(a, dict) else None
            b_ath = b.get("athlete") if isinstance(b, dict) else None
            if isinstance(a_ath, dict) and isinstance(b_ath, dict):
                a_id = a_ath.get("id") or a_ath.get("uid")
                b_id = b_ath.get("id") or b_ath.get("uid")
                a_name = a_ath.get("displayName") or a_ath.get("fullName") or ""
                b_name = b_ath.get("displayName") or b_ath.get("fullName") or ""
                if a_id and b_id:
                    fights_out.append({
                        "fight_name": obj.get("name") or "",
                        "weight_class": (obj.get("type", {}) or {}).get("text") if isinstance(obj.get("type"), dict) else "",
                        "athlete_a_id": str(a_id),
                        "athlete_a_name": str(a_name),
                        "athlete_b_id": str(b_id),
                        "athlete_b_name": str(b_name),
                    })
        for v in obj.values():
            _extract_fights_recursive(v, fights_out)
    elif isinstance(obj, list):
        for it in obj:
            _extract_fights_recursive(it, fights_out)

def _parse_fights_from_summary(payload: dict):
    fights = []
    if not isinstance(payload, dict):
        return fights
    for key in ["competitions", "events", "event", "header"]:
        if key in payload:
            _extract_fights_recursive(payload[key], fights)
    if not fights:
        _extract_fights_recursive(payload, fights)

    seen = set()
    uniq = []
    for f in fights:
        a = f.get("athlete_a_id")
        b = f.get("athlete_b_id")
        if not a or not b:
            continue
        k = tuple(sorted([str(a), str(b)]))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    return uniq

def build_ufc_picks(event_id: str, competition_id: str = "", hrefs: Optional[List[str]] = None):
    summ = fetch_ufc_event_summary(event_id, competition_id=competition_id, hrefs=hrefs)

    if not summ["ok"] or not isinstance(summ["payload"], dict):
        return pd.DataFrame(), pd.DataFrame(), {
            "error": "Could not load UFC event summary from ESPN.",
            "status": summ.get("status"),
            "url": summ.get("url"),
            "attempts": summ.get("attempts", []),
            "event_id": event_id,
            "competition_id": competition_id,
        }

    fights = _parse_fights_from_summary(summ["payload"])
    if not fights:
        return pd.DataFrame(), pd.DataFrame(), {
            "error": "No fights parsed from ESPN event summary (payload structure may have changed).",
            "status": summ.get("status"),
            "url": summ.get("url"),
            "attempts": summ.get("attempts", []),
            "payload_keys": list(summ["payload"].keys())[:60],
        }

    fighter_rows = []
    for f in fights:
        a_id = f.get("athlete_a_id", "")
        b_id = f.get("athlete_b_id", "")
        if not a_id or not b_id:
            continue

        ra = fetch_ufc_athlete(a_id)
        rb = fetch_ufc_athlete(b_id)

        a_payload = ra["payload"] if ra["ok"] and isinstance(ra["payload"], dict) else {}
        b_payload = rb["payload"] if rb["ok"] and isinstance(rb["payload"], dict) else {}

        a_feat = _athlete_features(a_payload)
        b_feat = _athlete_features(b_payload)

        if not a_feat.get("Name"):
            a_feat["Name"] = f.get("athlete_a_name", "")
        if not b_feat.get("Name"):
            b_feat["Name"] = f.get("athlete_b_name", "")

        a_feat["AthleteId"] = a_id
        b_feat["AthleteId"] = b_id
        a_feat["Side"] = "A"
        b_feat["Side"] = "B"

        a_feat["Opponent"] = b_feat["Name"]
        b_feat["Opponent"] = a_feat["Name"]

        a_feat["Fight"] = f"{a_feat['Name']} vs {b_feat['Name']}"
        b_feat["Fight"] = f"{a_feat['Name']} vs {b_feat['Name']}"

        a_feat["WeightClass"] = f.get("weight_class", "")
        b_feat["WeightClass"] = f.get("weight_class", "")

        fighter_rows.append(a_feat)
        fighter_rows.append(b_feat)

        time.sleep(0.02)

    fighters_df = pd.DataFrame(fighter_rows)
    if fighters_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "error": "Could not load fighter data from ESPN athlete endpoints.",
            "summary_url": summ.get("url"),
            "attempts": summ.get("attempts", []),
        }

    picks = []
    for fight_name, grp in fighters_df.groupby("Fight", dropna=False):
        if len(grp) < 2:
            continue
        a = grp.iloc[0].to_dict()
        b = grp.iloc[1].to_dict()
        if a.get("Side") != "A":
            a, b = b, a

        p_a, fin_proxy, raw = _pick_model_row(a, b)
        pick = a.get("Name") if p_a >= 0.5 else b.get("Name")
        prob = p_a if p_a >= 0.5 else (1.0 - p_a)

        delta_age = _safe_float(a.get("Age")) - _safe_float(b.get("Age"))
        delta_reach = _safe_float(a.get("ReachIn")) - _safe_float(b.get("ReachIn"))
        delta_win = _safe_float(a.get("WinPct")) - _safe_float(b.get("WinPct"))

        rank_adv = np.nan
        if not np.isnan(_safe_float(a.get("Rank"))) and not np.isnan(_safe_float(b.get("Rank"))):
            rank_adv = (_safe_float(b.get("Rank")) - _safe_float(a.get("Rank")))

        picks.append({
            "Fight": fight_name,
            "WeightClass": a.get("WeightClass", ""),
            "Pick": pick,
            "WinProb%": round(prob * 100.0, 1),
            "ModelRaw": round(raw, 3),
            "FinishProxy%": round(fin_proxy * 100.0, 1) if fin_proxy is not None and not np.isnan(fin_proxy) else np.nan,
            "ΔAge(A-B)": round(delta_age, 2) if not np.isnan(delta_age) else np.nan,
            "ΔReach(A-B)": round(delta_reach, 1) if not np.isnan(delta_reach) else np.nan,
            "ΔWinPct(A-B)": round(delta_win, 3) if not np.isnan(delta_win) else np.nan,
            "RankAdv(A better +)": round(rank_adv, 2) if rank_adv is not np.nan else np.nan,
        })

    fights_df = pd.DataFrame(picks).sort_values(["WinProb%"], ascending=False) if picks else pd.DataFrame()
    diag = {
        "summary_url": summ.get("url"),
        "attempts": summ.get("attempts", []),
        "n_fights": int(len(fights_df)),
        "parsed_pairs": int(len(fights)),
        "event_id": event_id,
        "competition_id": competition_id,
    }
    return fights_df, fighters_df, diag

def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC — Picks (ESPN data)")
    st.caption(
        "Upcoming card from ESPN scoreboard + fight list from ESPN summary. "
        "Model uses age/reach/stance/record + striking + takedowns + defense + ITD/decision when available."
    )

    with st.expander("Controls", expanded=True):
        lookahead = st.slider("Look ahead days (find next card)", 1, 45, 21, step=1)
        topn = st.slider("Show top N strongest picks (by model win prob)", 3, 15, 10, step=1)

    ev = fetch_ufc_events_next_days(lookahead_days=int(lookahead))
    events = ev.get("events", [])

    if debug:
        st.json({"scoreboard_checks": ev.get("checked", [])})

    if not events:
        st.warning("No UFC events returned by ESPN scoreboard in the lookahead window. Try increasing lookahead.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    def _label(e):
        dt = e.get("date", "")
        try:
            dtt = pd.to_datetime(dt, errors="coerce")
            dt_str = dtt.strftime("%a %b %d, %Y %I:%M %p") if not pd.isna(dtt) else dt
        except Exception:
            dt_str = dt
        comp = e.get("competition_id","")
        tag = f" (comp:{comp})" if comp else ""
        return f"{e.get('name','UFC Event')} — {dt_str}{tag}"

    labels = [_label(e) for e in events]
    idx = st.selectbox("Select Event", options=list(range(len(events))), format_func=lambda i: labels[i], index=0)
    picked = events[int(idx)]

    event_id = picked.get("event_id", "")
    comp_id = picked.get("competition_id", "")
    hrefs = picked.get("hrefs", []) or []

    if not event_id:
        st.error("Selected event has no event_id from ESPN.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with st.spinner("Loading fight card + fighters from ESPN..."):
        fights_df, fighters_df, diag = build_ufc_picks(event_id, competition_id=comp_id, hrefs=hrefs)

    if fights_df.empty:
        st.warning("Could not build UFC picks for this event.")
        st.info("Diagnostics below show which ESPN endpoints were attempted.")
        st.json(diag)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Picks (sorted by model win probability)")
    show = fights_df.head(int(topn)).copy()
    st.dataframe(
        show[["Fight","WeightClass","Pick","WinProb%","FinishProxy%","ΔAge(A-B)","ΔReach(A-B)","RankAdv(A better +)","ΔWinPct(A-B)"]],
        use_container_width=True,
        hide_index=True
    )

    with st.expander("Fighter data used", expanded=False):
        cols = [c for c in fighters_df.columns if c not in ["Side"]]
        st.dataframe(fighters_df[cols], use_container_width=True, hide_index=True)

    if debug:
        st.json(diag)

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
