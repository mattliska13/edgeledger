import os
import time
import re
from html import unescape
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from typing import List, Dict, Any, Tuple

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
# Tracker (kept identical)
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
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "") or get_key("DATAGOLF_KEY", "")

# =========================================================
# HTTP (kept identical, but used by UFC too)
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Dashboard/1.0 (streamlit)"})

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
# Odds math (kept identical)
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
# API Config (The Odds API) (kept identical)
# =========================================================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"

SPORT_KEYS_LINES = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}
SPORT_KEYS_PROPS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}
GAME_MARKETS = {"Moneyline": "h2h", "Spreads": "spreads", "Totals": "totals"}
PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================================================
# Sidebar UI (only change: includes UFC)
# =========================================================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>EDGE = YourProb − ImpliedProb(best price)</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
show_non_value = st.sidebar.checkbox("Show non-value rows (Edge ≤ 0)", value=False)

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
# Header (kept identical)
# =========================================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Ranked by **Edge = YourProb − ImpliedProb(best price)**. "
    "**Strict contradiction removal**: only one side per game/market (and one side per player/market). "
    "DK/FD only. Game lines / props / PGA run independently."
)

# IMPORTANT: keep existing behavior; ODDS key required for non-UFC, non-Tracker
if not ODDS_API_KEY.strip() and mode not in ("Tracker", "UFC"):
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar expander.')
    st.stop()

# =========================================================
# Caching (daily) (kept identical)
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
# Normalizers (kept identical)
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
                side = out.get("description")
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
# Core Best-Bet Logic (kept identical)
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
# Boards (kept identical)
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

    df_best = strict_no_contradictions(df_best, contradiction_cols=["Event", "Market"])
    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best).sort_values("Edge", ascending=False)
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

    df_best = strict_no_contradictions(df_best, contradiction_cols=["Event", "Market", "Player"])
    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best).sort_values("Edge", ascending=False)
    return df_best, {}

# =========================================================
# Charts (kept identical)
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
# PGA — DataGolf (kept identical)
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
# UFC Module (NEW FIXES — minimal-risk; no lxml; no csv)
# - Primary source: UFCStats upcoming list (https + http fallback)
# - Secondary fallback: completed list (in case upcoming is empty/blocked)
# - Regex parsing only; no external deps
# - Adds a "Force refresh UFC cache" button to solve stale/no-event issues
# =========================================================
UFC_UPCOMING = "https://ufcstats.com/statistics/events/upcoming?page=all"
UFC_UPCOMING_HTTP = "http://ufcstats.com/statistics/events/upcoming?page=all"
UFC_COMPLETED = "https://ufcstats.com/statistics/events/completed?page=all"
UFC_COMPLETED_HTTP = "http://ufcstats.com/statistics/events/completed?page=all"

def _ufc_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

def ufc_fetch_html(url: str, timeout: int = 25) -> str:
    # Hard guard: UFC should never take down the whole app
    try:
        r = SESSION.get(url, headers=_ufc_headers(), timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text or ""
    except Exception:
        # allow caller to try other URLs
        return ""

def _parse_events(html: str) -> pd.DataFrame:
    # Event listing table: anchor to event-details + date column
    # We accept both http and https in the href
    pat = re.compile(
        r'<a[^>]+href="(?P<url>https?://ufcstats\.com/event-details/[^"]+)"[^>]*>\s*(?P<name>[^<]+)\s*</a>'
        r'.*?<td[^>]*class="b-statistics__date"[^>]*>\s*(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})\s*</td>',
        flags=re.I | re.S,
    )

    rows = []
    for m in pat.finditer(html):
        name = unescape(m.group("name")).strip()
        date_raw = unescape(m.group("date")).strip()
        dt = pd.to_datetime(date_raw, errors="coerce")
        rows.append({"Event": name, "Date": dt, "DateStr": date_raw, "EventURL": m.group("url").strip()})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.drop_duplicates(subset=["EventURL"]).reset_index(drop=True)
    return df

@st.cache_data(ttl=60 * 20)
def ufc_list_events_cached() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    notes = {"sources_tried": [], "found": 0}

    for u in [UFC_UPCOMING, UFC_UPCOMING_HTTP, UFC_COMPLETED, UFC_COMPLETED_HTTP]:
        html = ufc_fetch_html(u)
        notes["sources_tried"].append({"url": u, "bytes": len(html or "")})
        if not html:
            continue
        df = _parse_events(html)
        if not df.empty:
            notes["found"] = int(len(df))
            # Sort: upcoming page is usually future; for safety sort by date desc
            if df["Date"].notna().any():
                df = df.sort_values("Date", ascending=False)
            return df.reset_index(drop=True), notes

    return pd.DataFrame(columns=["Event", "Date", "DateStr", "EventURL"]), notes

def _parse_event_fights(event_html: str) -> List[Dict[str, str]]:
    fights = []
    row_pat = re.compile(
        r'<a[^>]+href="(?P<fight>https?://ufcstats\.com/fight-details/[^"]+)"[^>]*>.*?</a>.*?'
        r'<a[^>]+href="(?P<red>https?://ufcstats\.com/fighter-details/[^"]+)"[^>]*>\s*(?P<red_name>[^<]+)\s*</a>.*?'
        r'<a[^>]+href="(?P<blue>https?://ufcstats\.com/fighter-details/[^"]+)"[^>]*>\s*(?P<blue_name>[^<]+)\s*</a>',
        flags=re.I | re.S
    )
    for m in row_pat.finditer(event_html):
        fights.append({
            "FightURL": m.group("fight").strip(),
            "RedURL": m.group("red").strip(),
            "BlueURL": m.group("blue").strip(),
            "Red": unescape(m.group("red_name")).strip(),
            "Blue": unescape(m.group("blue_name")).strip(),
        })

    seen = set()
    uniq = []
    for f in fights:
        if f["FightURL"] in seen:
            continue
        seen.add(f["FightURL"])
        uniq.append(f)
    return uniq

@st.cache_data(ttl=60 * 20)
def ufc_get_event_fights(event_url: str) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    meta = {"event_url": event_url, "bytes": 0}
    html = ufc_fetch_html(event_url)
    meta["bytes"] = len(html or "")
    if not html:
        return [], meta
    fights = _parse_event_fights(html)
    meta["fights_found"] = len(fights)
    return fights, meta

def _extract_fighter_profile(html: str) -> Dict[str, Any]:
    def pick(pattern):
        m = re.search(pattern, html, flags=re.I | re.S)
        return unescape(m.group(1)).strip() if m else ""

    name = pick(r'<span[^>]*class="b-content__title-highlight"[^>]*>\s*([^<]+)\s*</span>')
    record = pick(r'<span[^>]*class="b-content__title-record"[^>]*>[^:]*:\s*([^<]+)</span>')

    reach = pick(r'Reach:\s*</i>\s*([^<]+)<')
    stance = pick(r'STANCE:\s*</i>\s*([^<]+)<')
    dob = pick(r'DOB:\s*</i>\s*([^<]+)<')

    stats = {k: np.nan for k in ["SLpM","StrAcc","SApM","StrDef","TDAvg","TDAcc","TDDef","SubAvg"]}

    li_pat = re.compile(r'<li[^>]*class="b-list__box-list-item[^"]*">\s*(.*?)</li>', flags=re.I | re.S)
    for li in li_pat.findall(html):
        li2 = re.sub(r"\s+", " ", li)
        km = re.search(r'<i[^>]*>\s*([A-Za-z0-9\.\%\s]+):\s*</i>\s*([^<]+)', li2, flags=re.I)
        if not km:
            continue
        label = km.group(1).strip()
        val = km.group(2).strip()
        mapping = {
            "SLpM": "SLpM",
            "Str. Acc.": "StrAcc",
            "SApM": "SApM",
            "Str. Def": "StrDef",
            "TD Avg.": "TDAvg",
            "TD Acc.": "TDAcc",
            "TD Def.": "TDDef",
            "Sub. Avg.": "SubAvg",
        }
        if label in mapping:
            stats[mapping[label]] = val

    def parse_inches(s):
        if not isinstance(s, str):
            return np.nan
        m = re.search(r'(\d+(\.\d+)?)', s)
        return float(m.group(1)) if m else np.nan

    dob_dt = pd.to_datetime(dob, errors="coerce")
    age = np.nan
    if not pd.isna(dob_dt):
        age = (pd.Timestamp.now().normalize() - dob_dt.normalize()).days / 365.25

    w = l = d = np.nan
    mrec = re.search(r'(\d+)\s*-\s*(\d+)\s*-\s*(\d+)', record)
    if mrec:
        w, l, d = int(mrec.group(1)), int(mrec.group(2)), int(mrec.group(3))

    return {
        "Name": name,
        "Record": record,
        "W": w, "L": l, "D": d,
        "ReachIn": parse_inches(reach),
        "Stance": stance,
        "DOB": dob,
        "Age": float(age) if age == age else np.nan,
        "SLpM": stats.get("SLpM", np.nan),
        "StrAcc": stats.get("StrAcc", np.nan),
        "SApM": stats.get("SApM", np.nan),
        "StrDef": stats.get("StrDef", np.nan),
        "TDAvg": stats.get("TDAvg", np.nan),
        "TDAcc": stats.get("TDAcc", np.nan),
        "TDDef": stats.get("TDDef", np.nan),
        "SubAvg": stats.get("SubAvg", np.nan),
    }

def _coerce_profile(p: Dict[str, Any]) -> Dict[str, Any]:
    def to_num(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        if not s or s in ("--", "N/A"):
            return np.nan
        if s.endswith("%"):
            try:
                return float(s.replace("%","")) / 100.0
            except Exception:
                return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan

    for k in ["SLpM","SApM","TDAvg","SubAvg","StrAcc","StrDef","TDAcc","TDDef"]:
        p[k] = to_num(p.get(k))
    p["Age"] = pd.to_numeric(p.get("Age"), errors="coerce")
    p["ReachIn"] = pd.to_numeric(p.get("ReachIn"), errors="coerce")
    p["W"] = pd.to_numeric(p.get("W"), errors="coerce")
    p["L"] = pd.to_numeric(p.get("L"), errors="coerce")
    p["D"] = pd.to_numeric(p.get("D"), errors="coerce")
    return p

@st.cache_data(ttl=60 * 60)
def ufc_get_fighter_profile(url: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    meta = {"url": url, "bytes": 0}
    html = ufc_fetch_html(url)
    meta["bytes"] = len(html or "")
    if not html:
        return {"Name": "", "Age": np.nan, "ReachIn": np.nan, "Stance": "", "Record": "", "W": np.nan, "L": np.nan, "D": np.nan,
                "SLpM": np.nan, "StrAcc": np.nan, "SApM": np.nan, "StrDef": np.nan, "TDAvg": np.nan, "TDAcc": np.nan, "TDDef": np.nan, "SubAvg": np.nan}, meta
    prof = _extract_fighter_profile(html)
    return prof, meta

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))

def ufc_predict(red: Dict[str, Any], blue: Dict[str, Any]) -> Dict[str, Any]:
    r = _coerce_profile(dict(red))
    b = _coerce_profile(dict(blue))

    def win_rate(p):
        w, l = p.get("W"), p.get("L")
        if pd.isna(w) or pd.isna(l) or (w + l) <= 0:
            return 0.5
        return float(w) / float(w + l)

    def nz(x): return 0.0 if pd.isna(x) else float(x)

    r_wr = win_rate(r)
    b_wr = win_rate(b)

    stance_edge = 0.0
    if isinstance(r.get("Stance"), str) and isinstance(b.get("Stance"), str):
        rs = r["Stance"].lower()
        bs = b["Stance"].lower()
        if ("southpaw" in rs) and ("orthodox" in bs):
            stance_edge = 0.03
        elif ("orthodox" in rs) and ("southpaw" in bs):
            stance_edge = -0.03

    d_reach = nz(r.get("ReachIn") - b.get("ReachIn")) if (pd.notna(r.get("ReachIn")) and pd.notna(b.get("ReachIn"))) else 0.0
    d_age = nz(r.get("Age") - b.get("Age")) if (pd.notna(r.get("Age")) and pd.notna(b.get("Age"))) else 0.0

    score = (
        0.06 * (d_reach / 5.0) +
        -0.07 * (d_age / 5.0) +
        0.10 * ((nz(r.get("SLpM")) - nz(b.get("SLpM"))) / 2.0) +
        -0.10 * ((nz(r.get("SApM")) - nz(b.get("SApM"))) / 2.0) +
        0.08 * ((nz(r.get("StrAcc")) - nz(b.get("StrAcc"))) / 0.10) +
        0.08 * ((nz(r.get("StrDef")) - nz(b.get("StrDef"))) / 0.10) +
        0.08 * ((nz(r.get("TDAvg")) - nz(b.get("TDAvg"))) / 2.0) +
        0.08 * ((nz(r.get("TDAcc")) - nz(b.get("TDAcc"))) / 0.10) +
        0.09 * ((nz(r.get("TDDef")) - nz(b.get("TDDef"))) / 0.10) +
        0.04 * ((nz(r.get("SubAvg")) - nz(b.get("SubAvg"))) / 1.0) +
        0.20 * ((nz(r_wr - b_wr)) / 0.20) +
        stance_edge
    )

    p_red = float(_sigmoid(score))
    red_name = r.get("Name") or "Red"
    blue_name = b.get("Name") or "Blue"

    if p_red >= 0.5:
        pick = red_name
        p = p_red
    else:
        pick = blue_name
        p = 1.0 - p_red

    # Heuristic: finish likelihood (proxy using volume + grappling activity)
    finish_score = (
        0.35 * (nz(r.get("SLpM")) + nz(b.get("SLpM"))) / 10.0 +
        0.35 * (nz(r.get("TDAvg")) + nz(b.get("TDAvg"))) / 8.0 +
        0.30 * (nz(r.get("SubAvg")) + nz(b.get("SubAvg"))) / 4.0
    )
    p_finish = float(np.clip(finish_score, 0.15, 0.75))

    return {
        "Pick": pick,
        "WinProb": float(np.clip(p, 0.52, 0.80)),
        "Finish%": round(p_finish * 100.0, 1),
        "Decision%": round((1.0 - p_finish) * 100.0, 1),
        "Score": float(score),
        "Red": red_name,
        "Blue": blue_name,
    }

def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC — Fight Picks (UFCStats)")
    st.caption("Uses UFCStats (upcoming-first). Model uses age/reach/stance/record + striking + takedowns. Finish/Decision is a heuristic proxy.")

    # This solves “tomorrow card exists but app shows none” due to caching/stale fetch
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Force refresh UFC"):
            try:
                st.cache_data.clear()
            except Exception:
                pass
            st.rerun()

    events, notes = ufc_list_events_cached()

    if debug:
        st.json(notes)

    if events.empty:
        st.warning("No UFC events found from UFCStats (upcoming/completed).")
        st.caption("If UFCStats loads in your browser but not in Streamlit, your host may block outbound requests to that domain.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Prefer closest upcoming if we have usable dates
    events2 = events.copy()
    if events2["Date"].notna().any():
        now = pd.Timestamp.now().normalize()
        future = events2[events2["Date"] >= now].copy()
        if not future.empty:
            future = future.sort_values("Date", ascending=True)
            rest = events2[~events2.index.isin(future.index)]
            events2 = pd.concat([future, rest], ignore_index=True).drop_duplicates(subset=["EventURL"], keep="first")

    # Build select options
    def fmt_row(r):
        d = r.get("Date")
        if pd.notna(d):
            return f"{pd.to_datetime(d).strftime('%Y-%m-%d')} — {r.get('Event','')}"
        ds = str(r.get("DateStr") or "").strip()
        return f"{ds} — {r.get('Event','')}".strip(" —")

    options = [fmt_row(row) for _, row in events2.iterrows()]
    idx = st.selectbox("Select UFC Event", list(range(len(options))), format_func=lambda i: options[i], index=0)
    ev_url = str(events2.iloc[idx]["EventURL"])

    fights, meta = ufc_get_event_fights(ev_url)
    if debug:
        st.json(meta)

    if not fights:
        st.warning("No fights parsed on this event page (HTML changed or blocked).")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    rows = []
    prof_fail = 0
    for f in fights:
        r_prof, r_meta = ufc_get_fighter_profile(f["RedURL"])
        b_prof, b_meta = ufc_get_fighter_profile(f["BlueURL"])
        if (r_meta.get("bytes", 0) == 0) or (b_meta.get("bytes", 0) == 0):
            prof_fail += 1
            continue
        pred = ufc_predict(r_prof, b_prof)
        rows.append({
            "Fight": f"{pred['Red']} vs {pred['Blue']}",
            "Pick": pred["Pick"],
            "WinProb%": round(pred["WinProb"] * 100.0, 1),
            "Finish%": pred["Finish%"],
            "Decision%": pred["Decision%"],
            "Red_Age": round(float(pd.to_numeric(r_prof.get("Age"), errors="coerce")), 1) if pd.notna(pd.to_numeric(r_prof.get("Age"), errors="coerce")) else np.nan,
            "Blue_Age": round(float(pd.to_numeric(b_prof.get("Age"), errors="coerce")), 1) if pd.notna(pd.to_numeric(b_prof.get("Age"), errors="coerce")) else np.nan,
            "Red_Reach": pd.to_numeric(r_prof.get("ReachIn"), errors="coerce"),
            "Blue_Reach": pd.to_numeric(b_prof.get("ReachIn"), errors="coerce"),
            "Red_Stance": r_prof.get("Stance", ""),
            "Blue_Stance": b_prof.get("Stance", ""),
            "Red_Record": r_prof.get("Record", ""),
            "Blue_Record": b_prof.get("Record", ""),
        })
        time.sleep(0.03)

    if not rows:
        st.warning("Could not compute predictions (fighter pages failed to load).")
        if debug:
            st.json({"fighter_page_failures": prof_fail, "fights_total": len(fights)})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(rows).sort_values("WinProb%", ascending=False).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if debug and prof_fail:
        st.info(f"Skipped {prof_fail} fight(s) due to fighter page load failures.")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# Tracker UI (kept identical)
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
