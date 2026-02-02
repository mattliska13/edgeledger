# app.py
import os
import time
import re
import math
import html as ihtml
from datetime import datetime, date

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
# HTTP
# =========================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/1.0 (+streamlit)"})

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

if not ODDS_API_KEY.strip() and mode not in ("Tracker", "UFC"):
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar expander.')
    st.stop()

# =========================================================
# Caching (daily) — Odds API
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
# UFC — UFCStats-only Picks Module (no lxml, no csv)
# =========================================================
UFCSTATS_BASE = "http://ufcstats.com"

def _utc_now_ts() -> pd.Timestamp:
    # Always a pandas Timestamp; safe for comparisons
    return pd.Timestamp.utcnow()

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _strip(s):
    return re.sub(r"\s+", " ", (s or "")).strip()

def _html_unescape(s: str) -> str:
    try:
        return ihtml.unescape(s)
    except Exception:
        return s

@st.cache_data(ttl=60 * 10)
def ufc_fetch_text(url: str) -> dict:
    ok, status, payload, final_url = safe_get(url, params=None, timeout=25)
    if ok and isinstance(payload, str):
        return {"ok": True, "status": status, "text": payload, "url": final_url}
    # If json came back unexpectedly, coerce to string
    return {"ok": False, "status": status, "text": str(payload), "url": final_url}

def _parse_event_rows(html_text: str) -> list[dict]:
    # UFCStats events pages contain <a href=".../event-details/..."> and a date cell nearby.
    # We'll parse link + title, and also attempt to parse date strings (e.g., "Jan 27, 2024").
    out = []
    # Find table rows
    for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S):
        row = m.group(1)
        link_m = re.search(r'href="([^"]*event-details[^"]*)"', row, flags=re.I)
        if not link_m:
            continue
        url = link_m.group(1)
        if url.startswith("/"):
            url = UFCSTATS_BASE + url
        # title is usually link text
        title_m = re.search(r'href="[^"]*event-details[^"]*"\s*>(.*?)</a>', row, flags=re.I | re.S)
        title = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", title_m.group(1))) if title_m else "")
        # date is often in a <td> with month day year
        date_m = re.search(r"([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})", row)
        dt = None
        if date_m:
            try:
                dt = pd.to_datetime(date_m.group(1), errors="coerce")
            except Exception:
                dt = None
        out.append({"title": title or "UFC Event", "date": dt, "url": url})
    # Dedup by url
    seen = set()
    uniq = []
    for r in out:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        uniq.append(r)
    return uniq

def ufc_list_upcoming_events() -> dict:
    # UFCStats has a dedicated upcoming page sometimes; if not, fall back to completed page and just pick the next-ish.
    candidates = [
        f"{UFCSTATS_BASE}/statistics/events/upcoming?page=all",
        f"{UFCSTATS_BASE}/statistics/events/upcoming?page=0",
        f"{UFCSTATS_BASE}/statistics/events/completed?page=all",
        f"{UFCSTATS_BASE}/statistics/events/completed?page=0",
    ]
    attempts = []
    events = []
    for u in candidates:
        r = ufc_fetch_text(u)
        attempts.append({"url": r["url"], "ok": r["ok"], "status": r["status"]})
        if not r["ok"]:
            continue
        rows = _parse_event_rows(r["text"])
        if rows:
            events = rows
            break

    if not events:
        return {"ok": False, "error": "No UFC events found from UFCStats right now (site may be blocking or markup changed).", "attempts": attempts}

    # Prefer events with a date; sort ascending; pick next upcoming relative to utc-now - 8h (buffer)
    now = _utc_now_ts()
    cutoff = now - pd.Timedelta(hours=8)

    dated = [e for e in events if isinstance(e.get("date"), pd.Timestamp) and not pd.isna(e.get("date"))]
    if dated:
        dated = sorted(dated, key=lambda x: x["date"])
        upcoming = [e for e in dated if e["date"].tz_localize(None) >= cutoff.tz_localize(None)]
        use = upcoming[0] if upcoming else dated[-1]
    else:
        # If no dates parsed, just take the first
        use = events[0]

    return {"ok": True, "event": use, "events": events[:25], "attempts": attempts}

def _parse_fight_links(event_html: str) -> list[str]:
    links = re.findall(r'href="([^"]*fight-details[^"]*)"', event_html, flags=re.I)
    out = []
    for u in links:
        if u.startswith("/"):
            u = UFCSTATS_BASE + u
        if u.startswith("http"):
            out.append(u)
    # Dedup preserve order
    seen = set()
    uniq = []
    for u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
    return uniq

def _parse_fight_teams(fight_html: str) -> dict:
    # Fight details page includes links to fighter-details for red/blue corners.
    fighter_urls = re.findall(r'href="([^"]*fighter-details[^"]*)"', fight_html, flags=re.I)
    fighter_urls = [UFCSTATS_BASE + u if u.startswith("/") else u for u in fighter_urls]
    fighter_urls = [u for u in fighter_urls if u.startswith("http")]

    # Names: appear in <h2 class="b-fight-details__persons-title"> ... <a>NAME</a>
    # We'll just collect unique anchor texts adjacent to fighter-details links.
    names = []
    for m in re.finditer(r'href="[^"]*fighter-details[^"]*"\s*>(.*?)</a>', fight_html, flags=re.I | re.S):
        nm = _strip(_html_unescape(re.sub(r"<[^>]+>", " ", m.group(1))))
        if nm:
            names.append(nm)

    # Typically first two fighter-details links are the two fighters; sometimes repeated (ref links).
    # We'll take first two unique fighter urls.
    seen = set()
    uniq_urls = []
    for u in fighter_urls:
        if u in seen:
            continue
        seen.add(u)
        uniq_urls.append(u)
        if len(uniq_urls) >= 2:
            break

    # For names, take first two unique as well.
    seen_n = set()
    uniq_names = []
    for n in names:
        if n in seen_n:
            continue
        seen_n.add(n)
        uniq_names.append(n)
        if len(uniq_names) >= 2:
            break

    if len(uniq_urls) < 2:
        return {"ok": False, "error": "Could not parse fighters from fight-details page."}

    return {
        "ok": True,
        "fighter1_url": uniq_urls[0],
        "fighter2_url": uniq_urls[1],
        "fighter1_name": uniq_names[0] if len(uniq_names) > 0 else "Fighter A",
        "fighter2_name": uniq_names[1] if len(uniq_names) > 1 else "Fighter B",
    }

def _parse_stat_block(html_text: str, label: str) -> str:
    # Pull value next to a label on UFCStats fighter pages.
    # e.g. <i class="b-list__box-item-title">Reach:</i> 72"
    pat = rf"{re.escape(label)}\s*</i>\s*([^<]+)"
    m = re.search(pat, html_text, flags=re.I)
    if not m:
        return ""
    return _strip(_html_unescape(m.group(1)))

def _parse_record(html_text: str) -> tuple[int,int,int]:
    # record appears like "Record: 10-2-0"
    rec = _parse_stat_block(html_text, "Record:")
    m = re.search(r"(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", rec)
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))

def _parse_height_to_in(s: str) -> float:
    # 5' 11"
    s = s.replace(" ", "")
    m = re.search(r"(\d+)'(\d+)", s)
    if not m:
        return np.nan
    ft = int(m.group(1)); inch = int(m.group(2))
    return ft * 12 + inch

def _parse_reach_in(s: str) -> float:
    # 72" or 72
    m = re.search(r"(\d+(\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan

def _parse_dob_age(s: str) -> float:
    # DOB format like "Jan 01, 1990"
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return np.nan
        today = pd.Timestamp.utcnow().date()
        born = dt.date()
        age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
        return float(age)
    except Exception:
        return np.nan

def _parse_pct(s: str) -> float:
    # "45%" -> 0.45
    m = re.search(r"(\d+(\.\d+)?)\s*%", s)
    if not m:
        return np.nan
    return float(m.group(1)) / 100.0

def _parse_float(s: str) -> float:
    m = re.search(r"(-?\d+(\.\d+)?)", (s or ""))
    return float(m.group(1)) if m else np.nan

def _parse_last5_from_fights(html_text: str) -> dict:
    # Fighter page includes a fight history table; we parse W/L and Method for last 5.
    rows = re.findall(r"<tr[^>]*class=\"b-fight-details__table-row\"[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S)
    # Some pages omit class on header; keep only rows with at least one <td>
    parsed = []
    for r in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.I | re.S)
        if len(tds) < 6:
            continue
        # Result is first td, method is often near end (varies); we use a heuristic:
        res_txt = _strip(re.sub(r"<[^>]+>", " ", tds[0]))
        method_txt = _strip(re.sub(r"<[^>]+>", " ", " ".join(tds)))
        parsed.append({"result": res_txt, "method_blob": method_txt})

    # If we didn’t get anything, fallback: generic <tr> parsing and read first td
    if not parsed:
        for m in re.finditer(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S):
            row = m.group(1)
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.I | re.S)
            if len(tds) < 6:
                continue
            res_txt = _strip(re.sub(r"<[^>]+>", " ", tds[0]))
            method_txt = _strip(re.sub(r"<[^>]+>", " ", " ".join(tds)))
            parsed.append({"result": res_txt, "method_blob": method_txt})

    parsed = parsed[:5]
    if not parsed:
        return {"last5_winrate": np.nan, "last5_itd_rate": np.nan, "last5_dec_rate": np.nan}

    w = sum(1 for x in parsed if x["result"].upper().startswith("W"))
    n = len(parsed)
    itd = 0
    dec = 0
    for x in parsed:
        blob = x["method_blob"].upper()
        if "DEC" in blob:
            dec += 1
        if ("KO" in blob) or ("TKO" in blob) or ("SUB" in blob):
            itd += 1
    return {
        "last5_winrate": w / n if n else np.nan,
        "last5_itd_rate": itd / n if n else np.nan,
        "last5_dec_rate": dec / n if n else np.nan,
    }

def _parse_overall_finish_rates(html_text: str) -> dict:
    # From fight history rows; compute overall ITD vs decision share.
    rows = re.findall(r"<tr[^>]*class=\"b-fight-details__table-row\"[^>]*>(.*?)</tr>", html_text, flags=re.I | re.S)
    parsed = []
    for r in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.I | re.S)
        if len(tds) < 6:
            continue
        parsed.append(_strip(re.sub(r"<[^>]+>", " ", " ".join(tds))).upper())
    if not parsed:
        return {"itd_rate": np.nan, "dec_rate": np.nan}

    itd = sum(1 for b in parsed if ("KO" in b) or ("TKO" in b) or ("SUB" in b))
    dec = sum(1 for b in parsed if "DEC" in b)
    n = len(parsed)
    return {"itd_rate": itd / n if n else np.nan, "dec_rate": dec / n if n else np.nan}

@st.cache_data(ttl=60 * 30)
def ufc_fetch_fighter_profile(fighter_url: str) -> dict:
    r = ufc_fetch_text(fighter_url)
    if not r["ok"]:
        return {"ok": False, "error": "fighter page fetch failed", "status": r["status"], "url": r["url"]}

    h = r["text"]
    height = _parse_height_to_in(_parse_stat_block(h, "Height:"))
    reach = _parse_reach_in(_parse_stat_block(h, "Reach:"))
    stance = _strip(_parse_stat_block(h, "STANCE:")) or _strip(_parse_stat_block(h, "Stance:"))
    dob = _strip(_parse_stat_block(h, "DOB:"))
    age = _parse_dob_age(dob)
    w, l, d = _parse_record(h)

    # Career averages table: try to pull the common labels
    # These labels appear as "SLpM:", "Str. Acc.:", "SApM:", "Str. Def:", "TD Avg.:", "TD Acc.:", "TD Def.:", "Sub. Avg.:"
    slpm = _parse_float(_parse_stat_block(h, "SLpM:"))
    sapm = _parse_float(_parse_stat_block(h, "SApM:"))
    str_acc = _parse_pct(_parse_stat_block(h, "Str. Acc.:"))
    str_def = _parse_pct(_parse_stat_block(h, "Str. Def:"))
    td_avg = _parse_float(_parse_stat_block(h, "TD Avg.:"))
    td_acc = _parse_pct(_parse_stat_block(h, "TD Acc.:"))
    td_def = _parse_pct(_parse_stat_block(h, "TD Def.:"))
    sub_avg = _parse_float(_parse_stat_block(h, "Sub. Avg.:"))

    last5 = _parse_last5_from_fights(h)
    fin = _parse_overall_finish_rates(h)

    total = w + l + d
    win_pct = (w / total) if total > 0 else np.nan

    # "Rank" is not reliably available on UFCStats; use a stable proxy that still helps:
    # more UFC fights + higher win pct -> higher proxy
    rank_proxy = (math.log1p(total) * (win_pct if not np.isnan(win_pct) else 0.0))

    return {
        "ok": True,
        "url": r["url"],
        "age": age,
        "reach_in": reach,
        "height_in": height,
        "stance": stance,
        "wins": w,
        "losses": l,
        "draws": d,
        "total_fights": total,
        "win_pct": win_pct,
        "rank_proxy": rank_proxy,

        "slpm": slpm,
        "sapm": sapm,
        "str_acc": str_acc,
        "str_def": str_def,
        "td_avg": td_avg,
        "td_acc": td_acc,
        "td_def": td_def,
        "sub_avg": sub_avg,

        "last5_winrate": last5.get("last5_winrate"),
        "last5_itd_rate": last5.get("last5_itd_rate"),
        "last5_dec_rate": last5.get("last5_dec_rate"),
        "itd_rate": fin.get("itd_rate"),
        "dec_rate": fin.get("dec_rate"),
    }

def _z_pair(a: float, b: float, scale: float, flip: bool=False) -> float:
    # normalized difference, stable even if nan
    if np.isnan(a) and np.isnan(b):
        return 0.0
    if np.isnan(a):
        a = b
    if np.isnan(b):
        b = a
    diff = (a - b) / scale if scale else (a - b)
    return -diff if flip else diff

def ufc_predict_fight(f1: dict, f2: dict) -> dict:
    """
    Model uses:
    - age, reach, stance (tiny), record (win%), last5 winrate, rank_proxy
    - striking: slpm, sapm, str_acc, str_def
    - grappling: td_avg, td_acc, td_def, sub_avg
    - finish/decision: itd_rate, dec_rate + last5 proxies (heuristics)
    """
    # feature diffs (f1 - f2)
    d_age = _z_pair(f1.get("age", np.nan), f2.get("age", np.nan), scale=6.0, flip=True)   # younger better
    d_reach = _z_pair(f1.get("reach_in", np.nan), f2.get("reach_in", np.nan), scale=4.0)
    d_win = _z_pair(f1.get("win_pct", np.nan), f2.get("win_pct", np.nan), scale=0.15)
    d_l5 = _z_pair(f1.get("last5_winrate", np.nan), f2.get("last5_winrate", np.nan), scale=0.20)
    d_rank = _z_pair(f1.get("rank_proxy", np.nan), f2.get("rank_proxy", np.nan), scale=0.25)

    # striking edges
    net1 = (f1.get("slpm", np.nan) - f1.get("sapm", np.nan)) if (not np.isnan(f1.get("slpm", np.nan)) and not np.isnan(f1.get("sapm", np.nan))) else np.nan
    net2 = (f2.get("slpm", np.nan) - f2.get("sapm", np.nan)) if (not np.isnan(f2.get("slpm", np.nan)) and not np.isnan(f2.get("sapm", np.nan))) else np.nan
    d_net = _z_pair(net1, net2, scale=1.25)
    d_strdef = _z_pair(f1.get("str_def", np.nan), f2.get("str_def", np.nan), scale=0.12)
    d_stracc = _z_pair(f1.get("str_acc", np.nan), f2.get("str_acc", np.nan), scale=0.10)

    # grappling edges
    d_tdavg = _z_pair(f1.get("td_avg", np.nan), f2.get("td_avg", np.nan), scale=1.3)
    d_tdacc = _z_pair(f1.get("td_acc", np.nan), f2.get("td_acc", np.nan), scale=0.12)
    d_tddef = _z_pair(f1.get("td_def", np.nan), f2.get("td_def", np.nan), scale=0.12)
    d_sub = _z_pair(f1.get("sub_avg", np.nan), f2.get("sub_avg", np.nan), scale=0.8)

    # stance tiny heuristic: orthodox vs southpaw sometimes matters, but keep small weight
    s1 = (f1.get("stance") or "").lower()
    s2 = (f2.get("stance") or "").lower()
    d_stance = 0.0
    if s1 and s2 and s1 != s2:
        d_stance = 0.05  # tiny edge for variety; basically a coin-flip adjustment

    # finish heuristics
    d_itd = _z_pair(f1.get("itd_rate", np.nan), f2.get("itd_rate", np.nan), scale=0.25)
    d_l5_itd = _z_pair(f1.get("last5_itd_rate", np.nan), f2.get("last5_itd_rate", np.nan), scale=0.30)
    d_dec = _z_pair(f1.get("dec_rate", np.nan), f2.get("dec_rate", np.nan), scale=0.25)

    # score (logit-ish)
    score = (
        0.18 * d_win +
        0.14 * d_l5 +
        0.10 * d_rank +
        0.08 * d_age +
        0.06 * d_reach +
        0.14 * d_net +
        0.06 * d_strdef +
        0.04 * d_stracc +
        0.07 * d_tdavg +
        0.05 * d_tdacc +
        0.07 * d_tddef +
        0.03 * d_sub +
        0.03 * d_stance +
        0.05 * d_itd +
        0.03 * d_l5_itd -
        0.02 * d_dec
    )

    # convert to win prob-ish (50-85 band)
    p = _sigmoid(score)
    conf = 0.50 + (min(0.35, abs(p - 0.5) * 1.4))  # cap ~85%

    winner = "F1" if score >= 0 else "F2"

    # method heuristic (proxy)
    # If winner has high itd and opponent has weak defenses, lean ITD; else decision.
    if winner == "F1":
        w_itd = f1.get("itd_rate", np.nan)
        o_strdef = f2.get("str_def", np.nan)
        o_tddef = f2.get("td_def", np.nan)
    else:
        w_itd = f2.get("itd_rate", np.nan)
        o_strdef = f1.get("str_def", np.nan)
        o_tddef = f1.get("td_def", np.nan)

    itd_signal = 0.0
    if not np.isnan(w_itd):
        itd_signal += (w_itd - 0.45)
    if not np.isnan(o_strdef):
        itd_signal += (0.55 - o_strdef)
    if not np.isnan(o_tddef):
        itd_signal += (0.55 - o_tddef)

    method = "Decision"
    if itd_signal > 0.15:
        method = "Inside Distance"

    return {
        "score": float(score),
        "winner": winner,
        "confidence": float(conf),
        "method": method,
        "p_raw": float(p),
    }

def ufc_build_picks_for_event(event_url: str) -> dict:
    ev = ufc_fetch_text(event_url)
    if not ev["ok"]:
        return {"ok": False, "error": "Could not load UFC event page from UFCStats.", "status": ev["status"], "url": ev["url"]}

    fight_links = _parse_fight_links(ev["text"])
    if not fight_links:
        return {"ok": False, "error": "No fights parsed on this event page (HTML changed or blocked).", "event_url": ev["url"]}

    fights = []
    diag = {"event_url": ev["url"], "fight_count": len(fight_links), "fight_links_sample": fight_links[:3]}

    for fl in fight_links[:20]:  # safety cap
        fh = ufc_fetch_text(fl)
        if not fh["ok"]:
            continue
        team = _parse_fight_teams(fh["text"])
        if not team["ok"]:
            continue

        f1 = ufc_fetch_fighter_profile(team["fighter1_url"])
        f2 = ufc_fetch_fighter_profile(team["fighter2_url"])
        if not f1.get("ok") or not f2.get("ok"):
            continue

        pred = ufc_predict_fight(f1, f2)

        # label pick name
        pick_name = team["fighter1_name"] if pred["winner"] == "F1" else team["fighter2_name"]
        opp_name = team["fighter2_name"] if pred["winner"] == "F1" else team["fighter1_name"]

        fights.append({
            "Fight": f"{team['fighter1_name']} vs {team['fighter2_name']}",
            "Pick": pick_name,
            "Confidence%": round(pred["confidence"] * 100.0, 1),
            "Method": pred["method"],
            "Score": round(pred["score"], 3),

            "A_Age": f1.get("age"),
            "B_Age": f2.get("age"),
            "A_Reach": f1.get("reach_in"),
            "B_Reach": f2.get("reach_in"),
            "A_Stance": f1.get("stance"),
            "B_Stance": f2.get("stance"),

            "A_Record": f"{f1.get('wins',0)}-{f1.get('losses',0)}-{f1.get('draws',0)}",
            "B_Record": f"{f2.get('wins',0)}-{f2.get('losses',0)}-{f2.get('draws',0)}",
            "A_Win%": f1.get("win_pct"),
            "B_Win%": f2.get("win_pct"),
            "A_Last5W%": f1.get("last5_winrate"),
            "B_Last5W%": f2.get("last5_winrate"),

            "A_SLpM": f1.get("slpm"),
            "B_SLpM": f2.get("slpm"),
            "A_SApM": f1.get("sapm"),
            "B_SApM": f2.get("sapm"),
            "A_StrDef": f1.get("str_def"),
            "B_StrDef": f2.get("str_def"),
            "A_TDavg": f1.get("td_avg"),
            "B_TDavg": f2.get("td_avg"),
            "A_TDAcc": f1.get("td_acc"),
            "B_TDAcc": f2.get("td_acc"),
            "A_TDDef": f1.get("td_def"),
            "B_TDDef": f2.get("td_def"),

            "A_ITD": f1.get("itd_rate"),
            "B_ITD": f2.get("itd_rate"),
            "A_Dec": f1.get("dec_rate"),
            "B_Dec": f2.get("dec_rate"),
            "Notes": f"{pick_name} over {opp_name}",
        })

        time.sleep(0.03)

    if not fights:
        return {"ok": False, "error": "Could not build UFC picks for this event (fight list parsed but model table empty).", "diag": diag}

    df = pd.DataFrame(fights)

    # clean formats
    for c in ["A_Win%","B_Win%","A_Last5W%","B_Last5W%","A_StrDef","B_StrDef","A_TDAcc","B_TDAcc","A_TDDef","B_TDDef","A_ITD","B_ITD","A_Dec","B_Dec"]:
        if c in df.columns:
            df[c] = (pd.to_numeric(df[c], errors="coerce") * 100.0).round(1)

    return {"ok": True, "df": df, "diag": diag}

def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC — Picks (UFCStats-only)")
    st.caption(
        "Uses UFCStats upcoming events first. Model uses age/reach/stance/record + last five + rank proxy + striking + takedowns. "
        "Finish/Decision is a heuristic proxy."
    )

    # Event picker (auto selects next)
    ev = ufc_list_upcoming_events()
    if debug:
        st.json({"ufc_events_attempts": ev.get("attempts"), "ufc_event_selected": ev.get("event")})

    if not ev.get("ok"):
        st.warning(ev.get("error", "Could not load UFC events."))
        if debug:
            st.json(ev)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    selected = ev["event"]
    events = ev.get("events", [])
    options = [(e.get("title","UFC Event"), e.get("url")) for e in events if e.get("url")]
    default_idx = 0
    for i, (_, u) in enumerate(options):
        if u == selected.get("url"):
            default_idx = i
            break

    label_only = [f"{t}" for (t, _) in options]
    pick_idx = st.selectbox("Event", list(range(len(label_only))), format_func=lambda i: label_only[i], index=default_idx)
    event_url = options[pick_idx][1]

    if st.button("Refresh UFC (clear cache)"):
        st.cache_data.clear()
        st.rerun()

    out = ufc_build_picks_for_event(event_url)

    if not out.get("ok"):
        st.warning(out.get("error", "Could not build UFC picks for this event."))
        if debug:
            st.json(out)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df = out["df"].copy()
    df = df.sort_values(["Confidence%","Score"], ascending=False)

    show_cols = [
        "Fight","Pick","Confidence%","Method",
        "A_Age","B_Age","A_Reach","B_Reach","A_Stance","B_Stance",
        "A_Record","B_Record","A_Win%","B_Win%","A_Last5W%","B_Last5W%",
        "A_SLpM","B_SLpM","A_SApM","B_SApM","A_StrDef","B_StrDef",
        "A_TDavg","B_TDavg","A_TDAcc","B_TDAcc","A_TDDef","B_TDDef",
        "A_ITD","B_ITD","A_Dec","B_Dec",
        "Notes"
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    st.markdown("### Picks")
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

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
