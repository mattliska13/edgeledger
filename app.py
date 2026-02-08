# app.py — EdgeLedger
# CHANGE REQUEST IMPLEMENTED:
# ✅ Tracker is now an auto “Pick History” that can:
#   - Capture each night’s Top 25 picks (Moneyline + Spreads) for a selected sport
#   - Auto-grade results (W/L/P) using ESPN scoreboard finals
#   - Show win% + live weekly/monthly win% split by Moneyline vs Spreads
#   - Does NOT impact other modules (Game Lines / Props / PGA logic unchanged)
#
# Notes:
# - Uses SQLite (no CSV) for pick history + results.
# - “Capture Top 25” is idempotent per date/market (won’t duplicate).
# - Auto-grading is best-effort based on matching team names to ESPN.
# - Props/PGA logging still works (stored too), but the Tracker dashboard focuses on spread/moneyline stats.

import os
import time
import re
import sqlite3
from datetime import datetime, timezone, date
from typing import Dict, Any, List, Optional, Tuple

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

def safe_get(url: str, params: Optional[dict] = None, timeout: int = 25):
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

def utc_now():
    return datetime.now(timezone.utc)


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

mode = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA", "Tracker"], index=0)

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

if not ODDS_API_KEY.strip() and mode != "Tracker":
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
                        "HomeTeam": home,
                        "AwayTeam": away,
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

def filter_value(df_best: pd.DataFrame, show_non_value_flag: bool) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    out = out.sort_values("Edge", ascending=False)
    if show_non_value_flag:
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
# Tracker (NEW) — Nightly Top25 Picks + Auto Results (SQLite)
# =========================================================
PICKS_DB = "pick_history.sqlite"

PICK_COLUMNS = [
    "PickDate", "LoggedAt",
    "Mode", "Sport", "Market", "Event",
    "Selection", "Line",
    "BestBook", "BestPrice",
    "YourProb", "Implied", "Edge", "EV",
    "Status", "Result",
    "Commence", "HomeTeam", "AwayTeam"
]

def _pick_db() -> sqlite3.Connection:
    conn = sqlite3.connect(PICKS_DB, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            PickDate TEXT,
            LoggedAt TEXT,
            Mode TEXT,
            Sport TEXT,
            Market TEXT,
            Event TEXT,
            Selection TEXT,
            Line REAL,
            BestBook TEXT,
            BestPrice REAL,
            YourProb REAL,
            Implied REAL,
            Edge REAL,
            EV REAL,
            Status TEXT,
            Result TEXT,
            Commence TEXT,
            HomeTeam TEXT,
            AwayTeam TEXT
        )
    """)
    # Idempotency: one row per date + sport + market + event + selection + line
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_picks
        ON picks (PickDate, Sport, Market, Event, Selection, COALESCE(Line, -9999))
    """)
    conn.commit()
    return conn

def _tracker_load_all() -> pd.DataFrame:
    try:
        conn = _pick_db()
        df = pd.read_sql_query("SELECT * FROM picks", conn)
        conn.close()
        if df.empty:
            return pd.DataFrame(columns=PICK_COLUMNS)
        for c in PICK_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[PICK_COLUMNS].copy()
    except Exception:
        return pd.DataFrame(columns=PICK_COLUMNS)

def tracker_log_rows(rows: List[Dict[str, Any]]):
    """
    Compatible with the existing app: all existing "Log to Tracker" buttons still call this.
    Stored in SQLite now, not CSV.
    """
    if not rows:
        return _tracker_load_all()

    add = pd.DataFrame(rows)
    if add.empty:
        return _tracker_load_all()

    # ensure columns
    for c in PICK_COLUMNS:
        if c not in add.columns:
            add[c] = np.nan

    # default PickDate if missing
    if "PickDate" in add.columns:
        add["PickDate"] = add["PickDate"].fillna(pd.Timestamp.now().date().isoformat())
    else:
        add["PickDate"] = pd.Timestamp.now().date().isoformat()

    if "LoggedAt" in add.columns:
        add["LoggedAt"] = add["LoggedAt"].fillna(datetime.now().isoformat())
    else:
        add["LoggedAt"] = datetime.now().isoformat()

    add = add[PICK_COLUMNS].copy()

    try:
        conn = _pick_db()
        # insert row-by-row to honor unique index (ignore duplicates)
        cur = conn.cursor()
        sql = """
            INSERT OR IGNORE INTO picks (
                PickDate, LoggedAt, Mode, Sport, Market, Event, Selection, Line,
                BestBook, BestPrice, YourProb, Implied, Edge, EV, Status, Result, Commence, HomeTeam, AwayTeam
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        for _, r in add.iterrows():
            cur.execute(sql, tuple(r.values.tolist()))
        conn.commit()
        conn.close()
    except Exception:
        pass

    return _tracker_load_all()


# =========================================================
# ESPN Results (for auto-grading)
# =========================================================
def _norm_team(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

TEAM_ALIASES = {
    # NFL common
    "la rams": "los angeles rams",
    "la chargers": "los angeles chargers",
    "ny jets": "new york jets",
    "ny giants": "new york giants",
    "washington": "washington commanders",
}

def _apply_alias(n: str) -> str:
    return TEAM_ALIASES.get(n, n)

def _espn_sport_meta(sport: str) -> Tuple[str, str]:
    if sport == "NFL":
        return ("football", "nfl")
    if sport == "CFB":
        return ("football", "college-football")
    if sport == "CBB":
        return ("basketball", "mens-college-basketball")
    return ("football", "nfl")

@st.cache_data(ttl=60 * 20)
def espn_scoreboard(sport: str) -> Dict[str, Any]:
    cat, lg = _espn_sport_meta(sport)
    url = f"https://site.api.espn.com/apis/site/v2/sports/{cat}/{lg}/scoreboard"
    ok, status, payload, final_url = safe_get(url, params=None, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def _extract_finals_from_scoreboard(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()

    evs = payload.get("events")
    if not isinstance(evs, list):
        return pd.DataFrame()

    rows = []
    for ev in evs:
        comps = (ev or {}).get("competitions") or []
        if not comps:
            continue
        comp = comps[0] or {}
        status = (comp.get("status") or {}).get("type") or {}
        if status.get("state") != "post":
            continue

        competitors = comp.get("competitors") or []
        home = away = ""
        hs = as_ = None
        for c in competitors:
            if not isinstance(c, dict):
                continue
            ht = c.get("homeAway")
            team = c.get("team") or {}
            nm = team.get("displayName") or team.get("name") or ""
            sc = c.get("score")
            try:
                sc = int(sc)
            except Exception:
                sc = None
            if ht == "home":
                home = nm
                hs = sc
            elif ht == "away":
                away = nm
                as_ = sc

        dt = comp.get("date") or ev.get("date")
        rows.append({
            "Home": home,
            "Away": away,
            "HomeKey": _apply_alias(_norm_team(home)),
            "AwayKey": _apply_alias(_norm_team(away)),
            "HomeScore": hs,
            "AwayScore": as_,
            "DateUTC": pd.to_datetime(dt, errors="coerce", utc=True),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["HomeKey","AwayKey","HomeScore","AwayScore"])


def _parse_matchup(event_str: str) -> Tuple[str, str]:
    if not isinstance(event_str, str) or "@" not in event_str:
        return "", ""
    parts = [p.strip() for p in event_str.split("@", 1)]
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def tracker_autograde_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Grades spreads + moneyline picks that are Pending and have a matching ESPN final.
    Updates DB in-place.
    """
    if df.empty:
        return df, {"graded": 0, "checked": 0, "notes": "No picks"}

    # Focus on spreads + moneyline for requested stats
    x = df.copy()
    x["Status"] = x["Status"].fillna("Pending")
    x["Market"] = x["Market"].astype(str)

    todo = x[(x["Status"] != "Graded") & (x["Market"].isin(["Moneyline", "Spreads"]))].copy()
    checked = int(len(todo))
    if todo.empty:
        return df, {"graded": 0, "checked": checked, "notes": "Nothing to grade"}

    notes = []
    graded = 0

    # group by sport to limit scoreboard calls
    for sport in sorted(todo["Sport"].dropna().unique().tolist()):
        sb = espn_scoreboard(str(sport))
        if not sb.get("ok"):
            notes.append(f"{sport}: ESPN scoreboard unavailable")
            continue
        finals = _extract_finals_from_scoreboard(sb.get("payload", {}))
        if finals.empty:
            notes.append(f"{sport}: no finals in scoreboard payload")
            continue

        # build matchup key
        finals["MatchKey"] = finals["AwayKey"] + " @ " + finals["HomeKey"]

        # picks for this sport
        sub = todo[todo["Sport"] == sport].copy()
        if sub.empty:
            continue

        # normalize pick matchup
        away_list, home_list, mkeys = [], [], []
        for ev in sub["Event"].astype(str).tolist():
            a, h = _parse_matchup(ev)
            ak = _apply_alias(_norm_team(a))
            hk = _apply_alias(_norm_team(h))
            away_list.append(ak)
            home_list.append(hk)
            mkeys.append(f"{ak} @ {hk}")
        sub["AwayKey"] = away_list
        sub["HomeKey"] = home_list
        sub["MatchKey"] = mkeys

        merged = sub.merge(finals[["MatchKey","HomeScore","AwayScore"]], on="MatchKey", how="left")
        merged = merged.dropna(subset=["HomeScore","AwayScore"])

        if merged.empty:
            continue

        # compute results
        updates = []
        for _, r in merged.iterrows():
            market = str(r.get("Market", ""))
            sel = str(r.get("Selection", "")).strip()
            line = r.get("Line", np.nan)
            hs = float(r["HomeScore"])
            as_ = float(r["AwayScore"])

            # determine selected team key
            sel_key = _apply_alias(_norm_team(sel))

            result = ""
            if market == "Moneyline":
                # winner key
                if as_ > hs:
                    win_key = r["AwayKey"]
                elif hs > as_:
                    win_key = r["HomeKey"]
                else:
                    win_key = ""  # tie
                if win_key and sel_key == win_key:
                    result = "W"
                elif win_key:
                    result = "L"
                else:
                    result = "P"

            elif market == "Spreads":
                try:
                    sp = float(line)
                except Exception:
                    sp = np.nan
                if np.isnan(sp):
                    continue

                # which side is selection?
                is_sel_home = (sel_key == r["HomeKey"])
                is_sel_away = (sel_key == r["AwayKey"])
                if not (is_sel_home or is_sel_away):
                    continue

                ts = hs if is_sel_home else as_
                os_ = as_ if is_sel_home else hs

                # cover test: team score + spread vs opp score
                val = ts + sp - os_
                if val > 0:
                    result = "W"
                elif val < 0:
                    result = "L"
                else:
                    result = "P"

            if result:
                updates.append((result, "Graded", r["PickDate"], r["Sport"], r["Market"], r["Event"], r["Selection"], r.get("Line", None)))

        if not updates:
            continue

        # write updates to DB
        try:
            conn = _pick_db()
            cur = conn.cursor()
            for res_val, status_val, pdate, sp, mk, ev, sel, ln in updates:
                cur.execute(
                    """
                    UPDATE picks
                    SET Result = ?, Status = ?
                    WHERE PickDate = ? AND Sport = ? AND Market = ? AND Event = ? AND Selection = ?
                      AND ( (Line IS NULL AND ? IS NULL) OR (Line = ?) )
                    """,
                    (res_val, status_val, pdate, sp, mk, ev, sel, ln, ln)
                )
            conn.commit()
            conn.close()
            graded += len(updates)
        except Exception as e:
            notes.append(f"{sport}: DB update failed: {e}")

    df2 = _tracker_load_all()
    return df2, {"graded": graded, "checked": checked, "notes": "; ".join(notes) if notes else ""}


def tracker_stats(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      - summary_today_week_month (by Market)
      - history table (last 30 days)
    """
    if df.empty:
        empty = pd.DataFrame(columns=["Window","Market","Picks","Graded","Wins","Losses","Pushes","Win%"])
        return {"summary": empty, "history": pd.DataFrame()}

    x = df.copy()
    x["PickDate_dt"] = pd.to_datetime(x["PickDate"], errors="coerce")
    x["Status"] = x["Status"].fillna("Pending")
    x["Result"] = x["Result"].fillna("")
    x = x[x["Market"].isin(["Moneyline","Spreads"])].copy()

    now = pd.Timestamp.now()
    today = now.normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)

    windows = {
        "Today": x[x["PickDate_dt"] >= today],
        "This Week": x[x["PickDate_dt"] >= week_start],
        "This Month": x[x["PickDate_dt"] >= month_start],
    }

    tables = []
    for label, sub in windows.items():
        if sub.empty:
            continue
        y = sub.copy()
        y["Picks"] = 1
        y["Graded"] = (y["Status"] == "Graded").astype(int)
        y["Wins"] = (y["Result"] == "W").astype(int)
        y["Losses"] = (y["Result"] == "L").astype(int)
        y["Pushes"] = (y["Result"] == "P").astype(int)

        agg = y.groupby(["Market"], dropna=False).agg(
            Picks=("Picks","sum"),
            Graded=("Graded","sum"),
            Wins=("Wins","sum"),
            Losses=("Losses","sum"),
            Pushes=("Pushes","sum"),
        ).reset_index()

        denom = (agg["Wins"] + agg["Losses"] + agg["Pushes"]).replace(0, np.nan)
        agg["Win%"] = ((agg["Wins"] / denom) * 100.0).round(1).fillna(0.0)
        agg.insert(0, "Window", label)
        tables.append(agg)

    summary = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(
        columns=["Window","Market","Picks","Graded","Wins","Losses","Pushes","Win%"]
    )

    history = x.sort_values(["PickDate_dt","LoggedAt"], ascending=False).head(300).copy()
    return {"summary": summary, "history": history}


# =========================================================
# Boards
# =========================================================
def build_game_lines_board(sport: str, bet_type: str, show_non_value_override: Optional[bool] = None):
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

    # allow Tracker capture to override non-value filtering without impacting UI settings
    flag = show_non_value if show_non_value_override is None else bool(show_non_value_override)
    df_best = filter_value(df_best, show_non_value_flag=flag)

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

    df_best = filter_value(df_best, show_non_value_flag=show_non_value)
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
# Tracker Page UI (REPLACED)
# =========================================================
def capture_top25_for_market(sport: str, market_label: str, pick_date: str):
    """
    Captures Top 25 picks for Moneyline/Spreads from the existing model.
    Does NOT affect other modules.
    """
    df_best, err = build_game_lines_board(sport, market_label, show_non_value_override=True)
    if df_best.empty:
        return 0, err

    top25 = df_best.head(25).copy()

    rows = []
    for _, r in top25.iterrows():
        rows.append({
            "PickDate": pick_date,
            "LoggedAt": datetime.now().isoformat(),
            "Mode": "Nightly Top25",
            "Sport": sport,
            "Market": market_label,
            "Event": r.get("Event",""),
            "Selection": r.get("Outcome",""),
            "Line": r.get("LineBucket", np.nan) if market_label == "Spreads" else np.nan,
            "BestBook": r.get("BestBook",""),
            "BestPrice": r.get("BestPrice", np.nan),
            "YourProb": r.get("YourProb", np.nan),
            "Implied": r.get("ImpliedBest", np.nan),
            "Edge": r.get("Edge", np.nan),
            "EV": r.get("EV", np.nan),
            "Status": "Pending",
            "Result": "",
            "Commence": r.get("Commence",""),
            "HomeTeam": r.get("HomeTeam",""),
            "AwayTeam": r.get("AwayTeam",""),
        })

    tracker_log_rows(rows)
    return len(rows), {}

def render_tracker():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Tracker — Nightly Top 25 History (Auto Results)")
    st.caption("Capture Top 25 Moneyline + Spreads each night, then auto-grade from ESPN finals. Live weekly/monthly win% shown below.")

    df = _tracker_load_all()

    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])
    with c1:
        sport = st.selectbox("Sport (for capture + grading)", list(SPORT_KEYS_LINES.keys()), index=0, key="trk_sport")
    with c2:
        pick_date = st.date_input("Pick Date", value=date.today(), key="trk_date").isoformat()
    with c3:
        st.caption("Tip: Run capture once daily. Auto-grade after games finish.")

    cap1, cap2, cap3 = st.columns([1, 1, 2])
    with cap1:
        if st.button("Capture Top25 Moneyline"):
            n, err = capture_top25_for_market(sport, "Moneyline", pick_date)
            if n:
                st.success(f"Captured {n} Moneyline picks for {pick_date}.")
            else:
                st.warning(err.get("error", "Could not capture Moneyline picks."))
                if debug:
                    st.json(err)
    with cap2:
        if st.button("Capture Top25 Spreads"):
            n, err = capture_top25_for_market(sport, "Spreads", pick_date)
            if n:
                st.success(f"Captured {n} Spreads picks for {pick_date}.")
            else:
                st.warning(err.get("error", "Could not capture Spreads picks."))
                if debug:
                    st.json(err)
    with cap3:
        if st.button("Auto-grade results now"):
            df2, info = tracker_autograde_results(df)
            df = df2
            msg = f"Checked {info.get('checked',0)} picks; graded {info.get('graded',0)}."
            if info.get("notes"):
                msg += f" Notes: {info['notes']}"
            st.info(msg)

    # Always show current stats
    st.markdown("### Live Win% (Moneyline vs Spreads)")
    out = tracker_stats(df)
    summary = out["summary"]
    if summary.empty:
        st.info("No picks captured yet. Use the capture buttons above.")
    else:
        st.dataframe(
            summary[["Window","Market","Picks","Graded","Wins","Losses","Pushes","Win%"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Pick History (most recent first)")
    hist = out["history"]
    if hist.empty:
        st.info("No pick history yet.")
    else:
        show = hist.copy()
        # nice formatting
        show["YourProb%"] = pct01_to_100(show["YourProb"])
        show["Implied%"] = pct01_to_100(show["Implied"])
        show["Edge%"] = pct01_to_100(show["Edge"])
        cols = [
            "PickDate","Sport","Market","Event","Selection","Line",
            "BestBook","BestPrice","YourProb%","Implied%","Edge%","Result","Status"
        ]
        cols = [c for c in cols if c in show.columns]
        st.dataframe(show[cols].head(200), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# MAIN
# =========================================================
if mode == "Tracker":
    render_tracker()
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

    # Log Top Picks (still works; now stored in SQLite)
    if st.button("Log these Top Picks to Tracker"):
        rows = []
        for _, r in top.iterrows():
            rows.append({
                "PickDate": pd.Timestamp.now().date().isoformat(),
                "LoggedAt": datetime.now().isoformat(),
                "Mode": "Game Lines",
                "Sport": sport,
                "Market": bet_type,
                "Event": r.get("Event", ""),
                "Selection": r.get("Outcome", ""),
                "Line": r.get("LineBucket", np.nan),
                "BestBook": r.get("BestBook",""),
                "BestPrice": r.get("BestPrice",""),
                "YourProb": r.get("YourProb", np.nan),
                "Implied": r.get("ImpliedBest", np.nan),
                "Edge": r.get("Edge", np.nan),
                "EV": r.get("EV", np.nan),
                "Status": "Pending",
                "Result": "",
                "Commence": r.get("Commence",""),
                "HomeTeam": r.get("HomeTeam",""),
                "AwayTeam": r.get("AwayTeam",""),
            })
        tracker_log_rows(rows)
        st.success("Logged to Tracker ✅ (go to Tracker tab to auto-grade / view history).")

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
    st.caption("Strict rule: only ONE pick per player per market per game. Ranked by Edge.")

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
                "PickDate": pd.Timestamp.now().date().isoformat(),
                "LoggedAt": datetime.now().isoformat(),
                "Mode": "Player Props",
                "Sport": sport,
                "Market": prop_label,
                "Event": r.get("Event", ""),
                "Selection": sel,
                "Line": r.get("LineBucket", np.nan),
                "BestBook": r.get("BestBook",""),
                "BestPrice": r.get("BestPrice",""),
                "YourProb": r.get("YourProb", np.nan),
                "Implied": r.get("ImpliedBest", np.nan),
                "Edge": r.get("Edge", np.nan),
                "EV": r.get("EV", np.nan),
                "Status": "Pending",
                "Result": "",
                "Commence": "",
                "HomeTeam": "",
                "AwayTeam": "",
            })
        tracker_log_rows(rows)
        st.success("Logged to Tracker ✅")

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
                    "PickDate": pd.Timestamp.now().date().isoformat(),
                    "LoggedAt": datetime.now().isoformat(),
                    "Mode": "PGA",
                    "Sport": "PGA",
                    "Market": "Win",
                    "Event": out.get("meta", {}).get("event_name", ""),
                    "Selection": r.get("Player", ""),
                    "Line": np.nan,
                    "BestBook": "",
                    "BestPrice": np.nan,
                    "YourProb": pd.to_numeric(r.get("WinProb", np.nan), errors="coerce"),
                    "Implied": np.nan,
                    "Edge": np.nan,
                    "EV": np.nan,
                    "Status": "Pending",
                    "Result": "",
                    "Commence": "",
                    "HomeTeam": "",
                    "AwayTeam": "",
                })
            tracker_log_rows(rows)
            st.success("Logged PGA picks ✅")

    else:
        st.warning(err.get("error", "No PGA data available right now."))
        if debug:
            st.json(err)

    st.markdown("</div>", unsafe_allow_html=True)
