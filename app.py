# app.py
import os
import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================================
# Page + Responsive Theme (Desktop/Mobile)
# =========================================
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

@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .big-title { font-size: 1.35rem; }
  .subtle { font-size: 0.85rem; }
  .card { padding: 10px 10px; border-radius: 14px; }
  .stMarkdown p, .stCaption { font-size: 0.9rem; }
  canvas, svg, img { max-width: 100% !important; height: auto !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================
# Secrets / Env / Session Keys
# =========================================
def get_key(name: str, default: str = "") -> str:
    # session override
    if name in st.session_state and str(st.session_state[name]).strip():
        return str(st.session_state[name]).strip()
    # secrets
    if hasattr(st, "secrets") and name in st.secrets:
        v = str(st.secrets.get(name, "")).strip()
        if v:
            return v
    # env
    v = os.getenv(name, "").strip()
    if v:
        return v
    return default

ODDS_API_KEY = get_key("ODDS_API_KEY", "")
# Support BOTH secret names you mentioned
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "") or get_key("DATAGOLF_KEY", "")

# =========================================
# HTTP helpers
# =========================================
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

# =========================================
# Odds Math
# =========================================
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

# =========================================
# The Odds API Config
# =========================================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # ONLY DK/FD

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

# Common Odds API prop market keys
PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================================
# Sidebar UI (Radio Sections)
# =========================================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Edge = YourProb − ImpliedProb • Best Price DK/FD</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)

with st.sidebar.expander("API Keys (session override)", expanded=False):
    st.caption("Secrets should be preferred. If needed, paste keys here for this session only.")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY (or DATAGOLF_API_KEY)", value=DATAGOLF_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        # store under both names to be safe
        st.session_state["DATAGOLF_KEY"] = dg_in.strip()
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()

st.sidebar.markdown("---")
section = st.sidebar.radio("Section", ["Game Lines", "Player Props", "PGA"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================================
# Header
# =========================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Best bets are **value bets**: "
    "**Implied Probability** (from best available odds) vs **Your Probability** (no-vig baseline from DK/FD). "
    "**Edge = YourProb − ImpliedProb**. Only show **Edge > 0**. "
    "No contradictory picks (even off by 0.5; lines bucketed to nearest half point). "
    "Game Lines, Props, and PGA run **independently** with separate API calls/caches."
)

if not ODDS_API_KEY.strip():
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar.')
    st.stop()

# =========================================
# Caching (daily) — separate by function
# =========================================
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

# =========================================
# DataGolf (separate; only used in PGA)
# =========================================
DG_HOST = "https://feeds.datagolf.com"

@st.cache_data(ttl=60 * 60 * 12)
def dg_pre_tournament(tour: str = "pga"):
    # Uses DG’s baseline + course history & fit model; returns win/top10 probs etc.
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": tour,
        "add_position": "10",         # ensure top-10 is included (in addition to defaults)
        "dead_heat": "yes",
        "odds_format": "percent",     # probabilities as percent
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 12)
def dg_player_decompositions(tour: str = "pga"):
    # Predicted SG breakdowns (tee-to-green, putting, etc.) for upcoming event
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": tour, "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 24)
def dg_skill_ratings(display: str = "value"):
    # Skill ratings include many skills; we’ll use bogey avoidance if present + putting/t2g if present.
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": display, "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

# =========================================
# Normalizers
# =========================================
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
            book = bm.get("key") or bm.get("title")
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
    # event_payload is a dict for /events/{id}/odds
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    matchup = f"{away} @ {home}"

    for bm in (event_payload.get("bookmakers", []) or []):
        book = bm.get("key") or bm.get("title")
        for mk in (bm.get("markets", []) or []):
            mkey = mk.get("key")
            for out in (mk.get("outcomes", []) or []):
                player = out.get("name")
                side = out.get("description")  # Over/Under often; blank for some markets
                line = out.get("point")
                price = out.get("price")

                if player is None or price is None:
                    continue

                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": (str(side) if side is not None else "").strip(),
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

# =========================================
# Core Best-Bet Logic (Edge = YourProb - Implied)
# =========================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    return out

def compute_no_vig_two_way_within_book(df: pd.DataFrame, group_cols_book: list) -> pd.DataFrame:
    """
    For a given two-way set within a book (e.g., Over/Under, TeamA/TeamB),
    normalize implied probabilities to remove vig.
    """
    out = df.copy()
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    sums = out.groupby(group_cols_book)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out

def estimate_your_prob(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
    """
    YourProb baseline:
    - Use no-vig within book (two-way) when available, then average across books
    - Fallback to average implied across books
    """
    if df.empty:
        return df.copy()

    out = add_implied(df)
    out = compute_no_vig_two_way_within_book(out, group_cols_book=book_cols)

    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")

    out["YourProb"] = np.where(pd.notna(nv_avg), nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    return out

def best_price_and_edge(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
    """
    Pick best price across books for each unique bet, then compute:
    ImpliedBest (from BestPrice)
    Edge = YourProb - ImpliedBest
    EV = Edge * 100 (percentage points)
    """
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

def prevent_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    Keep only ONE pick per contradiction group (max Edge).
    This prevents Over & Under both showing, or both spread sides,
    even if off by half a point (LineBucket).
    """
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    return out.loc[idx].sort_values("Edge", ascending=False)

def keep_only_value_bets(df_best: pd.DataFrame) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    return out[out["Edge"] > 0].sort_values("Edge", ascending=False)

def finalize_display(df_best: pd.DataFrame) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["YourProb%"] = pct01_to_100(out["YourProb"])
    out["Implied%"] = pct01_to_100(out["ImpliedBest"])
    out["Edge%"] = pct01_to_100(out["Edge"])
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").round(2)
    out["⭐"] = "⭐"
    out["⭐ BestBook"] = out["⭐"] + " " + out["BestBook"].astype(str)
    return out

# =========================================
# Boards (independent calls per section)
# =========================================
def build_game_lines_board(sport: str, bet_type: str):
    sport_key = SPORT_KEYS_LINES[sport]
    market_key = GAME_MARKETS[bet_type]

    res = fetch_game_lines(sport_key)
    if debug:
        st.json({"endpoint": "odds(game_lines)", "status": res["status"], "url": res["url"], "params": res["params"]})

    if not res["ok"] or not is_list_of_dicts(res["payload"]):
        return pd.DataFrame(), {"error": "Game lines API failed", "status": res["status"], "payload": res["payload"]}

    df = normalize_lines(res["payload"])
    if df.empty:
        return pd.DataFrame(), {"error": "No normalized game lines"}

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        return pd.DataFrame(), {"error": "No rows for selected market"}

    # Keys
    key_cols = ["Event", "Market", "Outcome"]
    book_cols = ["Event", "Market", "Book"]
    best_cols = ["Event", "Market", "Outcome"]

    if market_key in ["spreads", "totals"]:
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]
        contradiction_cols = ["Event", "Market", "LineBucket"]
    else:
        contradiction_cols = ["Event", "Market"]  # moneyline: prevent both teams in same event

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_only_value_bets(df_best)
    df_best = finalize_display(df_best)

    return df_best, {}

def build_props_board(sport: str, prop_label: str, max_events_scan: int = 8, sleep_s: float = 0.06):
    sport_key = SPORT_KEYS_PROPS[sport]
    market_key = PROP_MARKETS[prop_label]

    ev_res = fetch_events(sport_key)
    if debug:
        st.json({"endpoint": "events", "status": ev_res["status"], "url": ev_res["url"], "params": ev_res["params"]})

    if not ev_res["ok"] or not is_list_of_dicts(ev_res["payload"]):
        return pd.DataFrame(), {"error": "Events API failed", "status": ev_res["status"], "payload": ev_res["payload"]}

    # Prefer soonest events; skip empties
    event_ids = []
    for e in ev_res["payload"]:
        if isinstance(e, dict) and e.get("id"):
            event_ids.append(e["id"])
    event_ids = event_ids[: int(max_events_scan)]
    if not event_ids:
        return pd.DataFrame(), {"error": "No upcoming events returned"}

    all_rows = []
    call_log = []
    for eid in event_ids:
        r = fetch_event_odds(sport_key, eid, market_key)
        call_log.append({"event_id": eid, "market": market_key, "status": r["status"], "ok": r["ok"]})

        # 422 means market not supported for that event (very common) -> skip quietly
        if not r["ok"] or not isinstance(r["payload"], dict):
            time.sleep(sleep_s)
            continue

        dfp = normalize_props(r["payload"])
        if not dfp.empty:
            all_rows.append(dfp)

        time.sleep(sleep_s)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {
            "error": "No props returned for DK/FD on scanned events (or this market isn’t available yet today).",
            "calls": call_log,
        }

    df = pd.concat(all_rows, ignore_index=True)

    # Keys
    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    best_cols = ["Event", "Market", "Player", "Side"]

    # Line bucket if present (prevents Over/Under contradictions even off by 0.5)
    has_line_bucket = "LineBucket" in df.columns and df["LineBucket"].notna().any()
    if has_line_bucket:
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]
        contradiction_cols = ["Event", "Market", "Player", "LineBucket"]
    else:
        # Anytime TD, etc.
        contradiction_cols = ["Event", "Market", "Player"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_only_value_bets(df_best)
    df_best = finalize_display(df_best)

    return df_best, {}

# =========================================
# Charts (percent axis)
# =========================================
def bar_prob(df, label_col, prob_col_percent, title):
    if df.empty or prob_col_percent not in df.columns:
        return
    fig = plt.figure()
    vals = pd.to_numeric(df[prob_col_percent], errors="coerce").fillna(0.0).values
    labs = df[label_col].astype(str).values
    plt.barh(labs, vals)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(100))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# =========================================
# PGA Module (advanced-ish using DataGolf)
# - Uses DG pre-tournament probabilities (includes course fit/history)
# - Adds SG components from player decompositions
# - Adds skill ratings if available (bogey avoidance / putting etc.)
# - Ranks WIN and TOP10 separately; shows top 10 picks each
# =========================================
def normalize_dg_pre_tournament(payload):
    # payload format can differ; we handle common patterns
    if not isinstance(payload, (dict, list)):
        return pd.DataFrame()

    # often returns dict with "preds" or "data"
    if isinstance(payload, dict):
        for k in ["preds", "data", "field", "players"]:
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break

    if not isinstance(payload, list):
        return pd.DataFrame()

    rows = []
    for r in payload:
        if not isinstance(r, dict):
            continue
        name = r.get("player_name") or r.get("name") or r.get("player")
        if not name:
            continue

        # Probabilities might come as percent already (0-100)
        win = r.get("win") or r.get("win_prob") or r.get("prob_win")
        top10 = r.get("top_10") or r.get("top10") or r.get("top_10_prob") or r.get("prob_top_10")
        # Course-fit model sometimes labeled; if present, prefer it
        win_ch = r.get("win_ch") or r.get("win_course_history") or r.get("win_fit") or r.get("win_fit_ch")

        rows.append({
            "Player": name,
            "DGID": r.get("dg_id") or r.get("player_id"),
            "WinPct_raw": win_ch if win_ch is not None else win,
            "Top10Pct_raw": top10,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["WinPct"] = pd.to_numeric(df["WinPct_raw"], errors="coerce")
    df["Top10Pct"] = pd.to_numeric(df["Top10Pct_raw"], errors="coerce")

    # convert percent -> 0-1
    df["WinProb"] = clamp01((df["WinPct"] / 100.0).fillna(np.nan))
    df["Top10Prob"] = clamp01((df["Top10Pct"] / 100.0).fillna(np.nan))
    return df.dropna(subset=["WinProb", "Top10Prob"], how="all")

def normalize_dg_decomp(payload):
    if not isinstance(payload, (dict, list)):
        return pd.DataFrame()
    if isinstance(payload, dict):
        # commonly 'players' or 'data'
        for k in ["players", "data", "preds"]:
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break
    if not isinstance(payload, list):
        return pd.DataFrame()

    rows = []
    for r in payload:
        if not isinstance(r, dict):
            continue
        name = r.get("player_name") or r.get("name") or r.get("player")
        if not name:
            continue

        # common fields (names vary; we grab what exists)
        t2g = r.get("sg_t2g") or r.get("t2g") or r.get("sg_ttg") or r.get("sg_tee_to_green")
        putt = r.get("sg_putt") or r.get("putt") or r.get("sg_putting")
        app = r.get("sg_app") or r.get("app") or r.get("sg_approach")
        ott = r.get("sg_ott") or r.get("ott") or r.get("sg_off_tee")
        arg = r.get("sg_arg") or r.get("arg") or r.get("sg_around_green")

        rows.append({
            "Player": name,
            "DGID": r.get("dg_id") or r.get("player_id"),
            "SG_T2G": t2g,
            "SG_Putt": putt,
            "SG_APP": app,
            "SG_OTT": ott,
            "SG_ARG": arg,
        })

    df = pd.DataFrame(rows)
    for c in ["SG_T2G", "SG_Putt", "SG_APP", "SG_OTT", "SG_ARG"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def normalize_dg_skill(payload):
    if not isinstance(payload, (dict, list)):
        return pd.DataFrame()
    if isinstance(payload, dict):
        for k in ["players", "data", "skill_ratings"]:
            if k in payload and isinstance(payload[k], list):
                payload = payload[k]
                break
    if not isinstance(payload, list):
        return pd.DataFrame()

    # We don’t know exact field names ahead of time; keep flexible.
    # We'll try to pick bogey avoidance & putting/t2g if present.
    rows = []
    for r in payload:
        if not isinstance(r, dict):
            continue
        name = r.get("player_name") or r.get("name") or r.get("player")
        if not name:
            continue

        # guess common keys
        bogey = r.get("bogey_avoidance") or r.get("bogey") or r.get("bogey_avoid")
        putt = r.get("putting") or r.get("sg_putt") or r.get("putt")
        t2g = r.get("tee_to_green") or r.get("sg_t2g") or r.get("t2g")

        rows.append({
            "Player": name,
            "DGID": r.get("dg_id") or r.get("player_id"),
            "BogeyAvoid": bogey,
            "Skill_Putt": putt,
            "Skill_T2G": t2g,
        })

    df = pd.DataFrame(rows)
    for c in ["BogeyAvoid", "Skill_Putt", "Skill_T2G"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def pga_board():
    if not DATAGOLF_API_KEY.strip():
        st.info('PGA is hidden until key is set. Add in Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY).')
        return

    res_pred = dg_pre_tournament("pga")
    res_dec = dg_player_decompositions("pga")
    res_skill = dg_skill_ratings("value")

    if debug:
        st.json({"dg_pre_tournament": {"ok": res_pred["ok"], "status": res_pred["status"], "url": res_pred["url"]}})
        st.json({"dg_decomp": {"ok": res_dec["ok"], "status": res_dec["status"], "url": res_dec["url"]}})
        st.json({"dg_skill": {"ok": res_skill["ok"], "status": res_skill["status"], "url": res_skill["url"]}})

    if not res_pred["ok"]:
        st.warning("DataGolf pre-tournament endpoint failed.")
        if debug:
            st.json(res_pred)
        return

    dfp = normalize_dg_pre_tournament(res_pred["payload"])
    if dfp.empty:
        st.warning("No PGA prediction rows returned from DataGolf.")
        if debug:
            st.json(res_pred["payload"])
        return

    # Merge SG decompositions + skill ratings (best-effort)
    dfd = normalize_dg_decomp(res_dec["payload"]) if res_dec["ok"] else pd.DataFrame()
    dfs = normalize_dg_skill(res_skill["payload"]) if res_skill["ok"] else pd.DataFrame()

    out = dfp.copy()
    if not dfd.empty:
        out = out.merge(dfd.drop_duplicates(subset=["Player"]), on="Player", how="left")
    if not dfs.empty:
        out = out.merge(dfs.drop_duplicates(subset=["Player"]), on="Player", how="left", suffixes=("", "_skill"))

    # “Advanced analytics” score: course fit/history is already baked into pre-tournament model.
    # We surface the drivers (SG_T2G, SG_Putt, BogeyAvoid) and rank by probabilities.
    out["Win%"] = pct01_to_100(out["WinProb"])
    out["Top10%"] = pct01_to_100(out["Top10Prob"])

    # Top 10 winners + Top 10 top10s
    top_win = out.sort_values("WinProb", ascending=False).head(10).copy()
    top_t10 = out.sort_values("Top10Prob", ascending=False).head(10).copy()

    st.subheader("PGA — Top 10 WIN Candidates (DataGolf course-fit/history + form baseline)")
    cols = ["Player", "Win%", "SG_T2G", "SG_Putt", "BogeyAvoid"]
    cols = [c for c in cols if c in top_win.columns]
    st.dataframe(top_win[cols], use_container_width=True, hide_index=True)

    st.subheader("PGA — Top 10 TOP-10 Candidates")
    cols2 = ["Player", "Top10%", "SG_T2G", "SG_Putt", "BogeyAvoid"]
    cols2 = [c for c in cols2 if c in top_t10.columns]
    st.dataframe(top_t10[cols2], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top 10)")
    chart = top_win.copy()
    chart["Label"] = chart["Player"].astype(str)
    bar_prob(chart, "Label", "Win%", "Win Probability (Top 10)")

# =========================================
# MAIN Sections
# =========================================
if section == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Independent dropdowns: these controls only affect game lines calls
    sport = st.selectbox("Sport (Game Lines)", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_bet_type")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="gl_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_top25")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No +EV game line bets right now."))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD) — +EV ONLY")
    st.caption("Ranked by **Edge** = YourProb − ImpliedProb(best price). Contradictions removed (half-point bucketed).")

    top = df_best.head(int(top_n)).copy()
    cols = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (+EV only)")
        snap = df_best.head(25).copy()
        cols2 = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif section == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Independent dropdowns: these controls only affect prop calls
    sport = st.selectbox("Sport (Props)", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_prop")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 12, 8, key="pp_scan")
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5, key="pp_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_top25")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No +EV props returned for DK/FD on scanned events."))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) — +EV ONLY")
    st.caption("Ranked by **Edge** = YourProb − ImpliedProb(best price). Contradictions removed (half-point bucketed).")

    top = df_best.head(int(top_n)).copy()
    cols = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (+EV only)")
        snap = df_best.head(25).copy()
        cols2 = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:  # PGA
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")

    if not DATAGOLF_API_KEY.strip():
        st.warning(
            'Missing DataGolf key. Add it in Streamlit Secrets as '
            'DATAGOLF_KEY="..." (or DATAGOLF_API_KEY="..."). '
            "PGA is hidden until then."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Run PGA independently (no interaction with Odds API calls)
    pga_board()

    st.markdown("</div>", unsafe_allow_html=True)
