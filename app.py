import os
import time
from datetime import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================
# Page + Responsive Theme
# =========================
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

# =========================
# Keys (Secrets -> Env -> Session override)
# =========================
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

# Accept either DATAGOLF_API_KEY OR DATAGOLF_KEY (you said secrets uses DATAGOLF_KEY)
DATAGOLF_API_KEY = (
    get_key("DATAGOLF_API_KEY", "").strip()
    or get_key("DATAGOLF_KEY", "").strip()
)

# =========================
# HTTP
# =========================
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

# =========================
# Odds math
# =========================
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

# =========================
# API Config (The Odds API)
# =========================
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

# NOTE: Keep these prop market keys separate from game lines.
# (player_anytime_td is confirmed working in your logs.)
PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_pass_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================
# Sidebar UI (Radio Buttons)
# =========================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Implied Prob • Your Prob • Edge • Best Price</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
compact = st.sidebar.toggle("Mobile / Compact layout", value=False)

modes = ["Game Lines", "Player Props", "PGA"]
mode = st.sidebar.radio("Mode", modes, index=0)

with st.sidebar.expander("API Keys (optional runtime entry)", expanded=False):
    st.caption("If Secrets aren’t set, paste keys here (session-only).")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF (DATAGOLF_API_KEY or DATAGOLF_KEY)", value=DATAGOLF_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Best bets = **Edge = YourProb − ImpliedProb(best price)** (American odds). "
    "Only show bets with Edge > 0. "
    "No contradictions anywhere (line bucketed to nearest 0.5). "
    "DK/FD only. Separate API calls per mode so nothing breaks the other."
)

# ==========================================================
# Caching (daily)
# ==========================================================
@st.cache_data(ttl=60 * 60 * 24)
def fetch_game_lines(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGION,
        "markets": ",".join(GAME_MARKETS.values()),
        "oddsFormat": "american",
        "bookmakers": BOOKMAKERS
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
        "bookmakers": BOOKMAKERS
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

# =========================
# DataGolf API
# =========================
DG_HOST = "https://feeds.datagolf.com"

@st.cache_data(ttl=60 * 60 * 6)
def dg_pre_tournament(tour="pga", add_position=10):
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": tour,
        "add_position": add_position,   # include top-10 prob output
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 6)
def dg_decompositions(tour="pga"):
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": tour, "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 6)
def dg_skill_ratings(tour="pga"):
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

# ==========================================================
# Normalizers
# ==========================================================
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
                side = out.get("description")  # often Over/Under; for ATD usually blank/None
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

# =========================
# DataGolf parsing FIX (model-aware)
# =========================
def _dg_extract_rows(payload, model_key: str | None = None):
    """
    DataGolf pre-tournament often returns predictions under model keys:
      payload["baseline"], payload["baseline_history_fit"], etc.
    This prevents 'parsed none' when meta exists but predictions aren't in payload["data"].
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []

    # Common containers
    for k in ["data", "predictions", "players", "results"]:
        if k in payload and isinstance(payload[k], list):
            return payload[k]

    # Model key at top-level
    if model_key and model_key in payload and isinstance(payload[model_key], list):
        return payload[model_key]

    # Infer from models_available if present
    models = payload.get("models_available")
    if isinstance(models, list) and models:
        for mk in models:
            if isinstance(mk, str) and mk in payload and isinstance(payload[mk], list):
                return payload[mk]

    # Last resort: any list-of-dicts
    for k, v in payload.items():
        if isinstance(v, list) and (len(v) == 0 or isinstance(v[0], dict)):
            if str(k).lower() in ["models_available"]:
                continue
            return v

    return []

def _first_present(d: dict, keys: list, default=np.nan):
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default

def normalize_pga(pre_payload, model_key: str, decomp_payload=None, skill_payload=None):
    # Pre-tournament rows
    pre_rows = _dg_extract_rows(pre_payload, model_key=model_key)
    if not pre_rows:
        return pd.DataFrame()

    pre_df = pd.DataFrame(pre_rows).copy()
    if pre_df.empty:
        return pre_df

    # Robust player name
    if "player_name" not in pre_df.columns:
        pre_df["player_name"] = pre_df.apply(
            lambda r: _first_present(
                r.to_dict(),
                ["player_name", "player", "name", "golfer", "player_display_name", "dg_player_name"],
                default=""
            ),
            axis=1,
        )
    pre_df["player_name"] = pre_df["player_name"].astype(str).str.strip()
    pre_df = pre_df[pre_df["player_name"] != ""].copy()

    # Robust win/top10 probabilities (DataGolf odds_format=percent => these are already in 0..100 sometimes)
    win_raw = None
    for c in ["win_prob", "win_probability", "win", "prob_win", "p_win", "win_odds"]:
        if c in pre_df.columns:
            win_raw = c
            break

    top10_raw = None
    for c in ["top10_prob", "top_10_prob", "top10", "prob_top10", "p_top10", "top10_probability"]:
        if c in pre_df.columns:
            top10_raw = c
            break

    # Convert to 0..1 safely
    def to_prob01(x):
        try:
            v = float(x)
        except Exception:
            return np.nan
        # If it's in 0..100, convert
        if v > 1.0:
            v = v / 100.0
        return v

    if win_raw is not None:
        pre_df["WinProb"] = pre_df[win_raw].apply(to_prob01)
    else:
        pre_df["WinProb"] = np.nan

    if top10_raw is not None:
        pre_df["Top10Prob"] = pre_df[top10_raw].apply(to_prob01)
    else:
        pre_df["Top10Prob"] = np.nan

    # Book odds columns (optional / best-effort)
    # If DataGolf includes win_odds or bookmaker odds, keep display fields but do not require them.
    pre_df["BestWinOddsDisp"] = ""
    pre_df["BestWinBook"] = ""

    # Merge decompositions (course fit/history/recent form/putting/t2g often live here)
    if isinstance(decomp_payload, dict) or isinstance(decomp_payload, list):
        drows = _dg_extract_rows(decomp_payload, model_key=None)
        ddf = pd.DataFrame(drows) if drows else pd.DataFrame()
    else:
        ddf = pd.DataFrame()

    if not ddf.empty:
        # find player name column
        pcol = None
        for c in ["player_name", "player", "name"]:
            if c in ddf.columns:
                pcol = c
                break
        if pcol is None:
            pcol = ddf.columns[0]
        ddf[pcol] = ddf[pcol].astype(str).str.strip()
        ddf = ddf.rename(columns={pcol: "player_name"})

        # try common signal cols (names vary by feed/version)
        rename_map = {}
        for src, dst in [
            ("sg_t2g", "sg_t2g"),
            ("strokes_gained_t2g", "sg_t2g"),
            ("sg_putt", "sg_putt"),
            ("strokes_gained_putting", "sg_putt"),
            ("bogey_avoidance", "bogey_avoidance"),
            ("course_fit", "course_fit"),
            ("course_history", "course_history"),
            ("history_fit", "course_history"),
            ("recent_form", "recent_form"),
            ("form", "recent_form"),
        ]:
            if src in ddf.columns:
                rename_map[src] = dst
        ddf = ddf.rename(columns=rename_map)

        keep = ["player_name"] + [c for c in ["sg_t2g", "sg_putt", "bogey_avoidance", "course_fit", "course_history", "recent_form"] if c in ddf.columns]
        ddf = ddf[keep].copy()

        for c in keep:
            if c != "player_name":
                ddf[c] = pd.to_numeric(ddf[c], errors="coerce")

        pre_df = pre_df.merge(ddf, on="player_name", how="left")

    # Merge skill ratings (fallback; sometimes contains SG splits)
    if isinstance(skill_payload, dict) or isinstance(skill_payload, list):
        srows = _dg_extract_rows(skill_payload, model_key=None)
        sdf = pd.DataFrame(srows) if srows else pd.DataFrame()
    else:
        sdf = pd.DataFrame()

    if not sdf.empty:
        pcol = None
        for c in ["player_name", "player", "name"]:
            if c in sdf.columns:
                pcol = c
                break
        if pcol is None:
            pcol = sdf.columns[0]
        sdf[pcol] = sdf[pcol].astype(str).str.strip()
        sdf = sdf.rename(columns={pcol: "player_name"})
        # choose a couple likely columns if present
        for c in sdf.columns:
            if c != "player_name":
                sdf[c] = pd.to_numeric(sdf[c], errors="coerce")
        # only merge columns that won't explode width
        candidate_cols = [c for c in ["sg_t2g", "sg_putt", "bogey_avoidance"] if c in sdf.columns]
        keep = ["player_name"] + candidate_cols
        sdf = sdf[keep].drop_duplicates("player_name")
        pre_df = pre_df.merge(sdf, on="player_name", how="left", suffixes=("", "_skill"))

        # fill missing from skill if needed
        for c in ["sg_t2g", "sg_putt", "bogey_avoidance"]:
            if c in pre_df.columns and f"{c}_skill" in pre_df.columns:
                pre_df[c] = pre_df[c].fillna(pre_df[f"{c}_skill"])
                pre_df = pre_df.drop(columns=[f"{c}_skill"])

    # Analytics score (robust, won't error if cols missing)
    def zscore(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() < 3:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

    comp = pd.Series(0.0, index=pre_df.index)
    if "course_fit" in pre_df.columns:
        comp += 0.25 * zscore(pre_df["course_fit"])
    if "course_history" in pre_df.columns:
        comp += 0.20 * zscore(pre_df["course_history"])
    if "recent_form" in pre_df.columns:
        comp += 0.20 * zscore(pre_df["recent_form"])
    if "sg_t2g" in pre_df.columns:
        comp += 0.20 * zscore(pre_df["sg_t2g"])
    if "sg_putt" in pre_df.columns:
        comp += 0.10 * zscore(pre_df["sg_putt"])
    if "bogey_avoidance" in pre_df.columns:
        comp += 0.05 * zscore(pre_df["bogey_avoidance"])

    pre_df["AnalyticsScore"] = comp.round(3)

    pre_df = pre_df.rename(columns={"player_name": "Player"})
    pre_df["WinProb"] = clamp01(pd.to_numeric(pre_df["WinProb"], errors="coerce").fillna(0.0001))
    pre_df["Top10Prob"] = clamp01(pd.to_numeric(pre_df["Top10Prob"], errors="coerce").fillna(0.0001))

    pre_df["WinProb%"] = pct01_to_100(pre_df["WinProb"])
    pre_df["Top10Prob%"] = pct01_to_100(pre_df["Top10Prob"])

    # Edge fields (only if we also have implied from some odds; keep NaN otherwise)
    pre_df["WinEdge"] = np.nan
    pre_df["WinEdge%"] = np.nan

    return pre_df

# ==========================================================
# Core Best-Bet Logic (Implied vs YourProb)
# ==========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    return out

def compute_no_vig_two_way_within_book(df: pd.DataFrame, group_cols_book: list) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    sums = out.groupby(group_cols_book)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out

def estimate_your_prob(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
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
    best["EV"] = (best["Edge"] * 100.0)  # percentage points
    return best

def prevent_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    out = out.loc[idx].sort_values("Edge", ascending=False)
    return out

def keep_only_value_bets(df_best: pd.DataFrame) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    return out[out["Edge"] > 0].sort_values("Edge", ascending=False)

# ==========================================================
# Boards (run independently per radio mode)
# ==========================================================
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
        contradiction_cols = ["Event", "Market", "LineBucket"]
    else:
        contradiction_cols = ["Event", "Market"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    # prevents contradictions across BOTH books because BestPrice is chosen per bet
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_only_value_bets(df_best)

    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    return df_best, {}

def build_props_board(sport: str, prop_label: str, max_events_scan: int = 5):
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

        # IMPORTANT: only normalize rows that match the market_key we requested
        dfp = normalize_props(r["payload"])
        if not dfp.empty:
            dfp = dfp[dfp["Market"] == market_key].copy()
        if not dfp.empty:
            all_rows.append(dfp)

        time.sleep(0.06)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {"error": "No props returned for DK/FD on scanned events (or market not posted yet).", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    best_cols = ["Event", "Market", "Player", "Side"]

    has_line_bucket = "LineBucket" in df.columns and df["LineBucket"].notna().any()
    if has_line_bucket:
        key_cols += ["LineBucket"]
        book_cols += ["LineBucket"]
        best_cols += ["LineBucket"]
        contradiction_cols = ["Event", "Market", "Player", "LineBucket"]
    else:
        # Anytime TD: one pick per player per event
        contradiction_cols = ["Event", "Market", "Player"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    # prevents contradictory picks across DK/FD: keep only best Edge per contradiction group
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_only_value_bets(df_best)

    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    return df_best, {}

# ==========================================================
# Charts (percent axis)
# ==========================================================
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

# =========================
# Hard requirements
# =========================
if mode in ["Game Lines", "Player Props"] and not ODDS_API_KEY.strip():
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar expander.')
    st.stop()

# ==========================================================
# MAIN — Radio modes (independent)
# ==========================================================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_type")
    top_n = st.slider("Top picks (EDGE)", 2, 10, 5, key="gl_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_top25")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No +EV game line bets right now."))
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD) — +EV ONLY")
    st.caption("Ranked by Edge (not YourProb%). Contradictions removed (half-point bucketed). Best price is starred (BestBook).")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

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
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_type")  # includes Anytime TD
    top_n = st.slider("Top picks (EDGE)", 2, 10, 5, key="pp_topn")
    show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_top25")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 10, 5, key="pp_scan")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No +EV props returned for DK/FD on scanned events."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) — +EV ONLY")
    st.caption("Ranked by Edge (not YourProb%). Contradictions removed (half-point bucketed). Best price is starred (BestBook).")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

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
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # PGA runs independently and does NOT touch odds-api endpoints
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")
    if not DATAGOLF_API_KEY.strip():
        st.warning('DATAGOLF key not set. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY="..."). PGA is hidden until then.')
        st.stop()

    pre = dg_pre_tournament(tour="pga", add_position=10)
    decomp = dg_decompositions(tour="pga")
    skill = dg_skill_ratings(tour="pga")

    meta = pre["payload"] if isinstance(pre["payload"], dict) else {}
    models_available = meta.get("models_available", [])
    if not isinstance(models_available, list) or not models_available:
        models_available = ["baseline", "baseline_history_fit"]

    model_key = st.selectbox(
        "Model",
        options=models_available,
        index=0 if "baseline_history_fit" not in models_available else models_available.index("baseline_history_fit"),
        help="DataGolf returns predictions under these model keys. baseline_history_fit usually includes course history/fit."
    )

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]},
            "dg_decomp": {"ok": decomp["ok"], "status": decomp["status"], "url": decomp["url"]},
            "dg_skill": {"ok": skill["ok"], "status": skill["status"], "url": skill["url"]},
            "dg_meta": {
                "event_name": meta.get("event_name"),
                "last_updated": meta.get("last_updated"),
                "models_available": meta.get("models_available"),
                "using_model": model_key
            }
        })

    if not pre["ok"]:
        st.error("DataGolf pre-tournament feed failed.")
        st.stop()

    dfpga = normalize_pga(
        pre["payload"],
        model_key=model_key,
        decomp_payload=(decomp["payload"] if decomp["ok"] else None),
        skill_payload=(skill["payload"] if skill["ok"] else None),
    )

    if dfpga.empty:
        st.error(f"No PGA prediction rows returned from DataGolf for model='{model_key}'.")
        st.stop()

    # Build rankings (no odds-based EV required; relies on prob + analytics)
    dfpga = dfpga.copy()
    dfpga["WinRankScore"] = dfpga["WinProb"].fillna(0) + 0.02 * dfpga["AnalyticsScore"].fillna(0)
    dfpga["Top10RankScore"] = dfpga["Top10Prob"].fillna(0) + 0.03 * dfpga["AnalyticsScore"].fillna(0)

    win = dfpga.sort_values("WinRankScore", ascending=False).head(10).copy()
    t10 = dfpga.sort_values("Top10RankScore", ascending=False).head(10).copy()

    st.markdown("### Top 10 Winners (Model + Course/History/Form/SG)")
    win_cols = ["Player", "WinProb%", "AnalyticsScore"]
    for c in ["course_fit", "course_history", "recent_form", "sg_t2g", "sg_putt", "bogey_avoidance"]:
        if c in win.columns:
            win_cols.append(c)
    st.dataframe(win[win_cols], use_container_width=True, hide_index=True)

    st.markdown("### Top 10 Top-10 Picks (Model + Course/History/Form/SG)")
    t10_cols = ["Player", "Top10Prob%", "AnalyticsScore"]
    for c in ["course_fit", "course_history", "recent_form", "sg_t2g", "sg_putt", "bogey_avoidance"]:
        if c in t10.columns:
            t10_cols.append(c)
    st.dataframe(t10[t10_cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top picks)")
    cwin = win.copy()
    cwin["Label"] = cwin["Player"].astype(str)
    bar_prob(cwin, "Label", "WinProb%", "Win Probability (Top Picks)")

    ct10 = t10.copy()
    ct10["Label"] = ct10["Player"].astype(str)
    bar_prob(ct10, "Label", "Top10Prob%", "Top-10 Probability (Top Picks)")

    st.markdown("</div>", unsafe_allow_html=True)
