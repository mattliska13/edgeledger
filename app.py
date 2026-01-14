import os
import time
from datetime import datetime
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

# =========================================================
# Keys (Secrets -> Env -> Session override)
# NOTE: you said your secrets are:
#   ODDS_API_KEY="..."
#   DATAGOLF_KEY="..."
# We'll support BOTH DATAGOLF_KEY and DATAGOLF_API_KEY names.
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
DATAGOLF_API_KEY = (
    get_key("DATAGOLF_API_KEY", "") or
    get_key("DATAGOLF_KEY", "")  # your current secrets name
)

# =========================================================
# HTTP
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
# Odds math (ROBUST: handles American OR Decimal returns)
# =========================================================
def decimal_to_implied(dec) -> float:
    try:
        d = float(dec)
    except Exception:
        return np.nan
    if d <= 1.0:
        return np.nan
    return 1.0 / d

def american_to_implied(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o == 0:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)

def price_to_implied(price) -> float:
    """
    Robust implied probability:
    - Decimal odds usually in [1.01, ~20]
    - Otherwise treat as American
    """
    try:
        p = float(price)
    except Exception:
        return np.nan

    if 1.01 <= p <= 20.0:
        return decimal_to_implied(p)

    return american_to_implied(p)

def decimal_to_american(dec) -> float:
    d = float(dec)
    if d >= 2.0:
        return round((d - 1.0) * 100.0, 0)
    return round(-100.0 / (d - 1.0), 0)

def price_to_american_display(price):
    """
    Return American odds string even if input is decimal.
    """
    try:
        p = float(price)
    except Exception:
        return ""
    if 1.01 <= p <= 20.0:
        a = decimal_to_american(p)
    else:
        a = p
    a = int(a) if abs(a - int(a)) < 1e-9 else a
    return f"+{a}" if float(a) > 0 else f"{a}"

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

# Use EXACT keys you asked for
PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_passing_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rushing_yds",
    "Receiving Yards": "player_receiving_yds",
    "Receptions": "player_receptions",
}

# =========================================================
# Sidebar UI
# =========================================================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Implied • YourProb • Edge • Best Price</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)
compact = st.sidebar.toggle("Mobile / Compact layout", value=False)

# Radio modes (independent, separate API calls)
modes = ["Game Lines", "Player Props"]
if DATAGOLF_API_KEY.strip():
    modes.append("PGA")

mode = st.sidebar.radio("Mode", modes, index=0)

with st.sidebar.expander("API Keys (session-only override)", expanded=False):
    st.caption("If Secrets are set, you don’t need this.")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY (or DATAGOLF_API_KEY)", value=DATAGOLF_API_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_API_KEY"] = dg_in.strip()
        DATAGOLF_API_KEY = dg_in.strip()

if st.sidebar.button("🔄 Refresh data (clear cache)"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Reloading...")

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "Best bets = **Edge = YourProb − ImpliedProb(best price)**. "
    "Ranked by **Edge** (descending). "
    "**No contradictions** (including across DK vs FD) by half-point bucketing + one-pick-per-bucket logic."
)

if not ODDS_API_KEY.strip():
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar.')
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

# =========================================================
# DataGolf (cached daily)
# =========================================================
DG_HOST = "https://feeds.datagolf.com"

@st.cache_data(ttl=60 * 60 * 24)
def dg_pre_tournament():
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": "pga",
        "add_position": 10,
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 24)
def dg_decompositions():
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": "pga", "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60 * 60 * 24)
def dg_skill_ratings():
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}
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
            book = bm.get("key") or bm.get("title")
            for mk in (bm.get("markets", []) or []):
                mkey = mk.get("key")
                for out in (mk.get("outcomes", []) or []):
                    line = out.get("point")
                    price = out.get("price")
                    rows.append({
                        "Scope": "GameLine",
                        "Event": matchup,
                        "Commence": commence,
                        "Market": mkey,
                        "Outcome": out.get("name"),
                        "Line": line,
                        "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                        "Price": price,
                        "Book": book,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["OddsDisplay"] = df["Price"].apply(price_to_american_display)
    return df.dropna(subset=["Market", "Outcome", "Price"])

def normalize_props(event_payload):
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
                # The Odds API uses 'description' for Over/Under in many props
                side = out.get("description") or out.get("label") or ""
                line = out.get("point")
                price = out.get("price")

                if player is None or price is None:
                    continue

                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": str(side).strip(),
                    "Line": line,
                    "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                    "Price": price,
                    "Book": book
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["OddsDisplay"] = df["Price"].apply(price_to_american_display)
    return df.dropna(subset=["Market", "Player", "Price"])

# =========================================================
# Core Best-Bet Logic (Implied vs YourProb)
#   - Baseline "YourProb" = no-vig within-book (2-way) then avg across books.
#   - BestBook = lowest implied prob (best payout) across DK/FD
#   - Contradictions removed (across books too) by group-based max Edge
# =========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(price_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    if "OddsDisplay" not in out.columns:
        out["OddsDisplay"] = out["Price"].apply(price_to_american_display)
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
    """
    BestBook per bet = lowest implied probability (works for American OR Decimal).
    """
    if df.empty:
        return df.copy()

    out = df.copy()
    if "Implied" not in out.columns:
        out = add_implied(out)

    out["Implied"] = pd.to_numeric(out["Implied"], errors="coerce")
    out = out.dropna(subset=["Implied", "Price"])

    idx = out.groupby(group_cols_best)["Implied"].idxmin()
    best = out.loc[idx].copy()

    best = best.rename(columns={
        "Price": "BestPriceRaw",
        "Book": "BestBook",
        "OddsDisplay": "BestOdds"
    })
    best["ImpliedBest"] = best["Implied"]

    best["Edge"] = (pd.to_numeric(best["YourProb"], errors="coerce") - pd.to_numeric(best["ImpliedBest"], errors="coerce"))
    best["EV"] = (best["Edge"] * 100.0)
    return best

def prevent_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    Keep only ONE pick per contradiction group (max Edge).
    This prevents:
      - Over & Under both showing
      - both spread sides showing
      - across DK/FD as well (since we pick only 1 row per group anyway)
    """
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

# =========================================================
# Boards (independent calls per mode)
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
        contradiction_cols = ["Event", "Market", "LineBucket"]
    else:
        contradiction_cols = ["Event", "Market"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    # Remove contradictions across DK/FD and within lines
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)

    # +EV only
    df_best = keep_only_value_bets(df_best)

    # Display
    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    df_best = df_best.sort_values("Edge", ascending=False)
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
        dfp = normalize_props(r["payload"])
        if not dfp.empty:
            all_rows.append(dfp)
        time.sleep(0.06)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {"error": "No props returned for DK/FD on scanned events (or market not posted yet).", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    # Keys
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
        contradiction_cols = ["Event", "Market", "Player"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)

    # Remove contradictions (Over vs Under, etc.) across books (DK/FD) too
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)

    # +EV only
    df_best = keep_only_value_bets(df_best)

    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}

# =========================================================
# Charts (percent axis)
# =========================================================
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

# =========================================================
# PGA: advanced picks using DataGolf
# Fix: robustly extract rows even when payload shape changes.
# Logic:
#   - Use baseline_history_fit if available, else baseline
#   - Combine:
#       win_prob (model) + form/fit signals (decompositions + skill ratings)
#   - Provide top 10 winners & top 10 top-10 bets (if available)
# =========================================================
def find_rows_with_player_name(payload):
    """
    Robustly search nested JSON for a list[dict] containing 'player_name'
    """
    if isinstance(payload, list):
        if len(payload) > 0 and isinstance(payload[0], dict) and ("player_name" in payload[0] or "player" in payload[0]):
            return payload
        # search inside items
        for item in payload:
            res = find_rows_with_player_name(item)
            if res is not None:
                return res
        return None

    if isinstance(payload, dict):
        # common keys
        for k in ["preds", "data", "results", "players"]:
            if k in payload and isinstance(payload[k], list):
                res = find_rows_with_player_name(payload[k])
                if res is not None:
                    return res
        # search any list values
        for v in payload.values():
            res = find_rows_with_player_name(v)
            if res is not None:
                return res
    return None

def to_prob01_from_percent(x):
    try:
        v = float(x)
    except Exception:
        return np.nan
    # DataGolf odds_format=percent -> 0..100
    if v > 1.0:
        return v / 100.0
    return v

def zscore(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd

def build_pga_board():
    if not DATAGOLF_API_KEY.strip():
        return pd.DataFrame(), {"error": 'Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY).'}
    pre = dg_pre_tournament()
    dec = dg_decompositions()
    skl = dg_skill_ratings()

    if debug:
        st.json({"dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]}})
        st.json({"dg_decomp": {"ok": dec["ok"], "status": dec["status"], "url": dec["url"]}})
        st.json({"dg_skill": {"ok": skl["ok"], "status": skl["status"], "url": skl["url"]}})

    if not pre["ok"] or not isinstance(pre["payload"], (dict, list)):
        return pd.DataFrame(), {"error": "DataGolf pre-tournament API failed", "payload": pre["payload"], "status": pre["status"]}

    rows = find_rows_with_player_name(pre["payload"])
    if rows is None or len(rows) == 0:
        # show helpful metadata if present
        meta = {}
        if isinstance(pre["payload"], dict):
            for k in ["event_name", "last_updated", "models_available"]:
                if k in pre["payload"]:
                    meta[k] = pre["payload"][k]
        return pd.DataFrame(), {"error": "No PGA prediction rows returned from DataGolf.", "meta": meta}

    df = pd.DataFrame(rows).copy()
    # player name
    if "player_name" in df.columns:
        df["Player"] = df["player_name"].astype(str)
    elif "player" in df.columns:
        df["Player"] = df["player"].astype(str)
    else:
        return pd.DataFrame(), {"error": "DataGolf rows missing player field."}

    # model availability
    models_available = []
    event_name = None
    if isinstance(pre["payload"], dict):
        models_available = pre["payload"].get("models_available", []) or []
        event_name = pre["payload"].get("event_name")

    # choose model
    model_col = "baseline_history_fit" if "baseline_history_fit" in models_available else "baseline" if "baseline" in models_available else None
    if model_col is None:
        # fallback: pick first column that looks like win probability
        candidates = [c for c in df.columns if "baseline" in c]
        model_col = candidates[0] if candidates else None

    # Identify win/top10 columns
    # DataGolf commonly provides columns like: 'win', 'top_10' OR model-specific keys
    win_col = None
    top10_col = None

    # 1) try common
    for c in ["win", "win_pct", "win_prob", "win_probability"]:
        if c in df.columns:
            win_col = c
            break
    for c in ["top_10", "top10", "top_10_pct", "top10_pct"]:
        if c in df.columns:
            top10_col = c
            break

    # 2) if not found, try model-based pattern
    if win_col is None and model_col and model_col in df.columns:
        win_col = model_col  # model column is usually win prob

    # Convert to probability 0..1
    df["WinProb"] = df[win_col].apply(to_prob01_from_percent) if win_col in df.columns else np.nan
    if top10_col and top10_col in df.columns:
        df["Top10Prob"] = df[top10_col].apply(to_prob01_from_percent)
    else:
        df["Top10Prob"] = np.nan

    df["WinProb"] = clamp01(pd.to_numeric(df["WinProb"], errors="coerce").fillna(np.nan))
    df["Top10Prob"] = clamp01(pd.to_numeric(df["Top10Prob"], errors="coerce").fillna(np.nan))

    # Merge decompositions for strokes gained components / fit-ish signals
    decomp_rows = find_rows_with_player_name(dec["payload"]) if dec["ok"] else None
    if decomp_rows:
        dfd = pd.DataFrame(decomp_rows).copy()
        if "player_name" in dfd.columns:
            dfd["Player"] = dfd["player_name"].astype(str)
        elif "player" in dfd.columns:
            dfd["Player"] = dfd["player"].astype(str)
        keep = ["Player"] + [c for c in dfd.columns if any(k in c.lower() for k in ["sg_", "t2g", "putt", "app", "arg", "ott", "bogey", "bd", "history", "fit", "form"])]
        dfd = dfd[keep].drop_duplicates("Player")
        df = df.merge(dfd, on="Player", how="left")

    # Merge skill ratings (longer-term)
    skill_rows = find_rows_with_player_name(skl["payload"]) if skl["ok"] else None
    if skill_rows:
        dfs = pd.DataFrame(skill_rows).copy()
        if "player_name" in dfs.columns:
            dfs["Player"] = dfs["player_name"].astype(str)
        elif "player" in dfs.columns:
            dfs["Player"] = dfs["player"].astype(str)
        keep = ["Player"] + [c for c in dfs.columns if any(k in c.lower() for k in ["sg_", "t2g", "putt", "bogey", "form", "skill", "rating"])]
        dfs = dfs[keep].drop_duplicates("Player")
        df = df.merge(dfs, on="Player", how="left")

    # Build a composite “PGA Score” emphasizing:
    # course fit/history + current form + SG:T2G + putting + bogey avoidance
    # We’ll use available columns if present.
    cand_cols = {
        "sg_t2g": [c for c in df.columns if "t2g" in c.lower()],
        "sg_putt": [c for c in df.columns if "putt" in c.lower()],
        "bogey": [c for c in df.columns if "bogey" in c.lower()],
        "history_fit": [c for c in df.columns if ("history" in c.lower() or "fit" in c.lower())],
        "form": [c for c in df.columns if "form" in c.lower()],
    }

    def pick_first(cols):
        return cols[0] if cols else None

    col_t2g = pick_first(cand_cols["sg_t2g"])
    col_putt = pick_first(cand_cols["sg_putt"])
    col_bogey = pick_first(cand_cols["bogey"])
    col_histfit = pick_first(cand_cols["history_fit"])
    col_form = pick_first(cand_cols["form"])

    score = 0.0
    # weights (reasonable defaults)
    if col_histfit:
        score += 0.28 * zscore(df[col_histfit])
    if col_form:
        score += 0.24 * zscore(df[col_form])
    if col_t2g:
        score += 0.26 * zscore(df[col_t2g])
    if col_putt:
        score += 0.14 * zscore(df[col_putt])
    if col_bogey:
        # lower bogeys = better, so invert
        score += 0.08 * (-zscore(df[col_bogey]))

    df["ModelScore"] = pd.to_numeric(score, errors="coerce")

    # Use WinProb as the primary “Actual probability estimate”
    # and score as a tie-break / refinement
    df["RankKeyWin"] = (df["WinProb"].fillna(0.0) * 1000.0) + df["ModelScore"].fillna(0.0)
    df["RankKeyTop10"] = (df["Top10Prob"].fillna(0.0) * 1000.0) + df["ModelScore"].fillna(0.0)

    # Return two boards
    out_win = df[["Player", "WinProb", "ModelScore", "RankKeyWin"]].copy()
    out_win = out_win.sort_values("RankKeyWin", ascending=False).head(10)
    out_win["WinProb%"] = pct01_to_100(out_win["WinProb"])

    out_t10 = df[["Player", "Top10Prob", "ModelScore", "RankKeyTop10"]].copy()
    out_t10 = out_t10.sort_values("RankKeyTop10", ascending=False).head(10)
    out_t10["Top10Prob%"] = pct01_to_100(out_t10["Top10Prob"])

    meta = {"event_name": event_name, "model_used": model_col, "models_available": models_available}
    return (out_win, out_t10, meta), {}

# =========================================================
# MAIN
# =========================================================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if compact:
        sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport_m")
        bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_type_m")
        top_n = st.slider("Top picks (by EDGE)", 2, 10, 5, key="gl_top_m")
        show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_snap_m")
    else:
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        sport = c1.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
        bet_type = c2.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_type")
        top_n = c3.slider("Top picks (EDGE)", 2, 10, 5, key="gl_top")
        show_top25 = c4.toggle("Show top 25", value=True, key="gl_snap")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No +EV game line bets right now."))
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD) — +EV ONLY")
    st.caption("Sorted by **EDGE**. BestBook highlighted. Contradictions removed (half-point bucketed).")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Odds)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (+EV only)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Independent dropdowns (separate API call path from Game Lines)
    if compact:
        sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport_m")
        prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_type_m")
        max_events_scan = st.slider("Events to scan (usage control)", 1, 12, 6, key="pp_scan_m")
        top_n = st.slider("Top picks (EDGE)", 2, 10, 5, key="pp_top_m")
        show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_snap_m")
    else:
        c1, c2, c3, c4, c5 = st.columns([1, 1.2, 1, 1, 1])
        sport = c1.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
        prop_label = c2.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_type")
        max_events_scan = c3.slider("Events to scan", 1, 12, 6, key="pp_scan")
        top_n = c4.slider("Top picks (EDGE)", 2, 10, 5, key="pp_top")
        show_top25 = c5.toggle("Show top 25", value=True, key="pp_snap")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No +EV props returned for DK/FD on scanned events."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) — +EV ONLY")
    st.caption("Sorted by **EDGE**. BestBook highlighted. Contradictions removed (half-point bucketed).")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in top.columns and top["LineBucket"].notna().any() else []) + \
           ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Odds)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (+EV only)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # PGA mode
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")
    st.caption("Top 10 projected winners + Top 10 projected Top-10 finishes using DataGolf + skill/decomposition signals.")

    boards, err = build_pga_board()
    if err:
        st.warning(err.get("error", "PGA module error."))
        if debug:
            st.json(err)
        st.stop()

    (win_board, t10_board, meta), _ = boards, err

    if debug and meta:
        st.json(meta)

    if meta and meta.get("event_name"):
        st.markdown(f"**Event:** {meta['event_name']}")

    st.markdown("### 🏆 Top 10 Win Candidates")
    st.dataframe(
        win_board[["Player", "WinProb%", "ModelScore"]].rename(columns={"ModelScore": "Fit/Form Score"}),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 🔟 Top 10 Top-10 Candidates")
    if "Top10Prob%" in t10_board.columns and t10_board["Top10Prob%"].notna().any():
        st.dataframe(
            t10_board[["Player", "Top10Prob%", "ModelScore"]].rename(columns={"ModelScore": "Fit/Form Score"}),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("DataGolf did not include Top-10 probabilities for this event in the returned payload.")

    # simple percent charts
    w = win_board.copy()
    w["Label"] = w["Player"].astype(str)
    bar_prob(w, "Label", "WinProb%", "Win Probability (Top 10)")

    t = t10_board.copy()
    if "Top10Prob%" in t.columns and t["Top10Prob%"].notna().any():
        t["Label"] = t["Player"].astype(str)
        bar_prob(t, "Label", "Top10Prob%", "Top-10 Probability (Top 10)")

    st.markdown("</div>", unsafe_allow_html=True)
