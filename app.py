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
# Supports: DATAGOLF_KEY or DATAGOLF_API_KEY
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
# Odds math (ROBUST for American OR Decimal inputs)
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
    try:
        p = float(price)
    except Exception:
        return np.nan
    if 1.01 <= p <= 200.0:
        return decimal_to_implied(p)
    return american_to_implied(p)

def decimal_to_american(dec) -> float:
    d = float(dec)
    if d >= 2.0:
        return round((d - 1.0) * 100.0, 0)
    return round(-100.0 / (d - 1.0), 0)

def price_to_american_display(price):
    try:
        p = float(price)
    except Exception:
        return ""
    if 1.01 <= p <= 200.0:
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

# EXACT keys you requested
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
    "Best bets = **Edge = YourProb − ImpliedProb(best odds)**. "
    "Ranked by **Edge**. "
    "**Contradictions removed** (including across DK vs FD) using half-point buckets + one pick per bucket."
)

if not ODDS_API_KEY.strip():
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar.')
    st.stop()

# =========================================================
# Cache (daily)
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

# ------------------ DataGolf cache (daily) ------------------
DG_HOST = "https://feeds.datagolf.com"

@st.cache_data(ttl=60 * 60 * 24)
def dg_pre_tournament(tour="pga", add_position=10):
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": tour,
        "add_position": add_position,
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def dg_decompositions(tour="pga"):
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": tour, "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def dg_skill_ratings(tour="pga"):
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

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
                        "OddsDisplay": price_to_american_display(price),
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
        book = bm.get("key") or bm.get("title")
        for mk in (bm.get("markets", []) or []):
            mkey = mk.get("key")
            for out in (mk.get("outcomes", []) or []):
                player = out.get("name")
                side = (out.get("description") or out.get("label") or "").strip()
                line = out.get("point")
                price = out.get("price")

                if player is None or price is None:
                    continue

                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player),
                    "Side": side,
                    "Line": line,
                    "LineBucket": line_bucket_half_point(line) if line is not None else np.nan,
                    "Price": price,
                    "Book": book,
                    "OddsDisplay": price_to_american_display(price),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Player", "Price"])

# =========================================================
# Core Best-Bet Logic
# =========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(price_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    if "OddsDisplay" not in out.columns:
        out["OddsDisplay"] = out["Price"].apply(price_to_american_display)
    return out

def is_two_way_over_under(df: pd.DataFrame) -> bool:
    if "Side" not in df.columns or df["Side"].isna().all():
        return False
    sides = set(s.strip().lower() for s in df["Side"].dropna().unique())
    return ("over" in sides) or ("under" in sides)

def estimate_your_prob_lines(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = add_implied(df)
    sums = out.groupby(book_cols)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)

    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")
    out["YourProb"] = np.where(pd.notna(nv_avg), nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    return out

def estimate_your_prob_props(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = add_implied(df)
    out["YourProb"] = out.groupby(key_cols)["Implied"].transform("mean")

    if "Side" in out.columns and is_two_way_over_under(out):
        pair_cols = ["Event", "Market", "Player", "Book"]
        if "LineBucket" in out.columns and out["LineBucket"].notna().any():
            pair_cols.insert(3, "LineBucket")

        mask_ou = out["Side"].str.lower().isin(["over", "under"])
        tmp = out.loc[mask_ou].copy()
        sums = tmp.groupby(pair_cols)["Implied"].transform("sum")
        tmp["NoVigProb"] = np.where(sums > 0, tmp["Implied"] / sums, np.nan)
        nv_avg = tmp.groupby(key_cols)["NoVigProb"].transform("mean")
        tmp["YourProb"] = np.where(pd.notna(nv_avg), nv_avg, tmp.groupby(key_cols)["Implied"].transform("mean"))
        out.loc[mask_ou, "YourProb"] = tmp["YourProb"].values

    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    return out

def best_price_and_edge(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
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
    best["EV"] = best["Edge"] * 100.0
    return best

def prevent_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    return out.loc[idx].sort_values("Edge", ascending=False)

def keep_value_or_all(df_best: pd.DataFrame, only_value: bool) -> pd.DataFrame:
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    if only_value:
        out = out[out["Edge"] > 0]
    return out.sort_values("Edge", ascending=False)

# =========================================================
# Boards
# =========================================================
def build_game_lines_board(sport: str, bet_type: str, only_value: bool = True):
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

    df = estimate_your_prob_lines(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_value_or_all(df_best, only_value=only_value)

    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)
    return df_best, {}

def build_props_board(sport: str, prop_label: str, max_events_scan: int = 6, only_value: bool = True):
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
        return pd.DataFrame(), {"error": "No prop rows normalized (market may not be posted yet on DK/FD).", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    key_cols = ["Event", "Market", "Player", "Side"]
    best_cols = ["Event", "Market", "Player", "Side"]

    has_line_bucket = "LineBucket" in df.columns and df["LineBucket"].notna().any()
    if has_line_bucket:
        key_cols += ["LineBucket"]
        best_cols += ["LineBucket"]
        contradiction_cols = ["Event", "Market", "Player", "LineBucket"]
    else:
        contradiction_cols = ["Event", "Market", "Player"]

    df = estimate_your_prob_props(df, key_cols=key_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_value_or_all(df_best, only_value=only_value)

    df_best["YourProb%"] = pct01_to_100(df_best["YourProb"])
    df_best["Implied%"] = pct01_to_100(df_best["ImpliedBest"])
    df_best["Edge%"] = pct01_to_100(df_best["Edge"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)
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
# PGA Helpers (RESTORED module)
# =========================================================
def _dg_extract_rows(payload):
    """
    DataGolf payload shapes vary.
    Try common keys, else list root.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ["data", "predictions", "players", "results"]:
            if k in payload and isinstance(payload[k], list):
                return payload[k]
    return []

def _coerce_percent_to_prob(x):
    # DataGolf with odds_format=percent returns 0-100 percents in many fields
    try:
        v = float(x)
    except Exception:
        return np.nan
    if v > 1.0:
        return v / 100.0
    return v

def _find_first(d: dict, keys: list):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None

def normalize_pga(pre_payload, decomp_payload=None, skill_payload=None):
    pre_rows = _dg_extract_rows(pre_payload)
    if not pre_rows:
        return pd.DataFrame()

    # Build base table from pre-tournament preds
    base = []
    for r in pre_rows:
        if not isinstance(r, dict):
            continue
        name = _find_first(r, ["player_name", "name", "player", "dg_name"])
        if not name:
            continue

        # Win probability keys vary
        win_raw = _find_first(r, ["win", "win_prob", "win_probability", "p_win", "prob_win", "win_pct"])
        top10_raw = _find_first(r, ["top10", "top_10", "top10_prob", "prob_top10", "p_top10", "top10_pct"])

        win_p = _coerce_percent_to_prob(win_raw) if win_raw is not None else np.nan
        t10_p = _coerce_percent_to_prob(top10_raw) if top10_raw is not None else np.nan

        # Try to find sportsbook odds fields if present (varies by feed/event)
        # We'll search any key containing "odds" & "win" etc.
        dk_win = None
        fd_win = None
        for k, v in r.items():
            lk = str(k).lower()
            if "odds" in lk and ("dk" in lk or "draft" in lk) and ("win" in lk or "to_win" in lk or "winner" in lk):
                dk_win = v
            if "odds" in lk and ("fd" in lk or "fanduel" in lk) and ("win" in lk or "to_win" in lk or "winner" in lk):
                fd_win = v

        base.append({
            "Player": str(name),
            "WinProb": win_p,
            "Top10Prob": t10_p,
            "DK_WinOdds": dk_win,
            "FD_WinOdds": fd_win,
        })

    df = pd.DataFrame(base)
    if df.empty:
        return df

    # Add decomposition / skill if available (used for "analytics" columns + score)
    # These feeds can be huge; we merge by player name (best-effort).
    def_rows = _dg_extract_rows(decomp_payload) if decomp_payload is not None else []
    skill_rows = _dg_extract_rows(skill_payload) if skill_payload is not None else []

    if def_rows:
        ddf = pd.DataFrame([r for r in def_rows if isinstance(r, dict)])
        # try common naming columns
        name_col = None
        for c in ["player_name", "name", "player", "dg_name"]:
            if c in ddf.columns:
                name_col = c
                break
        if name_col:
            # pick useful columns if present
            keep = [name_col]
            wanted = [
                "sg_t2g", "sg_putt", "sg_app", "sg_ott", "sg_arg",
                "course_fit", "course_history", "recent_form", "bogey_avoidance"
            ]
            for w in wanted:
                # allow partial matches
                for c in ddf.columns:
                    if str(c).lower() == w or w in str(c).lower():
                        keep.append(c)
                        break
            ddf = ddf[keep].copy()
            ddf = ddf.rename(columns={name_col: "Player"})
            df = df.merge(ddf, on="Player", how="left")

    if skill_rows:
        sdf = pd.DataFrame([r for r in skill_rows if isinstance(r, dict)])
        name_col = None
        for c in ["player_name", "name", "player", "dg_name"]:
            if c in sdf.columns:
                name_col = c
                break
        if name_col:
            # try common "rating" or "skill" fields
            keep = [name_col]
            for c in sdf.columns:
                lc = str(c).lower()
                if lc in ["rating", "skill", "skill_rating", "baseline", "value"] or "skill" in lc or "rating" in lc:
                    keep.append(c)
            sdf = sdf[list(dict.fromkeys(keep))].copy()
            sdf = sdf.rename(columns={name_col: "Player"})
            df = df.merge(sdf, on="Player", how="left")

    # Create an "AnalyticsScore" purely to enrich ranking when edges tie / missing odds.
    # Uses available fields without ever breaking if missing.
    def safe_num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series([np.nan] * len(df))

    sg_t2g = safe_num("sg_t2g")
    sg_putt = safe_num("sg_putt")
    bogey = safe_num("bogey_avoidance")

    # Normalize to z-scores (best-effort)
    def z(x):
        x = pd.to_numeric(x, errors="coerce")
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series([0.0] * len(x))
        return (x - mu) / sd

    df["Score_T2G"] = z(sg_t2g)
    df["Score_Putt"] = z(sg_putt)
    df["Score_Bogey"] = z(bogey)

    df["AnalyticsScore"] = (
        0.55 * df["Score_T2G"].fillna(0) +
        0.25 * df["Score_Putt"].fillna(0) +
        0.20 * df["Score_Bogey"].fillna(0)
    )

    # Compute best available win odds implied if odds fields exist
    df["DK_WinImplied"] = df["DK_WinOdds"].apply(price_to_implied)
    df["FD_WinImplied"] = df["FD_WinOdds"].apply(price_to_implied)
    df["BestWinImplied"] = df[["DK_WinImplied", "FD_WinImplied"]].min(axis=1, skipna=True)
    df["BestWinBook"] = np.where(
        df["DK_WinImplied"].fillna(9e9) <= df["FD_WinImplied"].fillna(9e9),
        "draftkings", "fanduel"
    )
    df["BestWinOdds"] = np.where(
        df["BestWinBook"] == "draftkings",
        df["DK_WinOdds"], df["FD_WinOdds"]
    )

    # Edge for win if book odds exist; else Edge is NaN (we'll rank by prob+analytics)
    df["WinImplied"] = clamp01(pd.to_numeric(df["BestWinImplied"], errors="coerce").fillna(np.nan))
    df["WinEdge"] = pd.to_numeric(df["WinProb"], errors="coerce") - df["WinImplied"]

    # Similar for Top10: we usually don’t have odds in these feeds, so rank by prob+analytics
    # If you later add a Top10 odds feed, we’ll compute implied/edge the same way.

    df["WinProb%"] = pct01_to_100(df["WinProb"])
    df["Top10Prob%"] = pct01_to_100(df["Top10Prob"])
    df["WinEdge%"] = pct01_to_100(df["WinEdge"])
    df["BestWinOddsDisp"] = df["BestWinOdds"].apply(price_to_american_display)

    return df

# =========================================================
# MAIN UI
# =========================================================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if compact:
        sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport_m")
        bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_type_m")
        only_value = st.toggle("Show +EV only", value=True, key="gl_only_ev_m")
        top_n = st.slider("Top picks (by EDGE)", 2, 15, 5, key="gl_top_m")
        show_top25 = st.toggle("Show top 25 snapshot", value=True, key="gl_snap_m")
    else:
        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
        sport = c1.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
        bet_type = c2.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_type")
        only_value = c3.toggle("Show +EV only", value=True, key="gl_only_ev")
        top_n = c4.slider("Top picks (EDGE)", 2, 15, 5, key="gl_top")
        show_top25 = c5.toggle("Show top 25", value=True, key="gl_snap")

    df_best, err = build_game_lines_board(sport, bet_type, only_value=only_value)
    if df_best.empty:
        st.warning(err.get("error", "No game line rows available for these settings."))
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD)")
    st.caption("Sorted by **EDGE**. BestBook shown. Contradictions removed (half-point bucketed).")

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
        st.markdown("### Snapshot — Top 25")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Outcome"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    if compact:
        sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport_m")
        prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_type_m")
        max_events_scan = st.slider("Events to scan (usage control)", 1, 20, 8, key="pp_scan_m")
        only_value = st.toggle("Show +EV only", value=True, key="pp_only_ev_m")
        top_n = st.slider("Top picks (EDGE)", 2, 15, 5, key="pp_top_m")
        show_top25 = st.toggle("Show top 25 snapshot", value=True, key="pp_snap_m")
    else:
        c1, c2, c3, c4, c5, c6 = st.columns([1, 1.2, 1, 1, 1, 1])
        sport = c1.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pp_sport")
        prop_label = c2.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pp_type")
        max_events_scan = c3.slider("Events to scan", 1, 20, 8, key="pp_scan")
        only_value = c4.toggle("Show +EV only", value=True, key="pp_only_ev")
        top_n = c5.slider("Top picks (EDGE)", 2, 15, 5, key="pp_top")
        show_top25 = c6.toggle("Show top 25", value=True, key="pp_snap")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan, only_value=only_value)
    if df_best.empty:
        st.warning(err.get("error", "No prop rows available for these settings."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) (DK/FD)")
    st.caption("Sorted by **EDGE**. BestBook shown. Contradictions removed (half-point bucketed).")

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
        st.markdown("### Snapshot — Top 25")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side"] + (["LineBucket"] if "LineBucket" in snap.columns and snap["LineBucket"].notna().any() else []) + \
                ["BestOdds", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # =========================================================
    # PGA (RESTORED RESULTS MODULE)
    # =========================================================
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA — Course Fit + Course History + Current Form (DataGolf)")

    if not DATAGOLF_API_KEY.strip():
        st.warning('DATAGOLF key not set. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY="...").')
        st.stop()

    # Controls
    if compact:
        only_value = st.toggle("Show +EV only (requires book odds)", value=False, key="pga_only_ev_m")
        top_n = st.slider("Top picks", 5, 20, 10, key="pga_top_m")
    else:
        c1, c2 = st.columns([1, 1])
        only_value = c1.toggle("Show +EV only (requires book odds)", value=False, key="pga_only_ev")
        top_n = c2.slider("Top picks", 5, 20, 10, key="pga_top")

    pre = dg_pre_tournament(tour="pga", add_position=10)
    decomp = dg_decompositions(tour="pga")
    skill = dg_skill_ratings(tour="pga")

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": pre["ok"], "status": pre["status"], "url": pre["url"]},
            "dg_decomp": {"ok": decomp["ok"], "status": decomp["status"], "url": decomp["url"]},
            "dg_skill": {"ok": skill["ok"], "status": skill["status"], "url": skill["url"]},
        })
        # Also show meta if present
        if isinstance(pre["payload"], dict):
            meta_keys = {k: pre["payload"].get(k) for k in ["event_name", "last_updated", "models_available"] if k in pre["payload"]}
            if meta_keys:
                st.json({"dg_meta": meta_keys})

    if not pre["ok"]:
        st.error("DataGolf pre-tournament feed failed.")
        st.stop()

    dfpga = normalize_pga(pre["payload"], decomp_payload=(decomp["payload"] if decomp["ok"] else None),
                          skill_payload=(skill["payload"] if skill["ok"] else None))
    if dfpga.empty:
        st.error("No PGA prediction rows returned from DataGolf (parsed none).")
        st.stop()

    # WIN board
    win = dfpga.copy()
    # If we have implied odds -> Edge exists; else sort by WinProb + AnalyticsScore
    win["SortKey"] = np.where(win["WinEdge"].notna(), win["WinEdge"], win["WinProb"].fillna(0) + 0.02 * win["AnalyticsScore"].fillna(0))
    win = win.sort_values("SortKey", ascending=False)

    if only_value:
        win = win[win["WinEdge"].fillna(-1) > 0]

    win_top = win.head(int(top_n)).copy()
    win_cols = ["Player", "WinProb%", "BestWinOddsDisp", "BestWinBook", "WinEdge%"]
    for extra in ["sg_t2g", "sg_putt", "bogey_avoidance", "AnalyticsScore"]:
        if extra in win_top.columns:
            win_cols.append(extra)
    win_cols = [c for c in win_cols if c in win_top.columns]

    st.markdown("### Top Win Candidates")
    st.dataframe(win_top[win_cols], use_container_width=True, hide_index=True)

    # TOP 10 board
    t10 = dfpga.copy()
    t10["SortKey"] = t10["Top10Prob"].fillna(0) + 0.03 * t10["AnalyticsScore"].fillna(0)
    t10 = t10.sort_values("SortKey", ascending=False)
    t10_top = t10.head(int(top_n)).copy()

    t10_cols = ["Player", "Top10Prob%"]
    for extra in ["sg_t2g", "sg_putt", "bogey_avoidance", "AnalyticsScore"]:
        if extra in t10_top.columns:
            t10_cols.append(extra)
    t10_cols = [c for c in t10_cols if c in t10_top.columns]

    st.markdown("### Top-10 Candidates")
    st.dataframe(t10_top[t10_cols], use_container_width=True, hide_index=True)

    # Charts
    st.markdown("#### Probability view (Top picks)")
    cwin = win_top.copy()
    cwin["Label"] = cwin["Player"].astype(str)
    cwin["WinProbPct"] = cwin["WinProb%"]
    bar_prob(cwin, "Label", "WinProbPct", "Win Probability (Top Picks)")

    ct10 = t10_top.copy()
    ct10["Label"] = ct10["Player"].astype(str)
    ct10["Top10ProbPct"] = ct10["Top10Prob%"]
    bar_prob(ct10, "Label", "Top10ProbPct", "Top-10 Probability (Top Picks)")

    st.markdown("</div>", unsafe_allow_html=True)
