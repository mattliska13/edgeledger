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

# Support either DATAGOLF_API_KEY or DATAGOLF_KEY (your Secrets uses DATAGOLF_KEY)
DATAGOLF_API_KEY = get_key("DATAGOLF_API_KEY", "")
if not DATAGOLF_API_KEY:
    DATAGOLF_API_KEY = get_key("DATAGOLF_KEY", "")

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

# The Odds API prop keys that you requested
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
compact = st.sidebar.toggle("Mobile / Compact layout", value=False)
show_non_value = st.sidebar.checkbox("Show non-value rows (Edge ≤ 0)", value=False)

modes = ["Game Lines", "Player Props", "PGA"]
mode = st.sidebar.radio("Mode", modes, index=0)

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
    "Best bets are ranked by **Edge = YourProb − ImpliedProb(best price)**. "
    "Contradictions are removed using half-point bucketing (so you won’t get Over + Under / both sides). "
    "DK/FD only. Game lines / props / PGA run independently."
)

if not ODDS_API_KEY.strip():
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
                side = out.get("description")  # Over/Under usually; sometimes blank
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
    - TWO-WAY markets: use no-vig within each book, then average across books
    - ONE-WAY markets (e.g., Anytime TD often only "Yes"): use market-consensus avg implied across books
      (this is what creates value via shopping: consensus vs best price implied)
    """
    if df.empty:
        return df.copy()

    out = add_implied(df)

    # Determine if this key group is "two-way" or "one-way"
    # We use count of distinct "Outcome/Side" within the group (per event/market/linebucket/player, etc.)
    # If <=1 -> treat as one-way.
    side_col = "Outcome" if "Outcome" in out.columns else "Side"
    grp = out.groupby(key_cols)[side_col].transform(lambda s: s.nunique(dropna=True))
    out["_n_sides"] = pd.to_numeric(grp, errors="coerce").fillna(1)

    # For two-way groups: compute no-vig within book and average
    out = compute_no_vig_within_book_two_way(out, group_cols_book=book_cols)
    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")

    out["YourProb"] = np.where(out["_n_sides"] >= 2, nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    out = out.drop(columns=["_n_sides"], errors="ignore")
    return out

def best_price_and_edge(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
    """
    For each unique bet group:
    - pick BEST PRICE across books
    - compute implied from that best price
    - Edge = YourProb - ImpliedBest
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
    best["EV"] = (best["Edge"] * 100.0)  # edge in percentage points
    return best

def prevent_contradictions(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    Keep only ONE pick per contradiction group (max Edge).
    This blocks Over+Under, both spread sides, etc. even if off by half point (LineBucket).
    """
    if df_best.empty:
        return df_best

    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    out = out.loc[idx].sort_values("Edge", ascending=False)
    return out

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
        contradiction_cols = ["Event", "Market", "LineBucket"]
    else:
        contradiction_cols = ["Event", "Market"]

    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)

    # Rank by EDGE (not YourProb%)
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
        time.sleep(0.05)

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
        # Anytime TD etc.
        contradiction_cols = ["Event", "Market", "Player"]

    # IMPORTANT:
    # - Anytime TD is often one-way; our YourProb logic will switch to market-consensus avg implied automatically.
    df = estimate_your_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_edge(df, group_cols_best=best_cols)
    df_best = prevent_contradictions(df_best, contradiction_cols=contradiction_cols)
    df_best = filter_value(df_best, show_non_value=show_non_value)
    df_best = decorate_probs(df_best)

    # Rank by EDGE
    df_best = df_best.sort_values("Edge", ascending=False)
    return df_best, {}

# =========================================================
# Charts (percent axis)
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
# DataGolf (PGA) — robust parsing + OAD
# =========================================================
DG_HOST = "https://feeds.datagolf.com"

def _find_first_list_of_dicts(payload, required_keys=None):
    """
    Recursively search a JSON-like structure for a list[dict] that contains required keys.
    """
    if required_keys is None:
        required_keys = []

    def ok_list(lst):
        if not (isinstance(lst, list) and lst and isinstance(lst[0], dict)):
            return False
        if not required_keys:
            return True
        keys = set(lst[0].keys())
        return all(k in keys for k in required_keys)

    if ok_list(payload):
        return payload

    if isinstance(payload, dict):
        for _, v in payload.items():
            found = _find_first_list_of_dicts(v, required_keys=required_keys)
            if found is not None:
                return found

    if isinstance(payload, list):
        for v in payload:
            found = _find_first_list_of_dicts(v, required_keys=required_keys)
            if found is not None:
                return found

    return None

@st.cache_data(ttl=60 * 60 * 12)
def datagolf_pre_tournament():
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": "pga",
        "add_position": "10",
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def datagolf_decomp():
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": "pga", "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def datagolf_skill():
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def build_pga_board():
    if not DATAGOLF_API_KEY.strip():
        return pd.DataFrame(), {"error": 'Missing DATAGOLF_KEY. Add it in Streamlit Secrets as DATAGOLF_KEY="..." (or DATAGOLF_API_KEY).'}

    res_pre = datagolf_pre_tournament()
    res_dec = datagolf_decomp()
    res_skl = datagolf_skill()

    if debug:
        st.json({
            "dg_pre_tournament": {"ok": res_pre["ok"], "status": res_pre["status"], "url": res_pre["url"]},
            "dg_decomp": {"ok": res_dec["ok"], "status": res_dec["status"], "url": res_dec["url"]},
            "dg_skill": {"ok": res_skl["ok"], "status": res_skl["status"], "url": res_skl["url"]},
        })

    if not res_pre["ok"]:
        return pd.DataFrame(), {"error": "DataGolf pre-tournament failed", "payload": res_pre["payload"], "status": res_pre["status"]}

    # Pre-tournament is percent odds_format, so win/top10 often appear as numeric percent.
    # Find a list of dicts containing at least player name
    rows = _find_first_list_of_dicts(res_pre["payload"], required_keys=["player_name"]) \
        or _find_first_list_of_dicts(res_pre["payload"], required_keys=["player"]) \
        or _find_first_list_of_dicts(res_pre["payload"], required_keys=[])

    if rows is None or not rows:
        # keep meta debug if present
        meta = {}
        if isinstance(res_pre["payload"], dict):
            for k in ["event_name", "last_updated", "models_available"]:
                if k in res_pre["payload"]:
                    meta[k] = res_pre["payload"].get(k)
        return pd.DataFrame(), {"error": "No PGA prediction rows returned from DataGolf (parsed none).", "dg_meta": meta}

    df = pd.DataFrame(rows).copy()

    # Normalize name column
    if "player_name" in df.columns:
        df["Player"] = df["player_name"].astype(str)
    elif "player" in df.columns:
        df["Player"] = df["player"].astype(str)
    else:
        # last resort
        df["Player"] = df.iloc[:, 0].astype(str)

    # Extract win/top10 probability fields (varies by schema)
    # We'll try common names; else fill NaN.
    win_candidates = ["win", "prob_win", "win_pct", "win_percent", "win_probability"]
    top10_candidates = ["top10", "prob_top10", "top_10", "top10_pct", "top10_percent", "top10_probability"]

    def first_col(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    win_col = first_col(win_candidates)
    top10_col = first_col(top10_candidates)

    # Many DG feeds label columns like "baseline_win", "baseline_history_fit_win"
    # So also scan for any column containing "win" and any containing "top10"
    if win_col is None:
        win_like = [c for c in df.columns if "win" in c.lower()]
        win_col = win_like[0] if win_like else None
    if top10_col is None:
        t10_like = [c for c in df.columns if "top10" in c.lower() or "top_10" in c.lower()]
        top10_col = t10_like[0] if t10_like else None

    df["WinProb"] = pd.to_numeric(df[win_col], errors="coerce") if win_col else np.nan
    df["Top10Prob"] = pd.to_numeric(df[top10_col], errors="coerce") if top10_col else np.nan

    # DataGolf returned percent; convert to 0-1
    # If values look like 0-100 -> divide by 100
    for col in ["WinProb", "Top10Prob"]:
        mx = pd.to_numeric(df[col], errors="coerce").max()
        if pd.notna(mx) and mx > 1.5:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df["WinProb"] = clamp01(df["WinProb"].fillna(0.001))
    df["Top10Prob"] = clamp01(df["Top10Prob"].fillna(0.001))

    # Pull decomposition / skill if available, but keep robust (optional)
    # We score course fit / history / form / T2G / putting / bogey avoidance if found.
    comp = pd.DataFrame()
    if res_dec["ok"]:
        dec_rows = _find_first_list_of_dicts(res_dec["payload"], required_keys=["player_name"]) \
            or _find_first_list_of_dicts(res_dec["payload"], required_keys=["player"])
        if dec_rows:
            comp = pd.DataFrame(dec_rows)

    skl = pd.DataFrame()
    if res_skl["ok"]:
        skl_rows = _find_first_list_of_dicts(res_skl["payload"], required_keys=["player_name"]) \
            or _find_first_list_of_dicts(res_skl["payload"], required_keys=["player"])
        if skl_rows:
            skl = pd.DataFrame(skl_rows)

    # Normalize name column for joins
    def norm_name(x):
        return str(x).strip().lower()

    df["name_key"] = df["Player"].map(norm_name)

    if not comp.empty:
        if "player_name" in comp.columns:
            comp["name_key"] = comp["player_name"].map(norm_name)
        elif "player" in comp.columns:
            comp["name_key"] = comp["player"].map(norm_name)

        # Attempt to locate useful columns
        # (field names can vary, so we scan by keywords)
        def find_kw(cols, kws):
            for c in cols:
                lc = c.lower()
                if all(k in lc for k in kws):
                    return c
            return None

        cols = list(comp.columns)
        # recent form
        c_form = find_kw(cols, ["recent"]) or find_kw(cols, ["form"])
        # course history / fit
        c_hist = find_kw(cols, ["history"]) or find_kw(cols, ["course"])
        c_fit = find_kw(cols, ["fit"]) or find_kw(cols, ["course", "fit"])
        # strokes gained components
        c_t2g = find_kw(cols, ["tee", "green"]) or find_kw(cols, ["t2g"])
        c_putt = find_kw(cols, ["putt"]) or find_kw(cols, ["putting"])
        c_bogey = find_kw(cols, ["bogey"])

        keep = {"name_key": "name_key"}
        if c_form: keep[c_form] = "RecentForm"
        if c_hist: keep[c_hist] = "CourseHistory"
        if c_fit: keep[c_fit] = "CourseFit"
        if c_t2g: keep[c_t2g] = "SG_T2G"
        if c_putt: keep[c_putt] = "SG_Putt"
        if c_bogey: keep[c_bogey] = "BogeyAvoid"

        comp2 = comp[list(keep.keys())].rename(columns=keep).copy()
        df = df.merge(comp2, on="name_key", how="left")

    # Build a simple composite OAD score (robustly handles missing columns)
    for c in ["RecentForm", "CourseHistory", "CourseFit", "SG_T2G", "SG_Putt", "BogeyAvoid"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize optional feature columns to 0-1 within field (if present)
    def norm01(s):
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() < 5:
            return pd.Series([np.nan] * len(s))
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx - mn == 0:
            return pd.Series([np.nan] * len(s))
        return (s - mn) / (mx - mn)

    feat_cols = ["RecentForm", "CourseHistory", "CourseFit", "SG_T2G", "SG_Putt", "BogeyAvoid"]
    for c in feat_cols:
        df[c + "_N"] = norm01(df[c])

    # Composite score (weights are reasonable defaults; you can tune later)
    # Uses WinProb/Top10Prob as anchor, then fit/form as tiebreakers.
    df["OADScore"] = (
        0.50 * df["Top10Prob"] +
        0.35 * df["WinProb"] +
        0.15 * df[["RecentForm_N", "CourseFit_N", "CourseHistory_N", "SG_T2G_N", "SG_Putt_N", "BogeyAvoid_N"]].mean(axis=1, skipna=True)
    )

    # Output board
    out = df[[
        "Player", "WinProb", "Top10Prob", "OADScore",
        "RecentForm", "CourseFit", "CourseHistory", "SG_T2G", "SG_Putt", "BogeyAvoid"
    ]].copy()

    out["Win%"] = pct01_to_100(out["WinProb"])
    out["Top10%"] = pct01_to_100(out["Top10Prob"])
    out = out.sort_values("OADScore", ascending=False)

    return out, {}

# =========================================================
# MAIN UI
# =========================================================
if mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0)
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1)
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No game lines rows available."))
        st.stop()

    st.subheader(f"{sport} — {bet_type} (DK/FD)")
    st.caption("Ranked by Edge. Contradictions removed (half-point bucketed).")

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

    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0)
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0)  # default Anytime TD
    top_n = st.slider("Top picks (ranked by EDGE)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)
    max_events_scan = st.slider("Events to scan (usage control)", 1, 12, 6)

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No props returned for DK/FD on scanned events."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label}) — DK/FD")
    st.caption(
        "Ranked by Edge. Contradictions removed. "
        "For one-way markets (like Anytime TD), YourProb uses consensus avg implied across books vs best price implied."
    )

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
    st.caption("Outputs: Top Winners, Top-10s, and One-and-Done options (ranked by a composite score).")

    df_pga, err = build_pga_board()
    if df_pga.empty:
        st.warning(err.get("error", "No PGA prediction rows returned from DataGolf."))
        if debug:
            st.json(err)
        st.stop()

    # Winners
    st.markdown("### Top 10 — Winners (model-based)")
    win_board = df_pga.sort_values("WinProb", ascending=False).head(10).copy()
    st.dataframe(
        win_board[["Player", "Win%", "Top10%", "OADScore"]],
        use_container_width=True,
        hide_index=True
    )

    # Top-10 bets
    st.markdown("### Top 10 — Top-10 Finish (model-based)")
    t10_board = df_pga.sort_values("Top10Prob", ascending=False).head(10).copy()
    st.dataframe(
        t10_board[["Player", "Top10%", "Win%", "OADScore"]],
        use_container_width=True,
        hide_index=True
    )

    # One & Done
    st.markdown("### One-and-Done — Best Options (composite)")
    st.caption("Score blends Top10%, Win%, and (when available) fit/history/form + SG components + bogey avoidance.")
    oad_board = df_pga.sort_values("OADScore", ascending=False).head(10).copy()
    cols = ["Player", "OADScore", "Top10%", "Win%", "RecentForm", "CourseFit", "CourseHistory", "SG_T2G", "SG_Putt", "BogeyAvoid"]
    cols = [c for c in cols if c in oad_board.columns]
    st.dataframe(oad_board[cols], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
