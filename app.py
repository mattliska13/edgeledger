import os
import time
import re
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
SESSION.headers.update({"User-Agent": "Dashboard/1.0 (streamlit)"})

def safe_get(url: str, params: dict | None = None, timeout: int = 25, headers: dict | None = None):
    try:
        if headers:
            r = SESSION.get(url, params=params, timeout=timeout, headers=headers)
        else:
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

# ✅ UFC integrated WITHOUT impacting other modules: new mode option only
mode = st.sidebar.radio("Mode", ["Game Lines", "Player Props", "PGA", "UFC Picks", "Tracker"], index=0)

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
    "DK/FD only. Game lines / props / PGA run independently. UFC runs independently."
)

# ✅ Do NOT require ODDS_API_KEY for UFC/Tracker
if not ODDS_API_KEY.strip() and mode not in ["Tracker", "UFC Picks"]:
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
# UFC Picks — pulls requested info automatically (no CSV)
# Data Source: ufcstats.com (HTML parse). Independent module.
# =========================================================

UFC_BASE = "http://ufcstats.com"
UFC_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) EdgeLedger/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def _sf(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        s = str(x).strip().replace("%", "")
        if s in ["", "--", "null", "None"]:
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _strip(s):
    return str(s).strip() if s is not None else ""

def _age_from_dob(dob_str: str) -> float:
    try:
        dt = pd.to_datetime(dob_str, errors="coerce")
        if pd.isna(dt):
            return np.nan
        today = pd.Timestamp.now().normalize()
        age = (today - dt).days / 365.25
        return float(np.floor(age * 10) / 10)
    except Exception:
        return np.nan

def _parse_first_href_map(html: str):
    """
    Returns a dict {anchor_text: href} for quick link lookup.
    Not perfect, but good enough for ufcstats pages where text is unique in rows.
    """
    hrefs = {}
    for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>', html, flags=re.I):
        href = m.group(1)
        txt = re.sub(r"\s+", " ", m.group(2)).strip()
        if txt and href:
            hrefs.setdefault(txt, href)
    return hrefs

@st.cache_data(ttl=60 * 60 * 6)
def ufc_fetch_html(url: str) -> str:
    ok, status, payload, _ = safe_get(url, params=None, timeout=30, headers=UFC_HEADERS)
    if not ok or not isinstance(payload, str):
        raise RuntimeError(f"UFCStats fetch failed ({status})")
    return payload

@st.cache_data(ttl=60 * 60 * 6)
def ufc_list_events(limit: int = 25) -> pd.DataFrame:
    html = ufc_fetch_html(f"{UFC_BASE}/statistics/events/completed?page=all")
    # There is a single table of events
    tables = pd.read_html(html)
    if not tables:
        return pd.DataFrame(columns=["Event", "Date", "EventURL"])
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Expected columns: 'EVENT', 'DATE', 'LOCATION'
    event_col = "EVENT" if "EVENT" in df.columns else df.columns[0]
    date_col = "DATE" if "DATE" in df.columns else df.columns[1]
    df = df[[event_col, date_col]].rename(columns={event_col: "Event", date_col: "Date"})
    # Extract event URLs from anchors
    href_map = _parse_first_href_map(html)
    # On this page, anchor text equals event name
    df["EventURL"] = df["Event"].map(lambda x: href_map.get(str(x).strip(), np.nan))
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["EventURL"]).sort_values("Date", ascending=False).head(int(limit))
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)

@st.cache_data(ttl=60 * 60 * 6)
def ufc_event_fights(event_url: str) -> pd.DataFrame:
    html = ufc_fetch_html(event_url)
    tables = pd.read_html(html)
    if not tables:
        return pd.DataFrame()
    df = tables[0].copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Typical columns include: 'FIGHT', 'WEIGHT_CLASS', 'METHOD', 'ROUND', 'TIME'
    # But read_html often splits; we'll keep what we can and enrich via href parsing.
    # Build href map for fighter + bout links
    # Bout link appears as anchor with text 'Details' not present; easier: regex for /fight-details/
    bout_links = re.findall(r'href="(http://ufcstats\.com/fight-details/[^"]+)"', html, flags=re.I)
    bout_links = list(dict.fromkeys(bout_links))  # unique preserve order
    # Fighter links appear as /fighter-details/
    fighter_links = re.findall(r'href="(http://ufcstats\.com/fighter-details/[^"]+)"', html, flags=re.I)

    # Extract fighters from visible table if possible
    # ufcstats event page table typically has columns: 'FIGHTER', 'FIGHTER.1' or 'FIGHTER' twice
    # We'll try to find two fighter name columns by heuristic.
    cols_lower = [c.lower() for c in df.columns]
    name_cols = []
    for i, c in enumerate(cols_lower):
        if "fighter" in c:
            name_cols.append(df.columns[i])
    # fallback: first two object columns
    if len(name_cols) < 2:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        name_cols = obj_cols[:2] if len(obj_cols) >= 2 else name_cols

    a_col = name_cols[0] if len(name_cols) >= 1 else None
    b_col = name_cols[1] if len(name_cols) >= 2 else None

    fights = []
    n = len(df)
    for i in range(n):
        fa = _strip(df.iloc[i][a_col]) if a_col else ""
        fb = _strip(df.iloc[i][b_col]) if b_col else ""
        wc = ""
        meth = ""
        rnd = ""
        tm = ""
        for c in df.columns:
            cl = str(c).lower()
            if "weight" in cl and wc == "":
                wc = _strip(df.iloc[i][c])
            if "method" in cl and meth == "":
                meth = _strip(df.iloc[i][c])
            if cl == "round" and rnd == "":
                rnd = _strip(df.iloc[i][c])
            if cl == "time" and tm == "":
                tm = _strip(df.iloc[i][c])

        bout = bout_links[i] if i < len(bout_links) else np.nan
        # fighter links come in pairs per row on page; best effort:
        f1 = fighter_links[2*i] if 2*i < len(fighter_links) else np.nan
        f2 = fighter_links[2*i + 1] if (2*i + 1) < len(fighter_links) else np.nan

        fights.append({
            "FighterA": fa,
            "FighterB": fb,
            "WeightClass": wc,
            "Method": meth,
            "Round": rnd,
            "Time": tm,
            "BoutURL": bout,
            "FighterA_URL": f1,
            "FighterB_URL": f2,
        })
    out = pd.DataFrame(fights)
    out = out.dropna(subset=["FighterA_URL", "FighterB_URL"], how="any")
    return out

@st.cache_data(ttl=60 * 60 * 24)
def ufc_fighter_profile(fighter_url: str) -> dict:
    html = ufc_fetch_html(fighter_url)

    # Parse key/value bio block using regex
    def pick(label):
        m = re.search(rf"{label}:</i>\s*([^<]+)</li>", html, flags=re.I)
        return m.group(1).strip() if m else ""

    height = pick("Height")
    weight = pick("Weight")
    reach = pick("Reach")
    stance = pick("STANCE")
    dob = pick("DOB")

    # Stats values in "b-list__box-list" in fixed order; regex labels:
    def stat(label):
        m = re.search(rf"<i[^>]*>{re.escape(label)}</i>\s*([^<]+)</li>", html, flags=re.I)
        return m.group(1).strip() if m else ""

    slpm = stat("SLpM:")
    sapm = stat("SApM:")
    str_acc = stat("Str. Acc.:")
    str_def = stat("Str. Def:")
    td_avg = stat("TD Avg.:")
    td_acc = stat("TD Acc.:")
    td_def = stat("TD Def.:")
    sub_avg = stat("Sub. Avg.:")

    # Record (wins/losses) appears near top e.g. "Record: 12-3-0"
    rec = ""
    mrec = re.search(r"Record:\s*</span>\s*<span[^>]*>\s*([^<]+)\s*</span>", html, flags=re.I)
    if mrec:
        rec = mrec.group(1).strip()
    wins = np.nan
    losses = np.nan
    if rec and "-" in rec:
        parts = rec.split("-")
        if len(parts) >= 2:
            wins = _sf(parts[0])
            losses = _sf(parts[1])

    # Bout history table for last five + ITD/DEC splits
    last5_w = 0
    last5_l = 0
    itd_wins = 0
    dec_wins = 0
    total_wins = 0

    tables = pd.read_html(html)
    hist = None
    for t in tables[::-1]:
        # history table usually has 'W/L' column
        cols = [str(c).lower() for c in t.columns]
        if any("w/l" in c for c in cols) and any("method" in c for c in cols):
            hist = t.copy()
            hist.columns = [str(c).strip() for c in hist.columns]
            break

    if hist is not None and not hist.empty:
        wl_col = None
        meth_col = None
        for c in hist.columns:
            if str(c).strip().lower() in ["w/l", "wl"]:
                wl_col = c
            if "method" in str(c).lower():
                meth_col = c
        if wl_col:
            recent = hist.head(5)
            for v in recent[wl_col].astype(str).values:
                vv = v.strip().upper()
                if vv == "W":
                    last5_w += 1
                elif vv == "L":
                    last5_l += 1

        # win method splits
        if wl_col and meth_col:
            for wl, method in zip(hist[wl_col].astype(str).values, hist[meth_col].astype(str).values):
                if wl.strip().upper() != "W":
                    continue
                total_wins += 1
                mm = method.upper()
                if "DEC" in mm:
                    dec_wins += 1
                if ("KO" in mm) or ("TKO" in mm) or ("SUB" in mm) or ("SUBMISSION" in mm):
                    itd_wins += 1

    last5 = f"{last5_w}-{last5_l}" if (last5_w + last5_l) > 0 else ""
    itd_pct = (100.0 * itd_wins / total_wins) if total_wins > 0 else np.nan
    dec_pct = (100.0 * dec_wins / total_wins) if total_wins > 0 else np.nan

    def to_in(reach_str):
        # reach like '72"' or '--'
        s = str(reach_str).replace('"', '').strip()
        return _sf(s)

    profile = {
        "Height": height,
        "Weight": weight,
        "Reach": to_in(reach),
        "Stance": stance.strip(),
        "DOB": dob.strip(),
        "Age": _age_from_dob(dob),
        "Wins": wins,
        "Losses": losses,
        "Last5": last5,
        "ITD%": itd_pct,
        "DEC%": dec_pct,
        "SLpM": _sf(slpm),
        "SApM": _sf(sapm),
        "StrAcc%": _sf(str_acc),
        "StrDef%": _sf(str_def),
        "TDAvg": _sf(td_avg),
        "TDAcc%": _sf(td_acc),
        "TDDef%": _sf(td_def),
        "SubAvg": _sf(sub_avg),
        "URL": fighter_url,
    }
    return profile

@st.cache_data(ttl=60 * 60 * 24)
def ufc_bout_totals(bout_url: str) -> dict:
    """
    Pulls per-fight totals for each fighter:
    - KD
    - TD (landed, attempted)
    - TD%
    - SUB
    - SIG STR landed/attempted
    - CTRL (mm:ss) to seconds
    """
    html = ufc_fetch_html(bout_url)
    tables = pd.read_html(html)
    # On fight-details pages, the first table is totals (two rows).
    # We'll attempt to locate a table with 'KD' and 'TD' columns.
    target = None
    for t in tables:
        cols = [str(c).upper() for c in t.columns]
        if "KD" in cols and any("TD" in c for c in cols):
            target = t.copy()
            break
    if target is None or target.empty:
        return {}

    target.columns = [str(c).strip() for c in target.columns]
    # Find fighter name column (often first)
    name_col = target.columns[0]

    def parse_landed_attempted(s):
        # strings like '3 of 8'
        try:
            parts = str(s).lower().split("of")
            if len(parts) == 2:
                landed = _sf(parts[0])
                att = _sf(parts[1])
                return landed, att
        except Exception:
            pass
        return (np.nan, np.nan)

    def ctrl_to_sec(s):
        try:
            s = str(s).strip()
            if ":" not in s:
                return np.nan
            mm, ss = s.split(":")
            return int(mm) * 60 + int(ss)
        except Exception:
            return np.nan

    rows = {}
    for _, r in target.iterrows():
        fighter = _strip(r.get(name_col, ""))
        kd = _sf(r.get("KD", np.nan))
        td_raw = r.get("TD", np.nan)
        td_l, td_a = parse_landed_attempted(td_raw)
        td_pct = _sf(r.get("TD%", np.nan))
        sub = _sf(r.get("SUB", np.nan))
        sig_raw = r.get("SIG. STR.", r.get("SIG STR", r.get("SIG STR.", np.nan)))
        sig_l, sig_a = parse_landed_attempted(sig_raw) if not pd.isna(sig_raw) else (np.nan, np.nan)
        ctrl = ctrl_to_sec(r.get("CTRL", np.nan))

        rows[fighter] = {
            "KD": kd,
            "TD_L": td_l,
            "TD_A": td_a,
            "TD%": td_pct,
            "SUB": sub,
            "SIG_L": sig_l,
            "SIG_A": sig_a,
            "CTRL_S": ctrl,
        }
    return rows

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _clip01(x):
    return float(np.clip(x, 0.001, 0.999))

def _stance_score(stance_a, stance_b):
    a = (stance_a or "").strip().lower()
    b = (stance_b or "").strip().lower()
    score = 0.0
    if "switch" in a:
        score += 0.08
    if "switch" in b:
        score -= 0.08
    if ("south" in a) and ("ortho" in b):
        score += 0.06
    if ("ortho" in a) and ("south" in b):
        score -= 0.06
    return score

def _style_guess_from_stats(p):
    """
    Lightweight style inference from stats:
    - higher TDAvg/SubAvg => grappling lean
    - higher SLpM/KD => striking/finishing lean
    """
    td = p.get("TDAvg", np.nan)
    sub = p.get("SubAvg", np.nan)
    slpm = p.get("SLpM", np.nan)
    kd = p.get("KD15", np.nan)

    grap = 0.0
    strike = 0.0
    fin = 0.0

    if not np.isnan(td):
        grap += np.clip((td - 1.0) / 4.0, 0.0, 1.0)
    if not np.isnan(sub):
        grap += np.clip(sub / 2.0, 0.0, 1.0)

    if not np.isnan(slpm):
        strike += np.clip((slpm - 3.0) / 4.0, 0.0, 1.0)
    if not np.isnan(kd):
        fin += np.clip(kd / 1.0, 0.0, 1.0)

    # reduce finish if very decision-heavy
    dec = p.get("DEC%", np.nan)
    if not np.isnan(dec):
        fin -= np.clip((dec - 40.0) / 80.0, 0.0, 0.3)

    strike = float(np.clip(strike, 0.0, 1.0))
    grap = float(np.clip(grap, 0.0, 1.0))
    fin = float(np.clip(fin + 0.25 * strike, 0.0, 1.0))
    return strike, grap, fin

def _parse_last5(s):
    if not isinstance(s, str):
        return (np.nan, np.nan)
    t = s.strip().upper()
    if not t:
        return (np.nan, np.nan)
    if "-" in t:
        parts = t.split("-")
        if len(parts) == 2:
            return (_sf(parts[0]), _sf(parts[1]))
    w = t.count("W")
    l = t.count("L")
    if w + l > 0:
        return (float(w), float(l))
    return (np.nan, np.nan)

def _implied_from_prob(p):
    p = _clip01(p)
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def ufc_predict_fight(fa: dict, fb: dict, bout_totals: dict | None = None):
    """
    Predicts win prob + ITD/DEC environment using:
    age, reach, stance, record, last five, TD def/acc/avg, pace (SLpM/SApM),
    inside distance + decision rates, plus per-fight TD/CTRL when available.
    Rank is NOT reliably available from ufcstats -> neutral.
    """
    # Base features
    age_term = 0.0
    if not np.isnan(fa.get("Age", np.nan)) and not np.isnan(fb.get("Age", np.nan)):
        # mild penalty for 36+
        def age_prime_bonus(age):
            if 27 <= age <= 32:
                return 0.08
            if 33 <= age <= 35:
                return 0.02
            if age < 24:
                return -0.03
            if age >= 36:
                return -0.08
            return 0.0
        age_term = age_prime_bonus(fa["Age"]) - age_prime_bonus(fb["Age"])

    reach_term = 0.0
    if not np.isnan(fa.get("Reach", np.nan)) and not np.isnan(fb.get("Reach", np.nan)):
        reach_term = np.clip((fa["Reach"] - fb["Reach"]) / 10.0, -0.20, 0.20)

    stance_term = _stance_score(fa.get("Stance",""), fb.get("Stance",""))

    # record win%
    def winpct(p):
        w = p.get("Wins", np.nan)
        l = p.get("Losses", np.nan)
        if np.isnan(w) or np.isnan(l) or (w + l) <= 0:
            return np.nan
        return w / (w + l)

    wp_a = winpct(fa)
    wp_b = winpct(fb)
    rec_term = 0.0
    if not np.isnan(wp_a) and not np.isnan(wp_b):
        rec_term = np.clip((wp_a - wp_b), -0.25, 0.25)

    # last five
    l5_aw, l5_al = _parse_last5(fa.get("Last5",""))
    l5_bw, l5_bl = _parse_last5(fb.get("Last5",""))
    mom_term = 0.0
    if not np.isnan(l5_aw) and not np.isnan(l5_al) and not np.isnan(l5_bw) and not np.isnan(l5_bl):
        a = l5_aw / max(1.0, (l5_aw + l5_al))
        b = l5_bw / max(1.0, (l5_bw + l5_bl))
        mom_term = np.clip(a - b, -0.20, 0.20)

    # wrestling/grappling
    tda_a = fa.get("TDAvg", np.nan)
    tda_b = fb.get("TDAvg", np.nan)
    tdacc_a = fa.get("TDAcc%", np.nan)
    tdacc_b = fb.get("TDAcc%", np.nan)
    tddef_a = fa.get("TDDef%", np.nan)
    tddef_b = fb.get("TDDef%", np.nan)

    grap_a = 0.0
    grap_b = 0.0
    if not np.isnan(tda_a):
        grap_a += (tda_a - 1.0) / 6.0
    if not np.isnan(tdacc_a):
        grap_a += (tdacc_a - 30.0) / 120.0
    if not np.isnan(tddef_a):
        grap_a += (tddef_a - 60.0) / 140.0

    if not np.isnan(tda_b):
        grap_b += (tda_b - 1.0) / 6.0
    if not np.isnan(tdacc_b):
        grap_b += (tdacc_b - 30.0) / 120.0
    if not np.isnan(tddef_b):
        grap_b += (tddef_b - 60.0) / 140.0

    grap_term = np.clip(grap_a - grap_b, -0.22, 0.22)

    # pace/defense
    slpm_a = fa.get("SLpM", np.nan)
    slpm_b = fb.get("SLpM", np.nan)
    sapm_a = fa.get("SApM", np.nan)
    sapm_b = fb.get("SApM", np.nan)
    strdef_a = fa.get("StrDef%", np.nan)
    strdef_b = fb.get("StrDef%", np.nan)

    strike_term = 0.0
    if not np.isnan(slpm_a) and not np.isnan(slpm_b):
        strike_term += (slpm_a - slpm_b) / 10.0
    if not np.isnan(sapm_a) and not np.isnan(sapm_b):
        strike_term += (sapm_b - sapm_a) / 10.0
    strike_term = float(np.clip(strike_term, -0.20, 0.20))

    def_term = 0.0
    if not np.isnan(strdef_a) and not np.isnan(strdef_b):
        def_term = float(np.clip((strdef_a - strdef_b) / 100.0, -0.10, 0.10))

    # Inside distance / decision tendencies (career win-method shares)
    itd_a = fa.get("ITD%", np.nan)
    itd_b = fb.get("ITD%", np.nan)
    dec_a = fa.get("DEC%", np.nan)
    dec_b = fb.get("DEC%", np.nan)

    # style inference (no manual style text)
    sa_str, sa_grap, sa_fin = _style_guess_from_stats(fa)
    sb_str, sb_grap, sb_fin = _style_guess_from_stats(fb)
    style_term = np.clip((sa_str - sb_str) * 0.10 + (sa_grap - sb_grap) * 0.10, -0.12, 0.12)

    # Bout-specific adjustment if we have fight totals (helps when you're re-running past events)
    bout_grap = 0.0
    bout_ctrl = 0.0
    if bout_totals and isinstance(bout_totals, dict):
        # these totals are for THIS bout; only meaningful for analysis of completed fights.
        pass

    # rank term neutral (ufcstats doesn't provide reliable ranking)
    rank_term = 0.0

    x = (
        0.80 * rank_term +
        0.85 * rec_term +
        0.55 * mom_term +
        0.55 * grap_term +
        0.45 * strike_term +
        0.20 * def_term +
        0.35 * age_term +
        0.25 * reach_term +
        0.20 * stance_term +
        0.35 * style_term +
        0.10 * bout_grap +
        0.06 * bout_ctrl
    )
    p_win_a = _clip01(_sigmoid(x))

    # Fight finish/decision environment
    # Use ITD/DEC tendencies + pace + defensive stats to set an ITD probability for fight.
    base_finish = 0.52
    if not np.isnan(itd_a) and not np.isnan(itd_b):
        base_finish = np.clip(0.35 + 0.50 * ((itd_a + itd_b) / 200.0), 0.25, 0.80)

    pace = 0.0
    if not np.isnan(slpm_a) and not np.isnan(slpm_b):
        pace += (slpm_a + slpm_b) / 12.0
    if not np.isnan(sapm_a) and not np.isnan(sapm_b):
        pace += (sapm_a + sapm_b) / 14.0
    pace = float(np.clip(pace, 0.0, 1.2))

    def_dec = 0.0
    if not np.isnan(strdef_a) and not np.isnan(strdef_b):
        def_dec += np.clip(((strdef_a + strdef_b) / 2.0 - 50.0) / 100.0, -0.2, 0.2) * 0.10
    if not np.isnan(tddef_a) and not np.isnan(tddef_b):
        def_dec += np.clip(((tddef_a + tddef_b) / 2.0 - 60.0) / 100.0, -0.2, 0.2) * 0.08

    dominance = abs(x)
    fin_tools = np.clip((sa_fin + sb_fin) / 2.0, 0.0, 1.0) * 0.10

    p_itd_fight = np.clip(
        base_finish
        + np.clip(0.10 * dominance, 0.0, 0.12)
        + 0.08 * pace
        + fin_tools
        - def_dec,
        0.18, 0.88
    )
    p_dec_fight = 1.0 - p_itd_fight

    return {
        "p_win_a": p_win_a,
        "p_win_b": 1.0 - p_win_a,
        "p_itd_fight": p_itd_fight,
        "p_dec_fight": p_dec_fight,
    }

def render_ufc_picks():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC Picks — Auto-Pulled Stats (Age/Reach/Stance/Record/Last5/TD/TDD/ITD/DEC + Pace)")
    st.caption(
        "This pulls fighter data automatically from UFCStats (no CSV). "
        "It does not touch Game Lines / Props / PGA logic."
    )

    colA, colB = st.columns([1.2, 1])
    with colA:
        limit = st.slider("Events to load", 5, 50, 25)
    with colB:
        st.caption("Note: UFCStats lists completed events. For upcoming cards, use latest completed or extend with another source.")

    try:
        events = ufc_list_events(limit=int(limit))
    except Exception as e:
        st.error(f"Could not load UFC events: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if events.empty:
        st.warning("No UFC events found.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    options = (events["DateStr"] + " — " + events["Event"]).tolist()
    pick_idx = 0
    sel = st.selectbox("Select Event", options, index=pick_idx)
    sel_row = events.iloc[options.index(sel)]
    event_name = str(sel_row["Event"])
    event_url = str(sel_row["EventURL"])

    if debug:
        st.json({"event": event_name, "event_url": event_url})

    try:
        fights = ufc_event_fights(event_url)
    except Exception as e:
        st.error(f"Could not load fights for event: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if fights.empty:
        st.warning("No fights parsed for this event.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    max_fights = st.slider("Fights to score (performance)", 1, int(min(20, len(fights))), int(min(12, len(fights))))
    fights = fights.head(int(max_fights)).copy()

    # Pull fighter profiles (cached) and generate picks
    rows = []
    with st.spinner("Pulling fighter stats and scoring fights..."):
        for _, r in fights.iterrows():
            fa_url = r.get("FighterA_URL")
            fb_url = r.get("FighterB_URL")
            bout_url = r.get("BoutURL")
            if not isinstance(fa_url, str) or not isinstance(fb_url, str):
                continue

            try:
                fa = ufc_fighter_profile(fa_url)
                fb = ufc_fighter_profile(fb_url)
            except Exception:
                continue

            # Optional: bout totals (mostly useful for analysis; may fail safely)
            bout_tot = {}
            if isinstance(bout_url, str) and bout_url.startswith("http"):
                try:
                    bout_tot = ufc_bout_totals(bout_url)
                except Exception:
                    bout_tot = {}

            pred = ufc_predict_fight(fa, fb, bout_totals=bout_tot)

            fighter_a = _strip(r.get("FighterA", "")) or "Fighter A"
            fighter_b = _strip(r.get("FighterB", "")) or "Fighter B"
            wc = _strip(r.get("WeightClass", ""))

            pA = pred["p_win_a"]
            p_pick = max(pA, 1.0 - pA)
            pick = fighter_a if pA >= 0.5 else fighter_b

            p_itd = pred["p_itd_fight"]
            p_dec = pred["p_dec_fight"]

            best_bet = "Moneyline"
            if p_itd >= 0.60 and p_pick < 0.74:
                best_bet = "Inside the Distance (lean)"
            if p_dec >= 0.58 and p_pick < 0.74:
                best_bet = "Decision (lean)"
            if p_pick >= 0.74:
                best_bet = "Moneyline"

            conf_bucket = "Lean"
            if p_pick >= 0.68:
                conf_bucket = "Solid"
            if p_pick >= 0.75:
                conf_bucket = "Strong"

            rows.append({
                "Event": event_name,
                "WeightClass": wc,
                "Fight": f"{fighter_a} vs {fighter_b}",
                "Pick": pick,
                "Win% (Pick)": round(100 * p_pick, 1),
                "Fair Odds (Pick)": _implied_from_prob(p_pick),
                "Fight ITD%": round(100 * p_itd, 1),
                "Fight DEC%": round(100 * p_dec, 1),
                "Confidence": conf_bucket,
                "Best Bet Type": best_bet,
                # Requested features (display for transparency)
                "AgeA": fa.get("Age", np.nan),
                "AgeB": fb.get("Age", np.nan),
                "ReachA": fa.get("Reach", np.nan),
                "ReachB": fb.get("Reach", np.nan),
                "StanceA": fa.get("Stance", ""),
                "StanceB": fb.get("Stance", ""),
                "RecA": f"{int(fa['Wins'])}-{int(fa['Losses'])}" if not np.isnan(fa.get("Wins", np.nan)) and not np.isnan(fa.get("Losses", np.nan)) else "",
                "RecB": f"{int(fb['Wins'])}-{int(fb['Losses'])}" if not np.isnan(fb.get("Wins", np.nan)) and not np.isnan(fb.get("Losses", np.nan)) else "",
                "Last5A": fa.get("Last5", ""),
                "Last5B": fb.get("Last5", ""),
                "TDAvgA": fa.get("TDAvg", np.nan),
                "TDAvgB": fb.get("TDAvg", np.nan),
                "TDDef%A": fa.get("TDDef%", np.nan),
                "TDDef%B": fb.get("TDDef%", np.nan),
                "TDAcc%A": fa.get("TDAcc%", np.nan),
                "TDAcc%B": fb.get("TDAcc%", np.nan),
                "ITD%A%": round(fa.get("ITD%", np.nan), 1) if not np.isnan(fa.get("ITD%", np.nan)) else np.nan,
                "ITD%B%": round(fb.get("ITD%", np.nan), 1) if not np.isnan(fb.get("ITD%", np.nan)) else np.nan,
                "DEC%A%": round(fa.get("DEC%", np.nan), 1) if not np.isnan(fa.get("DEC%", np.nan)) else np.nan,
                "DEC%B%": round(fb.get("DEC%", np.nan), 1) if not np.isnan(fb.get("DEC%", np.nan)) else np.nan,
                "SLpM_A": fa.get("SLpM", np.nan),
                "SLpM_B": fb.get("SLpM", np.nan),
                "SApM_A": fa.get("SApM", np.nan),
                "SApM_B": fb.get("SApM", np.nan),
            })
            time.sleep(0.03)

    if not rows:
        st.warning("No fights could be scored (UFCStats parsing may be throttled). Try fewer fights or reload.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    df_out = pd.DataFrame(rows).sort_values("Win% (Pick)", ascending=False)

    st.markdown("### Picks (ranked by Win% confidence)")
    view_cols = [
        "WeightClass","Fight","Pick","Win% (Pick)","Fair Odds (Pick)","Confidence","Best Bet Type","Fight ITD%","Fight DEC%",
        "AgeA","AgeB","ReachA","ReachB","StanceA","StanceB","RecA","RecB","Last5A","Last5B",
        "TDAvgA","TDAvgB","TDAcc%A","TDAcc%B","TDDef%A","TDDef%B","ITD%A%","ITD%B%","DEC%A%","DEC%B%",
        "SLpM_A","SLpM_B","SApM_A","SApM_B"
    ]
    view_cols = [c for c in view_cols if c in df_out.columns]
    st.dataframe(df_out[view_cols], use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Log UFC Picks to Tracker"):
            rows_to_log = []
            for _, r in df_out.iterrows():
                rows_to_log.append({
                    "LoggedAt": datetime.now().isoformat(),
                    "Mode": "UFC",
                    "Sport": "UFC",
                    "Market": r.get("Best Bet Type", "Moneyline"),
                    "Event": r.get("Event", event_name),
                    "Selection": r.get("Pick", ""),
                    "Line": "",
                    "BestBook": "",
                    "BestPrice": "",
                    "YourProb": (pd.to_numeric(r.get("Win% (Pick)"), errors="coerce") / 100.0),
                    "Implied": np.nan,
                    "Edge": np.nan,
                    "EV": np.nan,
                    "Status": "Pending",
                    "Result": "",
                })
            tracker_log_rows(rows_to_log)
            st.success("Logged UFC picks ✅ (grade later in Tracker)")
    with c2:
        st.caption("Tip: If you want edges vs books, we can add UFC odds from The Odds API later (separate, still isolated).")

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
        st.info("No tracked picks yet. Log picks from Game Lines / Props / PGA / UFC.")

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

if mode == "UFC Picks":
    render_ufc_picks()
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
