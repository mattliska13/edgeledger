import os
import time
from datetime import datetime, timedelta, timezone
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
SESSION.headers.update({"User-Agent": "EdgeLedger/1.0 (streamlit)"})

def safe_get(url: str, params: dict | None = None, timeout: int = 25):
    try:
        r = SESSION.get(url, params=params or {}, timeout=timeout)
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

# ✅ include UFC in radio without impacting others
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
# UFC Module (ESPN APIs only; no lxml; no CSV)
# =========================================================
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
ESPN_ATHLETE_V3 = "https://site.web.api.espn.com/apis/common/v3/sports/mma/ufc/athletes/{athlete_id}"
ESPN_RANKINGS = "https://site.web.api.espn.com/apis/v2/sports/mma/ufc/rankings"

def _utc_now():
    return datetime.now(timezone.utc)

def _parse_iso(s: str):
    try:
        # ESPN iso typically ends with Z
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

@st.cache_data(ttl=60 * 10)
def ufc_fetch_scoreboard():
    ok, status, payload, url = safe_get(ESPN_SCOREBOARD, params={"limit": 100}, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": url}

@st.cache_data(ttl=60 * 60)
def ufc_fetch_rankings():
    ok, status, payload, url = safe_get(ESPN_RANKINGS, params={"region": "us", "lang": "en"}, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": url}

@st.cache_data(ttl=60 * 60)
def ufc_fetch_athlete(athlete_id: str):
    url = ESPN_ATHLETE_V3.format(athlete_id=str(athlete_id))
    ok, status, payload, final_url = safe_get(url, params={"region": "us", "lang": "en"}, timeout=20)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def ufc_list_upcoming_events(scoreboard_payload: dict):
    events = []
    if not isinstance(scoreboard_payload, dict):
        return events

    cutoff = _utc_now() - timedelta(hours=8)  # ✅ fixed (no to_pytimedelta)

    for ev in (scoreboard_payload.get("events", []) or []):
        if not isinstance(ev, dict):
            continue
        ev_id = str(ev.get("id") or "").strip()
        nm = (ev.get("name") or ev.get("shortName") or "").replace("\n", " ").strip()
        dt = ev.get("date")
        comps = ev.get("competitions", []) or []
        if not ev_id or not dt:
            continue
        d = _parse_iso(str(dt))
        if not d:
            continue
        if d >= cutoff:
            events.append({
                "id": ev_id,
                "name": nm or f"UFC Event {ev_id}",
                "date": str(dt),
                "competitions": comps,
                "comps_count": len(comps),
            })

    events.sort(key=lambda x: x["date"])
    return events

def _rank_map_from_rankings(payload: dict):
    # Best-effort mapping athleteId -> rank (lower is better)
    m = {}
    if not isinstance(payload, dict):
        return m
    # structure can vary; try common shapes
    cats = payload.get("rankings") or payload.get("children") or payload.get("leagues") or []
    if isinstance(cats, dict):
        cats = [cats]
    def walk(obj):
        if isinstance(obj, dict):
            # entries
            entries = obj.get("ranks") or obj.get("entries") or obj.get("items") or []
            if isinstance(entries, dict):
                entries = [entries]
            for e in entries:
                if not isinstance(e, dict):
                    continue
                athlete = e.get("athlete") or e.get("competitor") or {}
                aid = athlete.get("id") or athlete.get("uid") or ""
                aid = str(aid).replace("s:mma~l:ufc~a:", "").strip()
                rk = e.get("rank") or e.get("position") or e.get("seed")
                try:
                    rk = int(rk)
                except Exception:
                    rk = None
                if aid and rk:
                    m[str(aid)] = rk
            # recurse
            for k in ["children", "rankings", "categories", "groups"]:
                if k in obj:
                    walk(obj[k])
        elif isinstance(obj, list):
            for it in obj:
                walk(it)
    walk(payload)
    return m

def _safe_num(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace("%", "")
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan

def _extract_record(payload: dict):
    # returns wins, losses, draws, win_pct
    w = l = d = np.nan
    if not isinstance(payload, dict):
        return w, l, d, np.nan
    rec = payload.get("record") or payload.get("records") or []
    # sometimes 'record' is dict with 'items'
    if isinstance(rec, dict):
        rec = rec.get("items") or rec.get("records") or []
    if isinstance(rec, list):
        # pick first overall record-like
        for r in rec:
            if not isinstance(r, dict):
                continue
            summ = r.get("summary") or r.get("displayValue") or r.get("value")
            if isinstance(summ, str) and "-" in summ:
                parts = summ.split("-")
                try:
                    w = int(parts[0]); l = int(parts[1])
                    d = int(parts[2]) if len(parts) > 2 else 0
                    break
                except Exception:
                    continue
    try:
        win_pct = w / max(1, (w + l))
    except Exception:
        win_pct = np.nan
    return w, l, d, win_pct

def _extract_bio(payload: dict):
    # age, reach_in, stance
    age = reach = np.nan
    stance = ""
    if not isinstance(payload, dict):
        return age, reach, stance
    bio = payload.get("athlete") or payload.get("bio") or payload
    # age sometimes in "age"
    age = _safe_num(bio.get("age")) if isinstance(bio, dict) else np.nan
    # reach in inches: may be "reach" like "72\""
    reach_raw = None
    if isinstance(bio, dict):
        reach_raw = bio.get("reach") or bio.get("reachIn") or bio.get("reach_in")
        stance = (bio.get("stance") or bio.get("stanceType") or "").strip()
    if isinstance(reach_raw, str):
        reach = _safe_num(reach_raw.replace('"', "").replace("in", "").strip())
    else:
        reach = _safe_num(reach_raw)
    return age, reach, stance

def _extract_stats(payload: dict):
    # striking, takedowns (best-effort)
    # returns: slpm, sapm, str_acc, str_def, td_avg, td_acc, td_def
    slpm = sapm = str_acc = str_def = td_avg = td_acc = td_def = np.nan
    if not isinstance(payload, dict):
        return slpm, sapm, str_acc, str_def, td_avg, td_acc, td_def

    # ESPN shapes vary; try payload["statistics"] list or payload["splits"]
    stats = payload.get("statistics") or payload.get("stats") or []
    if isinstance(stats, dict):
        stats = stats.get("splits") or stats.get("categories") or stats.get("statistics") or []

    # flatten name/value pairs where possible
    pairs = {}

    def grab(obj):
        if isinstance(obj, dict):
            # maybe already a pair
            n = obj.get("name") or obj.get("abbreviation")
            v = obj.get("value") if "value" in obj else obj.get("displayValue")
            if n and v is not None:
                pairs[str(n).lower()] = v
            # recurse
            for k in ["stats", "statistics", "splits", "categories", "items", "children"]:
                if k in obj:
                    grab(obj[k])
        elif isinstance(obj, list):
            for it in obj:
                grab(it)

    grab(stats)

    # common keys (best effort)
    slpm = _safe_num(pairs.get("slpm") or pairs.get("sig. str. landed per min") or pairs.get("strikes landed per min"))
    sapm = _safe_num(pairs.get("sapm") or pairs.get("sig. str. absorbed per min") or pairs.get("strikes absorbed per min"))
    str_acc = _safe_num(pairs.get("str_acc") or pairs.get("sig. str. acc.") or pairs.get("sig str acc") or pairs.get("striking accuracy"))
    str_def = _safe_num(pairs.get("str_def") or pairs.get("sig. str. def.") or pairs.get("sig str def") or pairs.get("striking defense"))
    td_avg = _safe_num(pairs.get("td_avg") or pairs.get("takedown avg") or pairs.get("takedowns landed per 15 min"))
    td_acc = _safe_num(pairs.get("td_acc") or pairs.get("takedown accuracy") or pairs.get("td acc"))
    td_def = _safe_num(pairs.get("td_def") or pairs.get("takedown defense") or pairs.get("td def"))

    return slpm, sapm, str_acc, str_def, td_avg, td_acc, td_def

def _extract_last5(payload: dict):
    # best-effort last-5 from eventLog-like sections
    # returns last5_win_pct, last5_wins, last5_losses
    wins = losses = 0
    if not isinstance(payload, dict):
        return np.nan, np.nan, np.nan

    log = payload.get("eventLog") or payload.get("events") or payload.get("history") or None
    items = []
    if isinstance(log, dict):
        items = log.get("events") or log.get("items") or []
    elif isinstance(log, list):
        items = log

    # count last 5 where result available
    for it in items[:20]:
        if not isinstance(it, dict):
            continue
        res = it.get("result") or it.get("outcome") or it.get("displayResult") or ""
        res = str(res).upper()
        if res.startswith("W"):
            wins += 1
        elif res.startswith("L"):
            losses += 1
        if wins + losses >= 5:
            break

    total = wins + losses
    if total == 0:
        return np.nan, np.nan, np.nan
    return wins / total, float(wins), float(losses)

def _stance_match_bonus(a: str, b: str):
    # small heuristic: opposite stances slightly increase variance
    a = (a or "").lower()
    b = (b or "").lower()
    if not a or not b:
        return 0.0
    if a == b:
        return -0.03
    return 0.03

def _inside_distance_proxy(rec_w: float, rec_l: float, slpm: float, td_avg: float):
    # proxy "finish likelihood": higher activity tends to correlate
    w = 0.0
    if np.isfinite(slpm):
        w += 0.25 * (slpm / 6.0)  # normalize
    if np.isfinite(td_avg):
        w += 0.15 * (td_avg / 6.0)
    if np.isfinite(rec_w) and np.isfinite(rec_l) and (rec_w + rec_l) > 0:
        w += 0.15 * (rec_w / (rec_w + rec_l))
    return float(np.clip(w, 0.0, 1.0))

def _build_ufc_model_row(aid: str, name: str, payload: dict, rank_map: dict):
    w, l, d, win_pct = _extract_record(payload)
    age, reach, stance = _extract_bio(payload)
    slpm, sapm, str_acc, str_def, td_avg, td_acc, td_def = _extract_stats(payload)
    last5_pct, last5_w, last5_l = _extract_last5(payload)

    rk = rank_map.get(str(aid), np.nan)
    rk = _safe_num(rk)

    return {
        "athlete_id": str(aid),
        "Fighter": name,
        "Age": age,
        "Reach": reach,
        "Stance": stance,
        "W": w,
        "L": l,
        "D": d,
        "WinPct": win_pct,
        "Rank": rk,
        "Last5WinPct": last5_pct,
        "SLpM": slpm,
        "SApM": sapm,
        "StrAcc%": str_acc,
        "StrDef%": str_def,
        "TD_Avg": td_avg,
        "TD_Acc%": td_acc,
        "TD_Def%": td_def,
    }

def _z_series(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce")
    if x.isna().all():
        return pd.Series(np.zeros(len(x)), index=x.index)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd

def ufc_predict_fights(fights: list[dict], fighter_df: pd.DataFrame):
    """
    fights: list with keys {fight_name, a_id, a_name, b_id, b_name}
    fighter_df rows contain the features for each athlete_id.
    returns dataframe predictions per fight.
    """
    if not fights or fighter_df.empty:
        return pd.DataFrame()

    fdf = fighter_df.copy()
    # normalize some columns
    num_cols = ["Age","Reach","WinPct","Rank","Last5WinPct","SLpM","SApM","StrAcc%","StrDef%","TD_Avg","TD_Acc%","TD_Def%"]
    for c in num_cols:
        if c in fdf.columns:
            fdf[c] = pd.to_numeric(fdf[c], errors="coerce")

    # create z columns
    # Rank: lower better => invert
    fdf["RankInv"] = np.where(fdf["Rank"].notna(), -fdf["Rank"], np.nan)

    fdf["z_win"] = _z_series(fdf["WinPct"])
    fdf["z_rank"] = _z_series(fdf["RankInv"])
    fdf["z_last5"] = _z_series(fdf["Last5WinPct"])
    fdf["z_reach"] = _z_series(fdf["Reach"])
    fdf["z_age"] = _z_series(-fdf["Age"])  # younger slight bonus

    # striking: higher SLpM, lower SApM, higher acc/def
    fdf["z_slpm"] = _z_series(fdf["SLpM"])
    fdf["z_sapm"] = _z_series(-fdf["SApM"])
    fdf["z_stracc"] = _z_series(fdf["StrAcc%"])
    fdf["z_strdef"] = _z_series(fdf["StrDef%"])

    # wrestling: higher TD avg/acc/def
    fdf["z_tdavg"] = _z_series(fdf["TD_Avg"])
    fdf["z_tdacc"] = _z_series(fdf["TD_Acc%"])
    fdf["z_tddef"] = _z_series(fdf["TD_Def%"])

    # Base score per fighter
    fdf["BaseScore"] = (
        0.28 * fdf["z_win"] +
        0.14 * fdf["z_rank"] +
        0.10 * fdf["z_last5"] +
        0.06 * fdf["z_reach"] +
        0.05 * fdf["z_age"] +
        0.12 * fdf["z_slpm"] +
        0.10 * fdf["z_sapm"] +
        0.05 * fdf["z_stracc"] +
        0.05 * fdf["z_strdef"] +
        0.03 * fdf["z_tdavg"] +
        0.03 * fdf["z_tdacc"] +
        0.03 * fdf["z_tddef"]
    )

    # build fight rows
    rows = []
    lookup = {str(r["athlete_id"]): r for _, r in fdf.iterrows()}

    for fx in fights:
        a_id = str(fx["a_id"]); b_id = str(fx["b_id"])
        a = lookup.get(a_id); b = lookup.get(b_id)
        if not a or not b:
            continue

        # stance heuristic
        stance_bonus = _stance_match_bonus(a.get("Stance",""), b.get("Stance",""))

        # inside distance proxy
        a_idp = _inside_distance_proxy(a.get("W",np.nan), a.get("L",np.nan), a.get("SLpM",np.nan), a.get("TD_Avg",np.nan))
        b_idp = _inside_distance_proxy(b.get("W",np.nan), b.get("L",np.nan), b.get("SLpM",np.nan), b.get("TD_Avg",np.nan))

        # score diff -> probability
        diff = float(a["BaseScore"] - b["BaseScore"]) + stance_bonus
        # logistic
        p_a = 1.0 / (1.0 + np.exp(-diff))
        p_a = float(np.clip(p_a, 0.05, 0.95))
        pick = fx["a_name"] if p_a >= 0.5 else fx["b_name"]
        p_pick = p_a if p_a >= 0.5 else (1.0 - p_a)

        # finish vs decision heuristic: if either side has high inside-distance proxy, lean finish
        finishness = float(np.clip((a_idp + b_idp) / 2.0, 0.0, 1.0))
        method = "Decision lean"
        if finishness >= 0.58:
            method = "Inside distance lean"

        rows.append({
            "Fight": fx["fight_name"],
            "Pick": pick,
            "Conf%": round(p_pick * 100.0, 1),
            "A": fx["a_name"],
            "B": fx["b_name"],
            "P(A)%": round(p_a * 100.0, 1),
            "MethodHint": method,
            "FinishProxy": round(finishness, 2),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Conf%", ascending=False)

def ufc_build_event_fights(event_obj: dict):
    """
    From scoreboard event object, extract fight list.
    """
    fights = []
    comps = event_obj.get("competitions", []) or []
    for c in comps:
        if not isinstance(c, dict):
            continue
        competitors = c.get("competitors", []) or []
        if len(competitors) < 2:
            continue
        # ESPN competitor entries
        a = competitors[0].get("athlete") or competitors[0].get("competitor") or {}
        b = competitors[1].get("athlete") or competitors[1].get("competitor") or {}
        a_id = a.get("id") or ""
        b_id = b.get("id") or ""
        a_name = a.get("displayName") or a.get("shortName") or competitors[0].get("displayName") or "Fighter A"
        b_name = b.get("displayName") or b.get("shortName") or competitors[1].get("displayName") or "Fighter B"
        if not a_id or not b_id:
            # try uid style
            a_id = str(a.get("uid") or "").replace("s:mma~l:ufc~a:", "")
            b_id = str(b.get("uid") or "").replace("s:mma~l:ufc~a:", "")
        fight_name = f"{a_name} vs {b_name}"
        fights.append({
            "fight_name": fight_name,
            "a_id": str(a_id),
            "b_id": str(b_id),
            "a_name": a_name,
            "b_name": b_name,
        })
    return fights

def render_ufc():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🥊 UFC — Model Picks (ESPN data)")
    st.caption(
        "Pulls upcoming UFC cards from ESPN scoreboard + athlete endpoints. "
        "Model uses age/reach/stance/record + striking + takedowns; finish/decision is a heuristic proxy."
    )

    sb = ufc_fetch_scoreboard()
    if debug:
        st.json({"ufc_scoreboard": {"ok": sb["ok"], "status": sb["status"], "url": sb["url"]}})

    if not sb["ok"] or not isinstance(sb["payload"], dict):
        st.warning("Could not load UFC events from ESPN scoreboard right now.")
        if debug:
            st.json(sb)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    events = ufc_list_upcoming_events(sb["payload"])
    if not events:
        st.warning("No UFC events found from ESPN scoreboard right now.")
        if debug:
            st.json({"payload_keys": list(sb["payload"].keys())})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # event picker
    label_map = {}
    for e in events:
        dt = e.get("date","")
        try:
            d = _parse_iso(dt)
            dt_disp = d.astimezone().strftime("%a %b %d %I:%M%p") if d else dt
        except Exception:
            dt_disp = dt
        label = f"{e['name']} — {dt_disp}"
        label_map[label] = e

    sel = st.selectbox("Select upcoming UFC event", list(label_map.keys()), index=0)
    event_obj = label_map.get(sel, events[0])

    fights = ufc_build_event_fights(event_obj)
    if not fights:
        st.warning("Could not parse fight list for this event from ESPN scoreboard payload.")
        if debug:
            st.json({"event_obj_keys": list(event_obj.keys()), "comps_count": event_obj.get("comps_count")})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### Fight card (parsed)")
    st.dataframe(pd.DataFrame([{"Fight": f["fight_name"], "A_ID": f["a_id"], "B_ID": f["b_id"]} for f in fights]),
                 use_container_width=True, hide_index=True)

    # rankings map
    rnk = ufc_fetch_rankings()
    rank_map = {}
    if rnk["ok"] and isinstance(rnk["payload"], dict):
        rank_map = _rank_map_from_rankings(rnk["payload"])
    if debug:
        st.json({"rankings_ok": rnk["ok"], "rank_map_size": len(rank_map)})

    # fetch athlete payloads
    athlete_ids = sorted(list({f["a_id"] for f in fights} | {f["b_id"] for f in fights}))
    fighter_rows = []
    diag = {"athletes": [], "missing": []}

    with st.spinner("Loading fighter data from ESPN..."):
        for aid in athlete_ids:
            resp = ufc_fetch_athlete(aid)
            diag["athletes"].append({"id": aid, "ok": resp["ok"], "status": resp["status"]})
            if not resp["ok"] or not isinstance(resp["payload"], dict):
                diag["missing"].append(aid)
                continue
            # name best-effort
            nm = ""
            try:
                nm = resp["payload"].get("athlete", {}).get("displayName") or resp["payload"].get("displayName") or ""
            except Exception:
                nm = ""
            fighter_rows.append(_build_ufc_model_row(aid, nm or f"ID {aid}", resp["payload"], rank_map))
            time.sleep(0.02)

    fighter_df = pd.DataFrame(fighter_rows)
    if fighter_df.empty:
        st.warning("Could not build UFC picks for this event (fighter data returned but model table empty).")
        if debug:
            st.json(diag)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Merge correct names from fight list if ESPN athlete name blank
    id_to_name = {}
    for f in fights:
        id_to_name[str(f["a_id"])] = f["a_name"]
        id_to_name[str(f["b_id"])] = f["b_name"]
    fighter_df["Fighter"] = fighter_df.apply(lambda r: id_to_name.get(str(r["athlete_id"]), r["Fighter"]), axis=1)

    st.markdown("### Fighter metrics (best-effort from ESPN)")
    show_cols = [
        "Fighter","Age","Reach","Stance","W","L","WinPct","Rank","Last5WinPct",
        "SLpM","SApM","StrAcc%","StrDef%","TD_Avg","TD_Acc%","TD_Def%"
    ]
    show_cols = [c for c in show_cols if c in fighter_df.columns]
    st.dataframe(fighter_df[show_cols].sort_values("Rank", ascending=True, na_position="last"),
                 use_container_width=True, hide_index=True)

    preds = ufc_predict_fights(fights, fighter_df)
    if preds.empty:
        st.warning("Could not build UFC picks for this event (fight list parsed but prediction table empty).")
        if debug:
            st.json({"diag": diag, "fighter_df_cols": list(fighter_df.columns)})
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("### 🧠 Model picks (ranked by confidence)")
    st.dataframe(preds, use_container_width=True, hide_index=True)

    if debug:
        with st.expander("UFC diagnostics"):
            st.json({"event": event_obj, "diag": diag})

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# MAIN
# =========================================================
if mode == "Tracker":
    render_tracker()
    st.stop()

if mode == "UFC":
    # ✅ hard-guard so UFC can NEVER break the rest of the app
    try:
        render_ufc()
    except Exception as e:
        st.error("UFC module failed, but the rest of the dashboard is unaffected.")
        if debug:
            st.exception(e)
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
