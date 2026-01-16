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
# Support either DATAGOLF_KEY or DATAGOLF_API_KEY
DATAGOLF_KEY = get_key("DATAGOLF_KEY", "") or get_key("DATAGOLF_API_KEY", "")

# =========================
# HTTP
# =========================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "EdgeLedger/1.0 (streamlit)"})

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

# =========================
# API Config (The Odds API)
# =========================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"  # ONLY these

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
    "Passing Yards": "player_pass_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# =========================
# Sidebar UI
# =========================
st.sidebar.markdown("<div class='big-title'>Dashboard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='subtle'>Implied • YourProb • Edge • Best Price</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

debug = st.sidebar.checkbox("Show debug logs", value=False)

mode = st.sidebar.radio("Section", ["Game Lines", "Player Props", "PGA", "Tracker"], index=0)

with st.sidebar.expander("API Keys (optional runtime entry)", expanded=False):
    st.caption("If Secrets aren’t set, paste keys here (session-only).")
    odds_in = st.text_input("ODDS_API_KEY", value=ODDS_API_KEY or "", type="password")
    dg_in = st.text_input("DATAGOLF_KEY", value=DATAGOLF_KEY or "", type="password")
    if odds_in.strip():
        st.session_state["ODDS_API_KEY"] = odds_in.strip()
        ODDS_API_KEY = odds_in.strip()
    if dg_in.strip():
        st.session_state["DATAGOLF_KEY"] = dg_in.strip()
        DATAGOLF_KEY = dg_in.strip()

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption(
    "**Best bets** = Edge = (YourProb − ImpliedProb(best price)). "
    "**Ranked by Edge**. "
    "**No contradictions** across DK/FD OR within a book. "
    "Separate API calls for game lines, props, and PGA."
)

if not ODDS_API_KEY.strip():
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..." or paste it in the sidebar expander.')
    st.stop()

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
            book = bm.get("key") or bm.get("title")
            for mk in (bm.get("markets", []) or []):
                mkey = mk.get("key")
                for out in (mk.get("outcomes", []) or []):
                    rows.append({
                        "Scope": "GameLine",
                        "Event": matchup,
                        "Commence": commence,
                        "Market": mkey,
                        "Outcome": out.get("name"),
                        "Line": out.get("point"),
                        "Price": out.get("price"),
                        "Book": book,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Outcome", "Price", "Book"])

def normalize_props(event_payload):
    """
    FIXED mapping for The Odds API props:
      - out['description'] is usually Player
      - out['name'] is usually Side (Over/Under/Yes)
    """
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
                desc = out.get("description")
                name = out.get("name")

                player = desc if isinstance(desc, str) and desc.strip() else name
                side = name if isinstance(name, str) else ""

                # rare feed reversal
                if isinstance(desc, str) and desc.strip().lower() in ["over", "under", "yes", "no"]:
                    player, side = name, desc

                price = out.get("price")
                if player is None or price is None:
                    continue

                rows.append({
                    "Scope": "Prop",
                    "Event": matchup,
                    "Market": mkey,
                    "Player": str(player).strip(),
                    "Side": str(side).strip(),
                    "Line": out.get("point"),
                    "Price": price,
                    "Book": book,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Market", "Player", "Price", "Book"])
    df = df[df["Player"].astype(str).str.strip() != ""]
    return df

# ==========================================================
# Prob/Edge model (market-derived baseline)
# ==========================================================
def add_implied(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Implied"] = out["Price"].apply(american_to_implied)
    out["Implied"] = clamp01(pd.to_numeric(out["Implied"], errors="coerce").fillna(0.5))
    return out

def compute_no_vig_two_way_within_book(df: pd.DataFrame, book_group_cols: list) -> pd.DataFrame:
    out = df.copy()
    sums = out.groupby(book_group_cols)["Implied"].transform("sum")
    out["NoVigProb"] = np.where(sums > 0, out["Implied"] / sums, np.nan)
    return out

def estimate_your_prob(df: pd.DataFrame, key_cols: list, book_group_cols: list) -> pd.DataFrame:
    """
    YourProb = avg of no-vig probs across books when possible, else avg implied.
    """
    if df.empty:
        return df.copy()

    out = add_implied(df)
    out = compute_no_vig_two_way_within_book(out, book_group_cols)

    nv_avg = out.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out.groupby(key_cols)["Implied"].transform("mean")
    out["YourProb"] = np.where(pd.notna(nv_avg), nv_avg, imp_avg)
    out["YourProb"] = clamp01(pd.to_numeric(out["YourProb"], errors="coerce").fillna(out["Implied"]))
    return out

def best_price_per_exact_bet(df: pd.DataFrame, group_cols_best: list) -> pd.DataFrame:
    """
    Best price across books for an exact bet identity (including exact line when present).
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

    best["Edge"] = pd.to_numeric(best["YourProb"], errors="coerce") - pd.to_numeric(best["ImpliedBest"], errors="coerce")
    best["EV"] = (best["Edge"] * 100.0).round(2)
    return best

def collapse_contradictions_one_pick(df_best: pd.DataFrame, contradiction_cols: list) -> pd.DataFrame:
    """
    ABSOLUTE contradiction rule:
    keep exactly ONE row per contradiction group: max Edge.
    This removes contradictions across DK/FD and within same book by construction.
    """
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_cols, dropna=False)["Edge"].idxmax()
    out = out.loc[idx].sort_values("Edge", ascending=False)
    return out

def keep_value_only(df_best: pd.DataFrame) -> pd.DataFrame:
    if df_best.empty:
        return df_best
    out = df_best.copy()
    out["Edge"] = pd.to_numeric(out["Edge"], errors="coerce")
    return out[out["Edge"] > 0].sort_values("Edge", ascending=False)

def add_display_cols(df_best: pd.DataFrame) -> pd.DataFrame:
    out = df_best.copy()
    out["YourProb%"] = pct01_to_100(out["YourProb"])
    out["Implied%"] = pct01_to_100(out["ImpliedBest"])
    out["Edge%"] = pct01_to_100(out["Edge"])
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").round(2)
    return out

# ==========================================================
# Boards (independent)
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
        return pd.DataFrame(), {"error": "No normalized game line rows"}

    df = df[df["Market"] == market_key].copy()
    if df.empty:
        return pd.DataFrame(), {"error": "No rows for selected market"}

    # For best-price-per-exact-line identity:
    key_cols = ["Event", "Market", "Outcome"]
    book_cols = ["Event", "Market", "Book"]
    best_cols = ["Event", "Market", "Outcome"]

    # Include exact line for spreads/totals so best price is correct for that exact number
    if market_key in ["spreads", "totals"]:
        key_cols += ["Line"]
        book_cols += ["Line"]
        best_cols += ["Line"]

    df = estimate_your_prob(df, key_cols=key_cols, book_group_cols=book_cols)
    df_best = best_price_per_exact_bet(df, group_cols_best=best_cols)

    # CONTRADICTION RULES (absolute):
    # - Moneyline: ONE pick per game
    # - Totals: ONE pick per game (no Over/Under even if line differs)
    # - Spreads: ONE pick per game (no both sides even if line differs)
    contradiction_cols = ["Event", "Market"]  # NOTE: ignores line on purpose

    df_best = collapse_contradictions_one_pick(df_best, contradiction_cols=contradiction_cols)
    df_best = keep_value_only(df_best)
    df_best = add_display_cols(df_best)

    # Final safety: ensure no duplicates can sneak through
    df_best = df_best.sort_values("Edge", ascending=False).drop_duplicates(subset=["Event", "Market"], keep="first")

    return df_best, {}

def build_props_board(sport: str, prop_label: str, max_events_scan: int = 6):
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

    # Identity for best price:
    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    best_cols = ["Event", "Market", "Player", "Side"]

    if df["Line"].notna().any():
        key_cols += ["Line"]
        book_cols += ["Line"]
        best_cols += ["Line"]

    df = estimate_your_prob(df, key_cols=key_cols, book_group_cols=book_cols)
    df_best = best_price_per_exact_bet(df, group_cols_best=best_cols)

    # CONTRADICTION RULE (absolute):
    # Only ONE pick per Event+Market+Player (prevents Over & Under even if lines differ across books)
    df_best = collapse_contradictions_one_pick(df_best, contradiction_cols=["Event", "Market", "Player"])
    df_best = keep_value_only(df_best)
    df_best = add_display_cols(df_best)

    # Final safety: remove any remaining duplicates at player level
    df_best = df_best.sort_values("Edge", ascending=False).drop_duplicates(subset=["Event", "Market", "Player"], keep="first")

    return df_best, {}

# ==========================================================
# Charts
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

# ==========================================================
# PGA (DataGolf) — restored module
# ==========================================================
DG_HOST = "https://feeds.datagolf.com"

@st.cache_data(ttl=60 * 60 * 6)
def dg_pre_tournament(model: str):
    url = f"{DG_HOST}/preds/pre-tournament"
    params = {
        "tour": "pga",
        "add_position": 10,
        "dead_heat": "yes",
        "odds_format": "percent",
        "file_format": "json",
        "key": DATAGOLF_KEY,
        "model": model,
    }
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def dg_decompositions():
    url = f"{DG_HOST}/preds/player-decompositions"
    params = {"tour": "pga", "file_format": "json", "key": DATAGOLF_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

@st.cache_data(ttl=60 * 60 * 24)
def dg_skill_ratings():
    url = f"{DG_HOST}/preds/skill-ratings"
    params = {"display": "value", "file_format": "json", "key": DATAGOLF_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url}

def _dg_rows(payload):
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    if isinstance(payload, list):
        return payload
    return []

def _find_name_col(df):
    for c in ["player_name", "name", "golfer", "player"]:
        if c in df.columns:
            return c
    return None

def build_pga_tables():
    if not DATAGOLF_KEY.strip():
        return None, "Missing DATAGOLF_KEY"

    # Prefer baseline_history_fit, else baseline
    used_model = None
    pre_df = None
    for m in ["baseline_history_fit", "baseline"]:
        res = dg_pre_tournament(m)
        if debug:
            st.json({"dg_pre_tournament": {"model": m, "status": res["status"], "url": res["url"]}})
        if res["ok"]:
            rows = _dg_rows(res["payload"])
            if rows:
                pre_df = pd.DataFrame(rows)
                used_model = m
                break

    if pre_df is None or pre_df.empty:
        return None, "No PGA prediction rows returned from DataGolf."

    # Identify player column
    pcol = _find_name_col(pre_df) or pre_df.columns[0]
    pre_df = pre_df.rename(columns={pcol: "Player"})
    pre_df["Player"] = pre_df["Player"].astype(str)

    # Identify win/top10 fields
    win_col = None
    top10_col = None
    for c in pre_df.columns:
        lc = c.lower()
        if win_col is None and lc in ["win", "win_prob", "win%"]:
            win_col = c
        if top10_col is None and lc in ["top_10", "top10", "top_10_prob", "top10_prob"]:
            top10_col = c
    if win_col is None:
        for c in pre_df.columns:
            if "win" in c.lower():
                win_col = c
                break
    if top10_col is None:
        for c in pre_df.columns:
            if "top" in c.lower() and "10" in c.lower():
                top10_col = c
                break

    if win_col is None or top10_col is None:
        return None, f"Could not detect win/top10 columns. Columns: {list(pre_df.columns)}"

    pre_df["WinProb"] = pd.to_numeric(pre_df[win_col], errors="coerce") / 100.0
    pre_df["Top10Prob"] = pd.to_numeric(pre_df[top10_col], errors="coerce") / 100.0
    pre_df = pre_df.dropna(subset=["WinProb", "Top10Prob"]).copy()
    pre_df["WinProb"] = clamp01(pre_df["WinProb"])
    pre_df["Top10Prob"] = clamp01(pre_df["Top10Prob"])

    # Merge skill + decompositions if available
    de = dg_decompositions()
    sk = dg_skill_ratings()
    if debug:
        st.json({"dg_decomp": {"status": de["status"], "url": de["url"]}})
        st.json({"dg_skill": {"status": sk["status"], "url": sk["url"]}})

    df_de = pd.DataFrame(_dg_rows(de["payload"])) if de["ok"] else pd.DataFrame()
    df_sk = pd.DataFrame(_dg_rows(sk["payload"])) if sk["ok"] else pd.DataFrame()

    if not df_sk.empty:
        n = _find_name_col(df_sk)
        if n:
            df_sk = df_sk.rename(columns={n: "Player"})
        # pick likely columns
        t2g = next((c for c in df_sk.columns if "t2g" in c.lower() or "tee_to_green" in c.lower()), None)
        putt = next((c for c in df_sk.columns if "putt" in c.lower()), None)
        bogey = next((c for c in df_sk.columns if "bogey" in c.lower() and ("avoid" in c.lower() or "avo" in c.lower())), None)
        keep = ["Player"] + [c for c in [t2g, putt, bogey] if c]
        df_sk = df_sk[keep].copy()
        pre_df = pre_df.merge(df_sk, on="Player", how="left")

    if not df_de.empty:
        n = _find_name_col(df_de)
        if n:
            df_de = df_de.rename(columns={n: "Player"})
        # keep common signal columns if present
        keep = ["Player"]
        for c in df_de.columns:
            lc = c.lower()
            if any(k in lc for k in ["course", "history", "fit", "recent", "form", "bogey"]):
                keep.append(c)
        keep = list(dict.fromkeys(keep))
        df_de = df_de[keep].copy()
        pre_df = pre_df.merge(df_de, on="Player", how="left")

    # Rank outputs
    win = pre_df.copy().sort_values("WinProb", ascending=False).head(10)
    top10 = pre_df.copy().sort_values("Top10Prob", ascending=False).head(10)

    # One-and-Done score blend (win + top10 + small nudges if skill cols exist)
    od = pre_df.copy()
    score = od["WinProb"] * 0.60 + od["Top10Prob"] * 0.40
    for c in od.columns:
        lc = c.lower()
        if "t2g" in lc or "tee_to_green" in lc:
            score = score + pd.to_numeric(od[c], errors="coerce").fillna(0) * 0.01
        if "putt" in lc:
            score = score + pd.to_numeric(od[c], errors="coerce").fillna(0) * 0.005
        if "bogey" in lc and ("avoid" in lc or "avo" in lc):
            score = score + pd.to_numeric(od[c], errors="coerce").fillna(0) * 0.005
    od["OD_Score"] = score
    od = od.sort_values("OD_Score", ascending=False).head(10)

    # format
    win["WinProb%"] = pct01_to_100(win["WinProb"])
    top10["Top10Prob%"] = pct01_to_100(top10["Top10Prob"])
    od["WinProb%"] = pct01_to_100(od["WinProb"])
    od["Top10Prob%"] = pct01_to_100(od["Top10Prob"])
    od["OD_Score"] = pd.to_numeric(od["OD_Score"], errors="coerce").round(4)

    return {"win": win, "top10": top10, "od": od, "model": used_model}, None

# ==========================================================
# TRACKER (hit rate day/week/month/year)
# ==========================================================
TRACK_FILE = "tracker.csv"

def init_tracker():
    if "tracker" not in st.session_state:
        if os.path.exists(TRACK_FILE):
            try:
                st.session_state["tracker"] = pd.read_csv(TRACK_FILE)
            except Exception:
                st.session_state["tracker"] = pd.DataFrame()
        else:
            st.session_state["tracker"] = pd.DataFrame()

def save_tracker(df):
    try:
        df.to_csv(TRACK_FILE, index=False)
    except Exception:
        pass

def to_dt(x):
    return pd.to_datetime(x, errors="coerce")

def window_filters(df: pd.DataFrame):
    out = {}
    d = df.copy()
    d["LoggedAt_dt"] = to_dt(d["LoggedAt"])
    d = d.dropna(subset=["LoggedAt_dt"])

    now = pd.Timestamp.now()
    today = now.normalize()

    out["Today"] = d[d["LoggedAt_dt"] >= today]
    week_start = today - pd.Timedelta(days=today.weekday())
    out["This Week"] = d[d["LoggedAt_dt"] >= week_start]
    month_start = today.replace(day=1)
    out["This Month"] = d[d["LoggedAt_dt"] >= month_start]
    year_start = today.replace(month=1, day=1)
    out["This Year"] = d[d["LoggedAt_dt"] >= year_start]
    return out

def hit_rate_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df[df["Status"] == "Graded"].copy()
    if g.empty:
        return pd.DataFrame()

    g["Result"] = g["Result"].fillna("").astype(str)
    g = g[g["Result"].isin(["W", "L", "P", "N/A"])].copy()
    if g.empty:
        return pd.DataFrame()

    g["Wins"] = (g["Result"] == "W").astype(int)
    g["Losses"] = (g["Result"] == "L").astype(int)
    g["Pushes"] = (g["Result"] == "P").astype(int)
    g["NA"] = (g["Result"] == "N/A").astype(int)
    g["Picks"] = 1

    agg = g.groupby(["Scope"], dropna=False).agg(
        Picks=("Picks", "sum"),
        Wins=("Wins", "sum"),
        Losses=("Losses", "sum"),
        Pushes=("Pushes", "sum"),
        NA=("NA", "sum"),
    ).reset_index()

    agg["HitRate"] = np.where(agg["Picks"] > 0, agg["Wins"] / agg["Picks"], 0.0)
    agg["HitRate%"] = (agg["HitRate"] * 100).round(1)
    agg.insert(0, "Window", label)
    return agg.sort_values(["HitRate", "Wins"], ascending=False)

# ==========================================================
# Pages
# ==========================================================
def render_game_lines():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0, key="gl_sport")
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1, key="gl_bet_type")
    top_n = st.slider("Top picks (by Edge)", 2, 10, 5, key="gl_topn")

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No +EV game line bets right now."))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader(f"{sport} — {bet_type} (DK/FD) — +EV ONLY")
    st.caption("Ranked by Edge. **Contradictions removed** across books and within a book (absolute rule).")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Outcome"]
    if "Line" in top.columns and top["Line"].notna().any():
        cols += ["Line"]
    cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]

    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    st.markdown("</div>", unsafe_allow_html=True)

def render_props():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0, key="pr_sport")
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=0, key="pr_prop")
    top_n = st.slider("Top picks (by Edge)", 2, 10, 5, key="pr_topn")
    max_events_scan = st.slider("Events to scan (usage control)", 1, 12, 6, key="pr_scan")

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No +EV props returned for DK/FD on scanned events."))
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader(f"{sport} — Player Props ({prop_label}) — +EV ONLY")
    st.caption("Ranked by Edge. **No contradictions** for a player/market across books or within a book.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"]
    if "Line" in top.columns and top["Line"].notna().any():
        cols += ["Line"]
    cols += ["BestPrice", "⭐ BestBook", "YourProb%", "Implied%", "Edge%", "EV"]

    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    bar_prob(chart, "Label", "YourProb%", "Your Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    st.markdown("</div>", unsafe_allow_html=True)

def render_pga():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if not DATAGOLF_KEY.strip():
        st.info('PGA is hidden until DATAGOLF_KEY exists in Secrets as DATAGOLF_KEY="...".')
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.subheader("PGA — Advanced Models (DataGolf)")
    st.caption("Course fit + course history + current form + SG:T2G + putting + bogey avoidance (when available).")

    out, err = build_pga_tables()
    if err:
        st.warning(err)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(f"**Model used:** `{out.get('model','')}`")

    st.markdown("### Top 10 — Win")
    st.dataframe(out["win"][["Player", "WinProb%"]], use_container_width=True, hide_index=True)

    st.markdown("### Top 10 — Top 10 Finish")
    st.dataframe(out["top10"][["Player", "Top10Prob%"]], use_container_width=True, hide_index=True)

    st.markdown("### One-and-Done — Top Options")
    st.dataframe(out["od"][["Player", "WinProb%", "Top10Prob%", "OD_Score"]], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

def render_tracker():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Tracker — Hit Rate (Day / Week / Month / Year)")
    st.caption("Track picks and grade them. Hit rate summaries auto-calc by time window.")

    init_tracker()
    df_tr = st.session_state.get("tracker", pd.DataFrame()).copy()

    if df_tr.empty:
        df_tr = pd.DataFrame(columns=[
            "LoggedAt", "Scope", "Sport", "Market", "Event", "Outcome", "Line", "BestBook", "BestPrice",
            "Edge", "Status", "Result"
        ])

    st.markdown("### Add pick")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        scope = st.selectbox("Scope", ["GameLine", "Prop", "PGA"], index=0)
        sport = st.text_input("Sport", value="")
        market = st.text_input("Market", value="")
    with c2:
        event = st.text_input("Event / Tournament", value="")
        outcome = st.text_input("Outcome / Player", value="")
        line = st.text_input("Line (optional)", value="")
    with c3:
        bestbook = st.text_input("BestBook", value="")
        bestprice = st.text_input("BestPrice", value="")
        edge = st.text_input("Edge (decimal)", value="")

    if st.button("Add to Tracker"):
        new_row = {
            "LoggedAt": datetime.now().isoformat(),
            "Scope": scope,
            "Sport": sport,
            "Market": market,
            "Event": event,
            "Outcome": outcome,
            "Line": line,
            "BestBook": bestbook,
            "BestPrice": bestprice,
            "Edge": pd.to_numeric(edge, errors="coerce"),
            "Status": "Pending",
            "Result": "",
        }
        df_tr = pd.concat([df_tr, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["tracker"] = df_tr
        save_tracker(df_tr)
        st.success("Added.")

    st.markdown("### Picks")
    st.dataframe(df_tr.tail(300), use_container_width=True, hide_index=True)

    st.markdown("### Grade picks")
    st.caption("Set Status=Graded and Result=W/L/P/N/A for tracking.")
    edited = st.data_editor(df_tr, use_container_width=True, num_rows="dynamic", key="tracker_editor")

    if st.button("Save Tracker"):
        st.session_state["tracker"] = edited
        save_tracker(edited)
        st.success("Saved.")

    df_final = st.session_state["tracker"].copy()

    st.markdown("### Hit Rate Summary (Today / Week / Month / Year)")
    windows = window_filters(df_final)
    tables = []
    for label, sub in windows.items():
        t = hit_rate_summary(sub, label=label)
        if not t.empty:
            tables.append(t)

    if not tables:
        st.info("No graded picks yet in these windows.")
    else:
        summary_all = pd.concat(tables, ignore_index=True)
        st.dataframe(
            summary_all[["Window", "Scope", "Picks", "Wins", "Losses", "Pushes", "NA", "HitRate%"]],
            use_container_width=True,
            hide_index=True
        )

    st.markdown("### Trend (by Week)")
    b = df_final.copy()
    b["LoggedAt_dt"] = to_dt(b["LoggedAt"])
    b = b.dropna(subset=["LoggedAt_dt"])
    iso = b["LoggedAt_dt"].dt.isocalendar()
    b["Period"] = (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2))

    g = b[b["Status"] == "Graded"].copy()
    if not g.empty:
        g["Wins"] = (g["Result"] == "W").astype(int)
        g["Picks"] = g["Result"].isin(["W", "L", "P", "N/A"]).astype(int)
        trend = g.groupby(["Period", "Scope"], dropna=False).agg(Picks=("Picks", "sum"), Wins=("Wins", "sum")).reset_index()
        trend["HitRate"] = np.where(trend["Picks"] > 0, trend["Wins"] / trend["Picks"], 0.0)

        fig = plt.figure()
        for sc in trend["Scope"].dropna().unique():
            dsc = trend[trend["Scope"] == sc]
            plt.plot(dsc["Period"].astype(str), dsc["HitRate"], marker="o", label=str(sc))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.title("Hit Rate by Week")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No graded picks to chart yet.")

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# Router
# ==========================================================
if mode == "Game Lines":
    render_game_lines()
elif mode == "Player Props":
    render_props()
elif mode == "PGA":
    render_pga()
else:
    render_tracker()
