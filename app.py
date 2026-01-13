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
/* Hide Streamlit chrome */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Sidebar theme */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%); }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

/* App container */
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1200px; }
@media (min-width: 1200px) { .block-container { max-width: 1600px; } }

/* Typography */
.big-title { font-size: 1.9rem; font-weight: 900; letter-spacing: -0.02em; margin: 0 0 0.2rem 0; }
.subtle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 0.35rem; }

/* Card */
.card { background: #0b1220; border: 1px solid rgba(148,163,184,0.18); border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; }

/* Pills */
.pill { display:inline-block; padding:0.18rem 0.55rem; border-radius:999px; background:rgba(255,255,255,0.08); margin-right:0.4rem; font-size:0.85rem; }
.small {font-size:0.85rem; color:#94a3b8;}
hr { border: none; border-top: 1px solid rgba(148,163,184,0.18); margin: 10px 0; }

/* Make dataframes/tables scroll on small screens */
div[data-testid="stDataFrame"] { width: 100%; }
div[data-testid="stDataFrame"] > div { overflow-x: auto !important; }

/* Tighten widgets */
div[data-testid="stSelectbox"], div[data-testid="stSlider"], div[data-testid="stRadio"], div[data-testid="stToggle"] { width: 100%; }

/* MOBILE */
@media (max-width: 768px) {
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .big-title { font-size: 1.35rem; }
  .subtle { font-size: 0.85rem; }
  .card { padding: 10px 10px; border-radius: 14px; }
  .stMarkdown p, .stCaption { font-size: 0.9rem; }
  canvas, svg, img { max-width: 100% !important; height: auto !important; }
  section[data-testid="stSidebar"] { min-width: 250px; }
}
@media (max-width: 420px) {
  .big-title { font-size: 1.2rem; }
  .pill { font-size: 0.75rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================
# Keys (Secrets first)
# =========================
def get_key(name: str, default: str = "") -> str:
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
# Odds helpers
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

def pct(series01):
    return (pd.to_numeric(series01, errors="coerce") * 100.0).round(2)

# =========================
# API Config
# =========================
ODDS_HOST = "https://api.the-odds-api.com/v4"
REGION = "us"
BOOKMAKERS = "draftkings,fanduel"   # only these two

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

# Correct market keys for The Odds API player props
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
st.sidebar.markdown("<div class='subtle'>Best Price • Actual Prob • Value • EV</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

mode = st.sidebar.radio("Mode", ["Best Bets (All)", "Game Lines", "Player Props", "PGA (optional)"], index=0)
debug = st.sidebar.checkbox("Show debug logs", value=False)

# Mobile stacking toggle
compact = st.sidebar.toggle("Mobile / Compact layout", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("<span class='pill'>Books: DK + FD</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<span class='pill'>Updated: {now_str()}</span>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.markdown("<div class='big-title'>EdgeLedger</div>", unsafe_allow_html=True)
st.caption("Responsive layout • No-contradictions • Best price across DK/FD • ‘Actual’ probability via no-vig • Value + EV ranking • Separate calls for Lines vs Props")

if not ODDS_API_KEY:
    st.error('Missing ODDS_API_KEY. Add it in Streamlit Secrets as ODDS_API_KEY="..."')
    st.stop()

# ==========================================================
# CACHING (daily-ish)
# ==========================================================
@st.cache_data(ttl=60*60*24)
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

@st.cache_data(ttl=60*60*24)
def fetch_events(sport_key: str):
    url = f"{ODDS_HOST}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    ok, status, payload, final_url = safe_get(url, params=params)
    return {"ok": ok, "status": status, "payload": payload, "url": final_url, "params": params}

@st.cache_data(ttl=60*60*24)
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

        for bm in ev.get("bookmakers", []) or []:
            book = bm.get("title") or bm.get("key")
            for mk in bm.get("markets", []) or []:
                mkey = mk.get("key")
                for out in mk.get("outcomes", []) or []:
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
    return df.dropna(subset=["Market", "Outcome", "Price"])

def normalize_props(event_payload):
    rows = []
    if not isinstance(event_payload, dict):
        return pd.DataFrame()

    away = event_payload.get("away_team")
    home = event_payload.get("home_team")
    matchup = f"{away} @ {home}"

    for bm in event_payload.get("bookmakers", []) or []:
        book = bm.get("title") or bm.get("key")
        for mk in bm.get("markets", []) or []:
            mkey = mk.get("key")
            for out in mk.get("outcomes", []) or []:
                player = out.get("name")
                side = out.get("description")  # Over/Under (may be blank for ATD)
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
                    "Price": price,
                    "Book": book
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df.dropna(subset=["Market", "Player", "Price"])

# ==========================================================
# Actual Prob + Best Price + No Contradictions
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

def estimate_actual_prob(df: pd.DataFrame, key_cols: list, book_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = add_implied(df)

    # no-vig per book grouping
    out_nv = compute_no_vig_two_way_within_book(out, group_cols_book=book_cols)

    nv_avg = out_nv.groupby(key_cols)["NoVigProb"].transform("mean")
    imp_avg = out_nv.groupby(key_cols)["Implied"].transform("mean")

    out_nv["ActualProb"] = np.where(pd.notna(nv_avg), nv_avg, imp_avg)
    out_nv["ActualProb"] = clamp01(pd.to_numeric(out_nv["ActualProb"], errors="coerce").fillna(out_nv["Implied"]))
    return out_nv

def best_price_and_value(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    out = out.dropna(subset=["Price"])

    idx = out.groupby(group_cols)["Price"].idxmax()
    best = out.loc[idx].copy()

    best = best.rename(columns={
        "Price": "BestPrice",
        "Book": "BestBook",
        "Implied": "BestImplied"
    })

    best["Value"] = (pd.to_numeric(best["ActualProb"], errors="coerce") - pd.to_numeric(best["BestImplied"], errors="coerce"))
    best["EV"] = (best["Value"] * 100.0)
    return best

def no_contradictions(df_best: pd.DataFrame, contradiction_key_cols: list) -> pd.DataFrame:
    if df_best.empty:
        return df_best

    out = df_best.copy()
    out["EV"] = pd.to_numeric(out["EV"], errors="coerce").fillna(-1e9)
    idx = out.groupby(contradiction_key_cols, dropna=False)["EV"].idxmax()
    out = out.loc[idx].sort_values("EV", ascending=False)
    return out

def add_no_bet_flag(df: pd.DataFrame, min_ev: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    out["NO BET"] = np.where(pd.to_numeric(out["EV"], errors="coerce").fillna(-999) <= min_ev, "NO BET", "")
    return out

# ==========================================================
# Game Lines Board
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
    contradiction_cols = ["Event", "Market"]

    if market_key in ["spreads", "totals"]:
        key_cols += ["Line"]
        book_cols += ["Line"]
        contradiction_cols += ["Line"]

    df_est = estimate_actual_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_value(df_est, group_cols=key_cols)
    df_best = no_contradictions(df_best, contradiction_key_cols=contradiction_cols)
    df_best = add_no_bet_flag(df_best, min_ev=0.0)

    df_best["ActualProb%"] = pct(df_best["ActualProb"])
    df_best["Implied%"] = pct(df_best["BestImplied"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    return df_best, {}

# ==========================================================
# Props Board
# ==========================================================
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

        time.sleep(0.08)

    if debug:
        st.json({"prop_calls": call_log})

    if not all_rows:
        return pd.DataFrame(), {"error": "No props returned for scanned events", "calls": call_log}

    df = pd.concat(all_rows, ignore_index=True)

    key_cols = ["Event", "Market", "Player", "Side"]
    book_cols = ["Event", "Market", "Player", "Book"]
    contradiction_cols = ["Event", "Market", "Player"]

    if "Line" in df.columns:
        key_cols += ["Line"]
        book_cols += ["Line"]
        contradiction_cols += ["Line"]

    df_est = estimate_actual_prob(df, key_cols=key_cols, book_cols=book_cols)
    df_best = best_price_and_value(df_est, group_cols=key_cols)

    # CONTRADICTION PREVENTION: keep only best side per Player+Line
    df_best = no_contradictions(df_best, contradiction_key_cols=contradiction_cols)
    df_best = add_no_bet_flag(df_best, min_ev=0.0)

    df_best["ActualProb%"] = pct(df_best["ActualProb"])
    df_best["Implied%"] = pct(df_best["BestImplied"])
    df_best["EV"] = pd.to_numeric(df_best["EV"], errors="coerce").round(2)

    return df_best, {}

# ==========================================================
# Charts (Percent axis)
# ==========================================================
def bar_prob(df, label_col, prob_col, title):
    if df.empty:
        return
    fig = plt.figure()
    vals = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0).values
    labs = df[label_col].astype(str).values
    plt.barh(labs, vals)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(100))
    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig)

# ==========================================================
# MAIN MODES
# ==========================================================
if mode == "Best Bets (All)":
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Layout that stacks on mobile when compact enabled
    if compact:
        colA = st.container()
        colB = st.container()
        colC = st.container()
    else:
        colA, colB, colC = st.columns([1.2, 1.2, 1])

    with colA:
        sport_lines = st.selectbox("Game Lines Sport", list(SPORT_KEYS_LINES.keys()), index=0)
        bet_type = st.selectbox("Game Lines Bet Type", list(GAME_MARKETS.keys()), index=1)
    with colB:
        sport_props = st.selectbox("Props Sport", list(SPORT_KEYS_PROPS.keys()), index=0)
        prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=1)
    with colC:
        top_n = st.slider("Top picks overall", 2, 10, 5)
        max_events_scan = st.slider("Props events to scan (usage control)", 1, 10, 5)

    df_lines, errL = build_game_lines_board(sport_lines, bet_type)
    df_props, errP = build_props_board(sport_props, prop_label, max_events_scan=max_events_scan)

    if df_lines.empty and errL:
        st.warning(f"Game Lines: {errL.get('error','No data')}")
    if df_props.empty and errP:
        st.warning(f"Props: {errP.get('error','No data')}")

    combined = []
    if not df_lines.empty:
        x = df_lines.copy()
        x["Category"] = f"{sport_lines} {bet_type}"
        x["Pick"] = x["Outcome"].astype(str)
        x["Entity"] = x["Event"].astype(str)
        x["LineOrProp"] = x.get("Line", "")
        combined.append(x)

    if not df_props.empty:
        x = df_props.copy()
        x["Category"] = f"{sport_props} {prop_label}"
        x["Pick"] = (x["Player"].astype(str) + " " + x["Side"].astype(str)).str.strip()
        x["Entity"] = x["Event"].astype(str)
        x["LineOrProp"] = x.get("Line", "")
        combined.append(x)

    if not combined:
        st.info("No data available right now (or props not posted yet on DK/FD).")
        st.stop()

    board = pd.concat(combined, ignore_index=True)
    board["EV"] = pd.to_numeric(board["EV"], errors="coerce").fillna(-9999)
    board = board.sort_values("EV", ascending=False)

    # Remove duplicates (already contradiction-clean inside each module)
    board = board.drop_duplicates(subset=["Category", "Entity", "Pick", "LineOrProp"], keep="first")

    top = board.head(int(top_n)).copy()
    top25 = board.head(25).copy()

    st.subheader("Top Bets (All Areas) — Ranked by EV")
    show_cols = ["Category", "Entity", "Pick", "LineOrProp", "BestPrice", "BestBook", "ActualProb%", "Implied%", "EV", "NO BET"]
    show_cols = [c for c in show_cols if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### Snapshot — Top 25 (All Areas)")
    show_cols2 = [c for c in show_cols if c in top25.columns]
    st.dataframe(top25[show_cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Game Lines":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_LINES.keys()), index=0)
    bet_type = st.selectbox("Bet Type", list(GAME_MARKETS.keys()), index=1)
    top_n = st.slider("Top picks (EV)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)

    df_best, err = build_game_lines_board(sport, bet_type)
    if df_best.empty:
        st.warning(err.get("error", "No game line data available right now."))
        st.stop()

    st.subheader(f"{sport} — {bet_type}")
    st.caption("ActualProb estimated from no-vig where both sides exist (DK/FD). BestPrice is best odds across DK/FD. No contradictions per line.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Outcome"] + (["Line"] if "Line" in top.columns else []) + ["BestPrice", "⭐ BestBook", "ActualProb%", "Implied%", "EV", "NO BET"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = chart["Outcome"].astype(str) + " | " + chart["Event"].astype(str)
    bar_prob(chart, "Label", "ActualProb%", "Actual Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Outcome"] + (["Line"] if "Line" in snap.columns else []) + ["BestPrice", "⭐ BestBook", "ActualProb%", "Implied%", "EV", "NO BET"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif mode == "Player Props":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    sport = st.selectbox("Sport", list(SPORT_KEYS_PROPS.keys()), index=0)
    prop_label = st.selectbox("Prop Type", list(PROP_MARKETS.keys()), index=1)
    top_n = st.slider("Top picks (EV)", 2, 10, 5)
    show_top25 = st.toggle("Show top 25 snapshot", value=True)
    max_events_scan = st.slider("Events to scan (usage control)", 1, 10, 5)

    df_best, err = build_props_board(sport, prop_label, max_events_scan=max_events_scan)
    if df_best.empty:
        st.warning(err.get("error", "No props returned for DK/FD on scanned events."))
        st.stop()

    st.subheader(f"{sport} — Player Props ({prop_label})")
    st.caption("ActualProb estimated from no-vig when both sides exist (Over/Under). BestPrice is best odds across DK/FD. No contradictions per player/line.")

    top = df_best.head(int(top_n)).copy()
    top["⭐ BestBook"] = "⭐ " + top["BestBook"].astype(str)

    cols = ["Event", "Player", "Side"] + (["Line"] if "Line" in top.columns else []) + ["BestPrice", "⭐ BestBook", "ActualProb%", "Implied%", "EV", "NO BET"]
    cols = [c for c in cols if c in top.columns]
    st.dataframe(top[cols], use_container_width=True, hide_index=True)

    st.markdown("#### Probability view (Top Picks)")
    chart = top.copy()
    chart["Label"] = (chart["Player"].astype(str) + " " + chart["Side"].astype(str)).str.strip()
    bar_prob(chart, "Label", "ActualProb%", "Actual Probability (Top Picks)")
    bar_prob(chart, "Label", "Implied%", "Implied Probability (Best Price)")

    if show_top25:
        st.markdown("### Snapshot — Top 25 (by EV)")
        snap = df_best.head(25).copy()
        snap["⭐ BestBook"] = "⭐ " + snap["BestBook"].astype(str)
        cols2 = ["Event", "Player", "Side"] + (["Line"] if "Line" in snap.columns else []) + ["BestPrice", "⭐ BestBook", "ActualProb%", "Implied%", "EV", "NO BET"]
        cols2 = [c for c in cols2 if c in snap.columns]
        st.dataframe(snap[cols2], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("PGA (Optional)")
    if not DATAGOLF_API_KEY:
        st.info('DATAGOLF_API_KEY not set. Add it in Streamlit Secrets as DATAGOLF_API_KEY="...". PGA is hidden until then.')
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    st.success("DATAGOLF_API_KEY detected. PGA module can be enabled next without impacting lines/props.")
    st.markdown("</div>", unsafe_allow_html=True)
