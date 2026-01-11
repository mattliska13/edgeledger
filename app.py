# app.py — EdgeLedger Dashboard
# ✅ Separate API keys + separate call paths:
#   1) Game Lines (Odds API /odds) — cached daily
#   2) Player Props (Odds API /events -> /events/{id}/odds) — cached daily
#   3) PGA (DataGolf) — cached daily
#
# ✅ Books: DraftKings + FanDuel only
# ✅ American odds
# ✅ Best price across books + implied prob + devig “true prob” + EV
# ✅ Auto-ranked Top 2–5 + Top 25 snapshot
# ✅ Robust error handling + debug logging
#
# Put keys in Streamlit secrets (recommended):
#   ODDS_API_KEY="d1a096c07dfb711c63560fcc7495fd0d"
#   DATAGOLF_API_KEY="909304744927252dd7a207f7dce4"

import os
import time
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ----------------------------
# Page / minimal theme polish
# ----------------------------
st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
h1,h2,h3 { letter-spacing:-0.02em; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
.block-container { padding-top: 1.25rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Dashboard")
st.caption("Game Lines + Player Props (DK/FD) + PGA (DataGolf) • Best Price • EV • Top Picks")

# ----------------------------
# API Keys (separate per source)
# ----------------------------
# Odds API key (NEW)
ODDS_DEFAULT = "d1a096c07dfb711c63560fcc7495fd0d"
# DataGolf key (you provided)
DATAGOLF_DEFAULT = "909304744927252dd7a207f7dce4"

def get_secret(name: str, default: str) -> str:
    # Order: Streamlit secrets -> env var -> default
    v = None
    try:
        v = st.secrets.get(name, None)
    except Exception:
        v = None
    if not v:
        v = os.getenv(name)
    return v or default

ODDS_API_KEY = get_secret("ODDS_API_KEY", ODDS_DEFAULT)
DATAGOLF_API_KEY = get_secret("DATAGOLF_API_KEY", DATAGOLF_DEFAULT)

# ----------------------------
# Constants
# ----------------------------
ODDS_BASE = "https://api.the-odds-api.com/v4"
BOOKMAKERS = ["draftkings", "fanduel"]

SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
}

GAME_LINE_MARKETS = ["h2h", "spreads", "totals"]

# IMPORTANT: Valid player prop keys (these avoid 422)
PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_pass_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

# ----------------------------
# Sidebar (minimal)
# ----------------------------
st.sidebar.markdown("## ⚙️ Controls")
tab_choice = st.sidebar.radio("View", ["Game Lines", "Player Props", "PGA (DataGolf)"], index=0)
debug_on = st.sidebar.toggle("Show Debug", value=False)

if tab_choice in ["Game Lines", "Player Props"]:
    sport_label = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
    sport_key = SPORTS[sport_label]
else:
    sport_label = "PGA"
    sport_key = None

top_n = st.sidebar.slider("Auto-ranked top picks", min_value=2, max_value=5, value=3, step=1)

if tab_choice == "Player Props":
    prop_label = st.sidebar.selectbox("Prop Type", list(PLAYER_PROP_MARKETS.keys()), index=0)
    prop_key = PLAYER_PROP_MARKETS[prop_label]
    max_events = st.sidebar.slider("Max events to scan", 1, 20, 10, step=1)
else:
    prop_label, prop_key, max_events = None, None, None

st.sidebar.markdown("---")
st.sidebar.caption("Books: DraftKings + FanDuel")
st.sidebar.caption("Caching: 24h (cuts API usage)")

# ----------------------------
# Core math helpers
# ----------------------------
def american_to_implied(odds: float) -> float:
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def payout_per_unit(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)

def ev_units(true_prob: float, odds: float) -> float:
    if pd.isna(true_prob) or pd.isna(odds):
        return np.nan
    profit = payout_per_unit(odds)
    return float(true_prob) * profit - (1.0 - float(true_prob))

def best_price_row(sub: pd.DataFrame) -> pd.Series:
    # best American odds: higher is better; for negative odds, closer to 0 is better (-105 beats -120)
    s = sub["Price"].astype(float).copy()
    score = s.copy()
    neg = score < 0
    score.loc[neg] = -score.loc[neg].abs()
    return sub.loc[int(score.idxmax())]

def pick_top(df: pd.DataFrame, n_min=2, n_max=5) -> pd.DataFrame:
    if df.empty:
        return df
    pos = df[df["EV"] > 0].copy()
    if pos.empty:
        return df.sort_values("EV", ascending=False).head(n_min)
    n = int(np.clip(len(pos), n_min, n_max))
    return pos.sort_values("EV", ascending=False).head(n)

# ----------------------------
# Debug buffer
# ----------------------------
debug_log = []
def dlog(obj):
    if debug_on:
        debug_log.append(obj)

# ----------------------------
# HTTP helpers
# ----------------------------
def get_json(url: str, params: dict, timeout=25):
    r = requests.get(url, params=params, timeout=timeout)
    try:
        payload = r.json()
    except Exception:
        payload = {"message": "Non-JSON response", "text": (r.text or "")[:2000]}
    meta = {
        "url": r.url,
        "status": r.status_code,
        "ok": r.ok,
        "error_code": payload.get("error_code") if isinstance(payload, dict) else None,
        "message": payload.get("message") if isinstance(payload, dict) else None,
        "payload_type": type(payload).__name__,
        "payload_len": len(payload) if isinstance(payload, list) else None,
    }
    return payload, meta

def is_api_error(payload) -> bool:
    return isinstance(payload, dict) and ("error_code" in payload or "message" in payload)

# ----------------------------
# Odds API calls (SEPARATED)
# ----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_game_lines(sport_key: str):
    url = f"{ODDS_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(GAME_LINE_MARKETS),
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }
    payload, meta = get_json(url, params)
    return payload, meta

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_events(sport_key: str):
    url = f"{ODDS_BASE}/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    payload, meta = get_json(url, params)
    return payload, meta

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_props_for_event(sport_key: str, event_id: str, market_key: str):
    # NOTE: This is the correct path for player props.
    url = f"{ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": market_key,
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }
    payload, meta = get_json(url, params)
    return payload, meta

# ----------------------------
# Normalizers
# ----------------------------
def normalize_game_lines(payload: list) -> pd.DataFrame:
    rows = []
    if not isinstance(payload, list):
        return pd.DataFrame()

    for ev in payload:
        if not isinstance(ev, dict):
            continue

        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}" if away and home else "Matchup"
        commence = ev.get("commence_time")

        for bm in ev.get("bookmakers", []) or []:
            bkey = bm.get("key")
            btitle = bm.get("title") or bkey
            if bkey not in BOOKMAKERS:
                continue

            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                if mkey not in GAME_LINE_MARKETS:
                    continue

                for o in m.get("outcomes", []) or []:
                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,
                            "Outcome": o.get("name"),
                            "Line": o.get("point", np.nan),
                            "Price": o.get("price", np.nan),
                            "Book": btitle,
                            "BookKey": bkey,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Implied"] = df["Price"].apply(american_to_implied)
    return df

def normalize_player_props(payloads: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in payloads:
        if not isinstance(ev, dict):
            continue

        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        matchup = f"{away} @ {home}" if away and home else "Matchup"
        commence = ev.get("commence_time")

        for bm in ev.get("bookmakers", []) or []:
            bkey = bm.get("key")
            btitle = bm.get("title") or bkey
            if bkey not in BOOKMAKERS:
                continue

            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                for o in m.get("outcomes", []) or []:
                    # Typical player prop outcome schema:
                    # - name: Player name
                    # - description: Over/Under OR Yes/No
                    # - point: line
                    # - price: american odds
                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,
                            "Player": o.get("name"),
                            "Side": o.get("description"),  # Over/Under/Yes/No (may be None for some markets)
                            "Line": o.get("point", np.nan),
                            "Price": o.get("price", np.nan),
                            "Book": btitle,
                            "BookKey": bkey,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df["Implied"] = df["Price"].apply(american_to_implied)
    return df

# ----------------------------
# Aggregation: best price + devig true prob + EV
# ----------------------------
def aggregate_best_price_ev(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    # Devig within (group + book) where two sides exist (Over/Under or two teams)
    if "Side" in df.columns:
        within = group_cols + ["BookKey"]
        sums = df.groupby(within)["Implied"].transform("sum")
        df = df.copy()
        df["Devig"] = np.where(sums > 0, df["Implied"] / sums, np.nan)
        df["ProbForConsensus"] = np.where(df["Devig"].notna(), df["Devig"], df["Implied"])
    else:
        df = df.copy()
        df["ProbForConsensus"] = df["Implied"]

    df["TrueProb"] = df.groupby(group_cols)["ProbForConsensus"].transform("mean")

    # Best odds per group across books
    best = df.groupby(group_cols, as_index=False).apply(best_price_row).reset_index(drop=True)

    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook", "Implied": "ImpliedBest"})
    best["EV"] = best.apply(lambda r: ev_units(r["TrueProb"], r["BestPrice"]), axis=1)

    # Simple labels
    best["Best"] = "🏆 " + best["BestBook"].astype(str)
    best = best.sort_values("EV", ascending=False)
    return best

# ----------------------------
# PGA (DataGolf) — separate API + key
# ----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def datagolf_outrights(market: str):
    # market: "win" or "top_10"
    url = "https://feeds.datagolf.com/betting-tools/outrights"
    params = {
        "tour": "pga",
        "market": market,             # win / top_10
        "odds_format": "american",
        "file_format": "json",
        "key": DATAGOLF_API_KEY,
    }
    payload, meta = get_json(url, params)
    return payload, meta

def normalize_datagolf(payload, label: str) -> pd.DataFrame:
    # DataGolf response formats can vary; attempt to extract list
    rows = []
    data_list = None
    if isinstance(payload, list):
        data_list = payload
    elif isinstance(payload, dict):
        for k in ("odds", "data", "results"):
            if isinstance(payload.get(k), list):
                data_list = payload[k]
                break
    if not data_list:
        return pd.DataFrame()

    for r in data_list:
        if not isinstance(r, dict):
            continue
        player = r.get("player_name") or r.get("name") or r.get("player")
        prob = r.get("dg_prob") or r.get("prob") or r.get("win_prob") or r.get("top10_prob") or r.get("probability")
        if player is None:
            continue

        rows.append(
            {
                "Market": label,
                "Player": player,
                "ModelProb": float(prob) if prob is not None and str(prob) != "nan" else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df

# ----------------------------
# Render views
# ----------------------------
if tab_choice == "Game Lines":
    st.subheader(f"🏈 {sport_label} — Game Lines (DraftKings + FanDuel)")

    with st.spinner("Loading game lines (cached daily)…"):
        payload, meta = oddsapi_game_lines(sport_key)
    dlog({"game_lines_meta": meta})

    if is_api_error(payload):
        st.error(f"Odds API error: {payload.get('message')} ({payload.get('error_code')})")
        if debug_on:
            st.json(payload)
            st.json(meta)
        st.stop()

    df = normalize_game_lines(payload)
    if df.empty:
        st.warning("No game line rows were normalized. (If you know games are happening, this is usually an API response/market availability issue.)")
        if debug_on:
            st.json(meta)
            st.write("Payload type:", type(payload).__name__)
            st.write("Payload len:", len(payload) if isinstance(payload, list) else None)
        st.stop()

    # group columns for a unique bet
    group_cols = ["Matchup", "Market", "Outcome", "Line"]
    agg = aggregate_best_price_ev(df, group_cols=group_cols)

    # Auto-ranked
    top = pick_top(agg, n_min=2, n_max=top_n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Book rows", f"{len(df):,}")
    c2.metric("Unique bets", f"{len(agg):,}")
    c3.metric("Top picks", f"{len(top):,}")

    st.markdown("### ✅ Auto-Ranked Top Bets by EV")
    show_cols = ["Matchup", "Market", "Outcome", "Line", "BestPrice", "Best", "TrueProb", "ImpliedBest", "EV"]
    show_cols = [c for c in show_cols if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### 📌 Snapshot — Top 25")
    snap = agg.head(25)
    show_cols = ["Matchup", "Market", "Outcome", "Line", "BestPrice", "Best", "TrueProb", "ImpliedBest", "EV"]
    show_cols = [c for c in show_cols if c in snap.columns]
    st.dataframe(snap[show_cols], use_container_width=True, hide_index=True)

elif tab_choice == "Player Props":
    st.subheader(f"🧩 {sport_label} — Player Props: {prop_label} (DraftKings + FanDuel)")
    st.caption("Props use a separate call path: /events → /events/{event_id}/odds (prevents game-line/prop interference).")

    with st.spinner("Loading events list (cached daily)…"):
        ev_payload, ev_meta = oddsapi_events(sport_key)
    dlog({"events_meta": ev_meta})

    if is_api_error(ev_payload):
        st.error(f"Odds API error (events): {ev_payload.get('message')} ({ev_payload.get('error_code')})")
        if debug_on:
            st.json(ev_payload)
            st.json(ev_meta)
        st.stop()

    if not isinstance(ev_payload, list) or len(ev_payload) == 0:
        st.warning("No events returned by /events right now.")
        if debug_on:
            st.json(ev_meta)
        st.stop()

    # IMPORTANT: don’t over-filter by time; just take what the API returned (up to max_events)
    event_ids = [e.get("id") for e in ev_payload if isinstance(e, dict) and e.get("id")]
    event_ids = event_ids[:max_events]

    if len(event_ids) == 0:
        st.warning("Events returned, but no event IDs found.")
        if debug_on:
            st.json(ev_payload[:2])
        st.stop()

    dlog({"scanning_event_ids": event_ids, "prop_key": prop_key})

    # Fetch each event's props (each request cached daily)
    prop_payloads = []
    prop_metas = []

    with st.spinner("Loading props per event (cached daily)…"):
        for eid in event_ids:
            p, m = oddsapi_props_for_event(sport_key, eid, prop_key)
            prop_metas.append(m)
            # Keep only successful payloads with bookmakers/markets
            if isinstance(p, dict) and (p.get("bookmakers") or p.get("markets")) and not is_api_error(p):
                prop_payloads.append(p)
            time.sleep(0.10)  # gentle pacing

    dlog({"prop_metas_sample": prop_metas[:3]})

    # If all came back 422, it means that market isn't available for those events/books yet,
    # OR the book doesn't offer it via API for the chosen sport.
    # (But we fixed the invalid-key issue; these are valid keys.)
    dfp = normalize_player_props(prop_payloads)
    if dfp.empty:
        st.warning("No player prop rows were normalized for this prop type from DK/FD right now.")
        if debug_on:
            st.write("Sample prop meta (first 3):")
            st.json(prop_metas[:3])
        st.stop()

    group_cols = ["Matchup", "Market", "Player", "Side", "Line"]
    agg = aggregate_best_price_ev(dfp, group_cols=group_cols)
    top = pick_top(agg, n_min=2, n_max=top_n)

    c1, c2, c3 = st.columns(3)
    c1.metric("Prop book rows", f"{len(dfp):,}")
    c2.metric("Unique props", f"{len(agg):,}")
    c3.metric("Top picks", f"{len(top):,}")

    st.markdown("### ✅ Auto-Ranked Top Props by EV")
    show_cols = ["Matchup", "Player", "Side", "Line", "BestPrice", "Best", "TrueProb", "ImpliedBest", "EV"]
    show_cols = [c for c in show_cols if c in top.columns]
    st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

    st.markdown("### 📌 Snapshot — Top 25 Props")
    snap = agg.head(25)
    show_cols = ["Matchup", "Player", "Side", "Line", "BestPrice", "Best", "TrueProb", "ImpliedBest", "EV"]
    show_cols = [c for c in show_cols if c in snap.columns]
    st.dataframe(snap[show_cols], use_container_width=True, hide_index=True)

else:
    st.subheader("⛳ PGA — DataGolf Picks (Win + Top-10)")
    pick_market = st.radio("Market", ["Win", "Top-10"], index=0, horizontal=True)
    dg_market = "win" if pick_market == "Win" else "top_10"

    with st.spinner("Loading DataGolf (cached daily)…"):
        payload, meta = datagolf_outrights(dg_market)
    dlog({"datagolf_meta": meta})

    if is_api_error(payload):
        st.error(f"DataGolf error: {payload.get('message')} ({payload.get('error_code')})")
        if debug_on:
            st.json(payload)
            st.json(meta)
        st.stop()

    df = normalize_datagolf(payload, label=pick_market)
    if df.empty:
        st.warning("No PGA rows normalized from DataGolf response.")
        if debug_on:
            st.json(meta)
            st.write("Payload type:", type(payload).__name__)
        st.stop()

    # We only have model probabilities here.
    # If you later add sportsbook odds for PGA, we can compute EV the same way.
    st.markdown("### DataGolf model probabilities")
    df = df.sort_values("ModelProb", ascending=False).head(25)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------
# Debug output
# ----------------------------
if debug_on:
    st.markdown("---")
    st.markdown("## 🔎 Debug")
    st.json(debug_log)
