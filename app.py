# app.py — EdgeLedger Dashboard (NFL/CFB)
# Game Lines + Player Props (separate API calls), Best Price + EV, Top 2–5 picks, robust edge-case handling
# Books: FanDuel + DraftKings only
#
# IMPORTANT: Your previous 422 errors were caused by INVALID market keys.
# Correct Odds API v4 player-prop keys include:
#   player_anytime_td, player_pass_yds, player_pass_tds, player_rush_yds, player_reception_yds, player_receptions
# Source: The Odds API betting markets list (player props section).

import os
import time
from datetime import datetime, timezone

import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Config / Theme
# ----------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide",
)

# Rename "app" in sidebar header area by using a custom title in the page and keeping sidebar minimal
st.markdown(
    """
<style>
/* Global font + vibe */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
h1, h2, h3 { letter-spacing: -0.02em; }

/* Make sidebar cleaner */
section[data-testid="stSidebar"] > div { padding-top: 12px; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { margin-top: 0.25rem; }

/* Subtle card feel */
.block-container { padding-top: 1.25rem; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Dashboard")
st.caption("Game lines + player props • Best price across DraftKings/FanDuel • EV ranking • Separate API calls")

# ----------------------------
# Secrets / API Key
# ----------------------------
# Prefer Streamlit secrets, fallback to env var, fallback to hard-coded (your provided key)
DEFAULT_KEY = "d1a096c07dfb711c63560fcc7495fd0d"
API_KEY = None

try:
    API_KEY = st.secrets.get("ODDS_API_KEY", None)
except Exception:
    API_KEY = None

if not API_KEY:
    API_KEY = os.getenv("ODDS_API_KEY") or DEFAULT_KEY

# ----------------------------
# Constants
# ----------------------------
SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
}

BOOKMAKERS = ["draftkings", "fanduel"]  # only these two as requested

# Correct (valid) player prop market keys for NFL/NCAAF (subset you requested)
PLAYER_PROP_GROUPS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_pass_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_reception_yds",
    "Receptions": "player_receptions",
}

GAME_LINE_MARKETS = ["h2h", "spreads", "totals"]

BASE_URL = "https://api.the-odds-api.com/v4"


# ----------------------------
# Helpers: odds/prob/EV
# ----------------------------
def american_to_implied(odds: float) -> float:
    """Implied probability from American odds (includes vig)."""
    if odds is None or pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def implied_to_american(p: float) -> float:
    """Convert probability to American odds (not used for UI, but handy)."""
    if p <= 0 or p >= 1:
        return np.nan
    if p >= 0.5:
        return - (p / (1 - p)) * 100
    return ((1 - p) / p) * 100


def payout_multiplier_from_american(odds: float) -> float:
    """Profit per $1 stake."""
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def ev_per_dollar(true_prob: float, american_odds: float) -> float:
    """
    Expected profit per $1 stake, given a "true" probability and American odds.
    EV = p*profit - (1-p)*1
    """
    if true_prob is None or pd.isna(true_prob) or pd.isna(american_odds):
        return np.nan
    profit = payout_multiplier_from_american(american_odds)
    return float(true_prob) * profit - (1.0 - float(true_prob))


def safe_get(dct, key, default=None):
    return dct.get(key, default) if isinstance(dct, dict) else default


# ----------------------------
# Debug logger
# ----------------------------
def log_debug(state_list, obj):
    ts = datetime.now().strftime("%H:%M:%S")
    state_list.append({"t": ts, "msg": obj})


# ----------------------------
# API Calls (separate + cached)
# ----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 1 daily call per sport for game lines
def api_fetch_game_lines(sport_key: str) -> dict:
    url = f"{BASE_URL}/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(GAME_LINE_MARKETS),
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }
    r = requests.get(url, params=params, timeout=20)
    return {"status": r.status_code, "ok": r.ok, "url": r.url, "payload": r.json() if r.content else None}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 1 daily call per sport for events list
def api_fetch_events(sport_key: str) -> dict:
    url = f"{BASE_URL}/sports/{sport_key}/events"
    params = {"apiKey": API_KEY}
    r = requests.get(url, params=params, timeout=20)
    return {"status": r.status_code, "ok": r.ok, "url": r.url, "payload": r.json() if r.content else None}


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)  # 1 daily call per sport+market for props (scoped by events)
def api_fetch_props_for_events(sport_key: str, event_ids: tuple, market_key: str) -> dict:
    """
    Calls event-odds endpoint per event.
    NOTE: This is still multiple HTTP requests, but it's cached for 24h, so it won't re-burn credits.
    """
    out = []
    for eid in event_ids:
        url = f"{BASE_URL}/sports/{sport_key}/events/{eid}/odds"
        params = {
            "apiKey": API_KEY,
            "regions": "us",
            "markets": market_key,
            "oddsFormat": "american",
            "bookmakers": ",".join(BOOKMAKERS),
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            payload = r.json() if r.content else None
            out.append(
                {
                    "event_id": eid,
                    "status": r.status_code,
                    "ok": r.ok,
                    "url": r.url,
                    "payload": payload,
                }
            )
        except Exception as e:
            out.append({"event_id": eid, "status": None, "ok": False, "url": url, "payload": {"error": str(e)}})
        time.sleep(0.15)  # gentle pacing
    return {"market": market_key, "results": out}


# ----------------------------
# Normalizers
# ----------------------------
def normalize_game_lines(events_payload) -> pd.DataFrame:
    """
    From /odds (featured markets) -> one row per outcome per book per event.
    """
    rows = []
    if not isinstance(events_payload, list):
        return pd.DataFrame()

    for g in events_payload:
        if not isinstance(g, dict):
            continue

        event_id = g.get("id")
        home = g.get("home_team")
        away = g.get("away_team")
        matchup = f"{away} @ {home}"
        commence = g.get("commence_time")

        for bm in g.get("bookmakers", []) or []:
            bkey = bm.get("key")
            btitle = bm.get("title") or bkey
            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                for o in m.get("outcomes", []) or []:
                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,             # h2h / spreads / totals
                            "Outcome": o.get("name"),   # team name or Over/Under
                            "Side": o.get("name"),
                            "Line": o.get("point", np.nan),
                            "Price": o.get("price", np.nan),
                            "Book": btitle,
                            "BookKey": bkey,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # normalize types
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    return df.dropna(subset=["Price"])


def normalize_player_props(props_results: dict) -> pd.DataFrame:
    """
    From /events/{eventId}/odds (non-featured markets) -> one row per player outcome per book per event.
    """
    rows = []

    results = props_results.get("results", []) if isinstance(props_results, dict) else []
    for item in results:
        if not isinstance(item, dict):
            continue
        if not item.get("ok"):
            continue

        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue

        event_id = payload.get("id") or item.get("event_id")
        home = payload.get("home_team")
        away = payload.get("away_team")
        matchup = f"{away} @ {home}"
        commence = payload.get("commence_time")

        for bm in payload.get("bookmakers", []) or []:
            bkey = bm.get("key")
            btitle = bm.get("title") or bkey
            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                for o in m.get("outcomes", []) or []:
                    # In player props, outcomes typically include:
                    # - name: player name
                    # - description: "Over"/"Under" or "Yes"/"No" depending on market
                    # - point: line (yards/receptions/tds)
                    # - price: american odds
                    player = o.get("name")
                    desc = o.get("description")  # Over/Under/Yes/No
                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,
                            "Player": player,
                            "Side": desc if desc is not None else o.get("name"),
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

    # Guard against missing player names (some markets can use team-based naming)
    if "Player" not in df.columns:
        df["Player"] = None

    return df


# ----------------------------
# Pricing + EV (best across books)
# ----------------------------
def compute_best_price_and_ev(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """
    df is per-book rows. We:
      1) Compute implied prob per row
      2) Build a "consensus true prob" using average de-vigged probabilities (where possible)
      3) Identify best price for each selection across books
      4) EV uses true_prob (consensus) and best_price
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["Implied"] = work["Price"].apply(american_to_implied)

    # Consensus true prob:
    # For two-sided markets (spreads/totals or Over/Under props), we can de-vig within each book
    # by normalizing implied probs across the two outcomes with same (event, market, line, entity).
    # If grouping doesn't find both sides, fallback to simple average implied (still useful).

    group_for_devig = key_cols.copy()
    # We need "Side" present; if not, fallback
    if "Side" not in work.columns:
        work["TrueProb"] = work["Implied"]
    else:
        # De-vig within bookmaker when both sides exist
        within_cols = group_for_devig + ["BookKey"]
        sums = work.groupby(within_cols)["Implied"].transform("sum")
        work["DevigProb"] = np.where(sums > 0, work["Implied"] / sums, np.nan)

        # Consensus probability across books
        # Prefer de-vig if available, else implied
        work["ProbForConsensus"] = np.where(work["DevigProb"].notna(), work["DevigProb"], work["Implied"])
        work["TrueProb"] = work.groupby(group_for_devig)["ProbForConsensus"].transform("mean")

    # Pick best price per selection:
    # For bets, "best" is max American odds (more positive / less negative is better payout).
    # Example: -105 is better than -115; +120 is better than +110.
    best = (
        work.sort_values("Price", ascending=False)
        .groupby(group_for_devig, as_index=False)
        .first()
        .rename(columns={"Price": "BestPrice", "Book": "BestBook"})
    )

    # Attach EV
    best["EV_$1"] = best.apply(lambda r: ev_per_dollar(r.get("TrueProb", np.nan), r.get("BestPrice", np.nan)), axis=1)
    best["EV_%"] = best["EV_$1"] * 100.0

    # Pretty labels
    best["Best"] = "🏆 " + best["BestBook"].astype(str)

    # Sort high-to-low EV
    best = best.sort_values("EV_$1", ascending=False)

    return best


# ----------------------------
# Sidebar Controls
# ----------------------------
debug = st.sidebar.toggle("Show debug", value=False)

sport_label = st.sidebar.selectbox("Sport", list(SPORTS.keys()), index=0)
sport_key = SPORTS[sport_label]

scope = st.sidebar.radio("Scope", ["Game Lines", "Player Props"], index=0)

prop_label = None
prop_key = None
if scope == "Player Props":
    prop_label = st.sidebar.selectbox("Prop Type", list(PLAYER_PROP_GROUPS.keys()), index=0)
    prop_key = PLAYER_PROP_GROUPS[prop_label]

# Auto-ranked top picks count
top_n = st.sidebar.slider("Auto-ranked top picks", min_value=2, max_value=5, value=3, step=1)

st.sidebar.markdown("---")
st.sidebar.caption("Books: DraftKings + FanDuel only")

# Debug buffer
debug_log = []
if debug:
    log_debug(debug_log, f"API_KEY loaded: {'YES' if bool(API_KEY) else 'NO'}")
    log_debug(debug_log, f"Using sport_key={sport_key}, scope={scope}, prop_key={prop_key}")


# ----------------------------
# Main Load
# ----------------------------
# GAME LINES (single daily call per sport)
if scope == "Game Lines":
    with st.spinner("Loading game lines (cached daily)…"):
        resp = api_fetch_game_lines(sport_key)

    if debug:
        log_debug(debug_log, {"endpoint": "odds(game_lines)", "status": resp["status"], "ok": resp["ok"], "url": resp["url"]})

    payload = resp.get("payload")

    # Handle API errors gracefully
    if isinstance(payload, dict) and payload.get("error_code"):
        st.error(f"API error: {payload.get('message')} ({payload.get('error_code')})")
        if debug:
            st.json(payload)
        st.stop()

    df_books = normalize_game_lines(payload)

    if df_books.empty:
        st.warning("No game line rows were normalized from the API response.")
        if debug:
            st.write("Raw payload type:", type(payload).__name__)
            st.json(payload if isinstance(payload, dict) else {"events_len": len(payload) if isinstance(payload, list) else None})
        st.stop()

    # Build best-price rows and EV
    # Keys for a unique bet selection: EventID + Market + Outcome (+ Line for spreads/totals)
    key_cols = ["EventID", "Matchup", "Market", "Outcome", "Line"]
    best = compute_best_price_and_ev(df_books, key_cols=key_cols)

    # Drop any weird rows
    best = best.dropna(subset=["BestPrice", "TrueProb"], how="any")

    # Clean display
    disp = best.copy()
    disp["CommenceTime"] = disp.get("CommenceTime", None)
    disp = disp.rename(
        columns={
            "Market": "Type",
            "Outcome": "Selection",
            "Line": "Line",
            "BestPrice": "Price (Best)",
            "BestBook": "Best Book",
            "TrueProb": "True Prob",
        }
    )

    # Reorder columns (safely)
    cols = [c for c in ["Matchup", "Type", "Selection", "Line", "Price (Best)", "Best", "True Prob", "EV_%"] if c in disp.columns]
    disp = disp[cols]

    # Summary + Top picks
    st.subheader(f"EdgeLedger — Game Lines ({sport_label})")

    c1, c2, c3 = st.columns(3)
    c1.metric("Events (books rows)", f"{len(df_books):,}")
    c2.metric("Unique bets (best-priced)", f"{len(best):,}")
    c3.metric("Top picks shown", f"{top_n}")

    st.markdown("### Top Bets Ranked by EV")
    top_picks = disp.head(top_n)
    st.dataframe(top_picks, use_container_width=True)

    st.markdown("### Snapshot — Top 25 Best Picks")
    st.dataframe(disp.head(25), use_container_width=True)

# PLAYER PROPS (events call + per-event odds call; cached daily)
else:
    # 1) Get events (cached daily)
    with st.spinner("Loading events (cached daily)…"):
        ev_resp = api_fetch_events(sport_key)

    if debug:
        log_debug(debug_log, {"endpoint": "events", "status": ev_resp["status"], "ok": ev_resp["ok"], "url": ev_resp["url"]})

    ev_payload = ev_resp.get("payload")

    if isinstance(ev_payload, dict) and ev_payload.get("error_code"):
        st.error(f"API error: {ev_payload.get('message')} ({ev_payload.get('error_code')})")
        if debug:
            st.json(ev_payload)
        st.stop()

    if not isinstance(ev_payload, list) or len(ev_payload) == 0:
        st.warning("No upcoming events returned for this sport right now.")
        if debug:
            st.write("Events payload type:", type(ev_payload).__name__)
            st.json(ev_payload if isinstance(ev_payload, dict) else {"events_len": len(ev_payload) if isinstance(ev_payload, list) else None})
        st.stop()

    # Take a handful of upcoming events (to control calls)
    # You can raise this later; caching prevents repeated credit burn.
    event_ids = tuple([x.get("id") for x in ev_payload if isinstance(x, dict) and x.get("id")][:12])

    if debug:
        log_debug(debug_log, {"events_returned": len(ev_payload), "event_ids_scanned": list(event_ids), "prop_market": prop_key})

    if not event_ids:
        st.warning("Events were returned but no event IDs could be extracted.")
        if debug:
            st.json(ev_payload[:3])
        st.stop()

    # 2) Pull props per event (cached daily by sport+market+event_ids)
    with st.spinner(f"Loading player props ({prop_label}) (cached daily)…"):
        props_resp = api_fetch_props_for_events(sport_key, event_ids, prop_key)

    # If everything 422s or empty, explain clearly (and show debug)
    results = props_resp.get("results", [])
    ok_count = sum(1 for r in results if isinstance(r, dict) and r.get("ok"))
    bad_count = sum(1 for r in results if isinstance(r, dict) and not r.get("ok"))

    if debug:
        log_debug(
            debug_log,
            {
                "endpoint": "event_odds(props)",
                "market": prop_key,
                "events_queried": len(event_ids),
                "ok_events": ok_count,
                "failed_events": bad_count,
                "sample_fail": next((r for r in results if isinstance(r, dict) and not r.get("ok")), None),
            },
        )

    # Normalize
    df_props_books = normalize_player_props(props_resp)

    # If empty, show why (without spamming)
    if df_props_books.empty:
        st.warning("No player props available for this prop type (or no books offering it for these events).")
        if debug:
            # show compact failure diagnostics
            fails = [
                {
                    "event_id": r.get("event_id"),
                    "status": r.get("status"),
                    "ok": r.get("ok"),
                    "url": r.get("url"),
                    "payload": r.get("payload"),
                }
                for r in results[:5]
                if isinstance(r, dict)
            ]
            st.markdown("**Debug sample (first 5 event calls):**")
            st.json(fails)
        st.stop()

    # Compute best price + EV
    # Unique prop selection key: EventID + Market + Player + Side + Line
    key_cols = ["EventID", "Matchup", "Market", "Player", "Side", "Line"]
    best_props = compute_best_price_and_ev(df_props_books, key_cols=key_cols)
    best_props = best_props.dropna(subset=["BestPrice", "TrueProb"], how="any")

    # Display
    st.subheader(f"EdgeLedger — Player Props ({sport_label})")
    st.caption(f"Prop Type: {prop_label} • Market key: `{prop_key}` • Best price across DK/FD")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prop rows (books)", f"{len(df_props_books):,}")
    c2.metric("Unique props (best-priced)", f"{len(best_props):,}")
    c3.metric("Top picks shown", f"{top_n}")

    disp = best_props.rename(
        columns={
            "Market": "Type",
            "Player": "Player",
            "Side": "Side",
            "Line": "Line",
            "BestPrice": "Price (Best)",
            "TrueProb": "True Prob",
        }
    )

    cols = [c for c in ["Matchup", "Player", "Side", "Line", "Price (Best)", "Best", "True Prob", "EV_%"] if c in disp.columns]
    disp = disp[cols]

    st.markdown("### Top Bets Ranked by EV")
    st.dataframe(disp.head(top_n), use_container_width=True)

    st.markdown("### Snapshot — Top 25 Best Picks")
    st.dataframe(disp.head(25), use_container_width=True)

# ----------------------------
# Debug output
# ----------------------------
if debug:
    st.markdown("---")
    st.markdown("## 🔎 Debug log")
    st.json(debug_log)
