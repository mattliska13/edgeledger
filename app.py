# app.py — EdgeLedger Dashboard (NFL / CFB / CBB game lines + NFL player props + PGA DataGolf)
# - Separate API calls for Game Lines vs Player Props (no cross-impact)
# - Only DraftKings + FanDuel (per your request)
# - Robust empty checks + debug panel
# - Best price + implied prob + EV + auto-ranked Top 2–5 bets
# - Player prop dropdown (Anytime TD, Pass Yds, Pass TDs, Rush Yds, Rec Yds, Receptions)
# - PGA uses DataGolf (win + top-10) with your key

import os
import re
import json
import time
from datetime import datetime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Page / Theme
# -----------------------------
st.set_page_config(
    page_title="Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Global typography */
html, body, [class*="css"]  {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}

/* Nice title */
h1, h2, h3 {
  letter-spacing: -0.02em;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.title("📈 EdgeLedger — Dashboard")
st.caption("Game Lines + Player Props (DK/FD) + PGA (DataGolf) • Best Price • EV • Top Bets")

# -----------------------------
# Keys (prefer secrets, fallback to provided)
# -----------------------------
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "6a5d08e7c2407da6fb95b86ad9619bf0")
DATAGOLF_KEY = st.secrets.get("DATAGOLF_KEY", "909304744927252dd7a207f7dce4")

# -----------------------------
# Constants
# -----------------------------
SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "CBB": "basketball_ncaab",
    "PGA": "pga",  # DataGolf tour code
}

BOOKMAKERS = ["draftkings", "fanduel"]  # only these

GAME_LINE_MARKETS = ["h2h", "spreads", "totals"]

# Odds API player prop market keys you requested
NFL_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Passing Yards": "player_passing_yds",
    "Pass TDs": "player_pass_tds",
    "Rushing Yards": "player_rushing_yds",
    "Receiving Yards": "player_receiving_yds",
    "Receptions": "player_receptions",
}

# -----------------------------
# Helpers
# -----------------------------
def now_et_iso():
    # Streamlit Cloud runs UTC; this is only for labels/debug
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def safe_get_json(resp: requests.Response):
    try:
        return resp.json()
    except Exception:
        return {"_non_json": True, "status_code": resp.status_code, "text": resp.text[:5000]}


def american_to_implied(odds: float) -> float:
    """Implied probability from American odds (no vig removal)."""
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return np.nan
    odds = float(odds)
    if odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def payout_per_unit(odds: float) -> float:
    """Profit (not total return) per 1 unit stake."""
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    return 100.0 / abs(odds)


def devig_two_way(p1: float, p2: float) -> tuple[float, float]:
    """Simple proportional devig for two-outcome markets."""
    if any(pd.isna([p1, p2])):
        return (np.nan, np.nan)
    s = p1 + p2
    if s <= 0:
        return (np.nan, np.nan)
    return (p1 / s, p2 / s)


def expected_value_units(model_prob: float, odds: float) -> float:
    """EV in units for a 1-unit stake."""
    if pd.isna(model_prob) or pd.isna(odds):
        return np.nan
    win_profit = payout_per_unit(odds)
    lose = 1.0
    return model_prob * win_profit - (1.0 - model_prob) * lose


def pick_top_n_by_ev(df: pd.DataFrame, min_n=2, max_n=5) -> pd.DataFrame:
    if df.empty or "EV" not in df.columns:
        return df
    pos = df[df["EV"] > 0].copy()
    if len(pos) == 0:
        # still show a small set (best of the bad) so UI isn't empty
        return df.sort_values("EV", ascending=False).head(min_n)
    n = int(np.clip(len(pos), min_n, max_n))
    return pos.sort_values("EV", ascending=False).head(n)


def clean_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z\s\-\.']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Sidebar controls (minimal)
# -----------------------------
st.sidebar.markdown("## ⚙️ Controls")
sport_label = st.sidebar.selectbox("Sport", ["NFL", "CFB", "CBB", "PGA"], index=0)

# Keep sidebar simple: only scope + prop type if needed
if sport_label == "PGA":
    scope = st.sidebar.selectbox("Scope", ["PGA Picks"], index=0)
else:
    scope = st.sidebar.selectbox("Scope", ["Game Lines", "Player Props"], index=0)

prop_label = None
if scope == "Player Props":
    # For now: only NFL props are wired (Odds API props are inconsistent for many sports/books)
    if sport_label != "NFL":
        st.sidebar.info("Player Props are currently enabled for NFL only (DK/FD). Switch sport to NFL.")
    prop_label = st.sidebar.selectbox("Player Prop Type", list(NFL_PROP_MARKETS.keys()), index=0)

debug_on = st.sidebar.toggle("Show Debug", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Bookmakers: DraftKings + FanDuel")
st.sidebar.caption(f"Last refresh: {now_et_iso()}")

# -----------------------------
# API: Odds API (Game Lines)
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_fetch_game_lines(sport_key: str) -> tuple[list, dict]:
    """
    Single call for game lines (daily cached).
    Uses /v4/sports/{sport_key}/odds with markets h2h,spreads,totals and bookmakers DK/FD.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": ",".join(GAME_LINE_MARKETS),
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        ok = r.ok
        payload = safe_get_json(r)
        meta = {
            "endpoint": "odds(game_lines)",
            "url": url,
            "status": r.status_code,
            "ok": ok,
            "params": {k: (v if k != "apiKey" else "***") for k, v in params.items()},
            "payload_type": type(payload).__name__,
            "payload_len": (len(payload) if isinstance(payload, list) else None),
            "error": payload if isinstance(payload, dict) and ("message" in payload or "error_code" in payload) else None,
        }
        return (payload if isinstance(payload, list) else [], meta)
    except Exception as e:
        return ([], {"endpoint": "odds(game_lines)", "error": str(e), "ok": False})


def normalize_game_lines(raw_events: list) -> pd.DataFrame:
    rows = []
    for ev in raw_events or []:
        if not isinstance(ev, dict):
            continue
        event_id = ev.get("id")
        home = ev.get("home_team")
        away = ev.get("away_team")
        commence = ev.get("commence_time")
        matchup = f"{away} @ {home}" if away and home else (ev.get("sport_title") or "Event")

        for bm in ev.get("bookmakers", []) or []:
            bm_key = bm.get("key")
            bm_title = bm.get("title") or bm_key
            if bm_key not in BOOKMAKERS:
                continue

            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                if mkey not in GAME_LINE_MARKETS:
                    continue

                for out in m.get("outcomes", []) or []:
                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,  # h2h/spreads/totals
                            "Outcome": out.get("name"),  # team / Over / Under
                            "Line": out.get("point", np.nan),
                            "Price": out.get("price", np.nan),
                            "BookKey": bm_key,
                            "Book": bm_title,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # enforce numeric
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")

    # Implied
    df["Implied"] = df["Price"].apply(american_to_implied)

    return df


def best_price_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple books into one row per (Matchup, Market, Outcome, Line) with best odds.
    For American odds:
      - Positive: higher is better
      - Negative: closer to 0 is better (e.g., -105 > -120)
    """
    if df.empty:
        return df

    gcols = ["Matchup", "Market", "Outcome", "Line"]

    def best_idx(sub: pd.DataFrame) -> int:
        prices = sub["Price"].astype(float)
        # Convert to comparable "goodness" score:
        # positives: keep
        # negatives: -abs(odds) so -105 > -120
        score = prices.copy()
        neg = score < 0
        score.loc[neg] = -score.loc[neg].abs()
        return int(score.idxmax())

    idx = df.groupby(gcols, dropna=False, sort=False).apply(best_idx)
    best = df.loc[idx.values].copy()

    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook", "BookKey": "BestBookKey", "Implied": "ImpliedBest"})
    # Keep a readable label for type
    best["Type"] = best["Market"].map({"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}).fillna(best["Market"])

    # Remove vig (two-way markets) to create a "model_prob" from market consensus
    # We’ll devig by pairing outcomes within each (Matchup, Market, Line) group.
    best["ModelProb"] = np.nan

    for (matchup, market, line), sub in best.groupby(["Matchup", "Market", "Line"], dropna=False):
        if len(sub) < 2:
            # cannot devig one-sided
            best.loc[sub.index, "ModelProb"] = sub["ImpliedBest"].values
            continue

        # For totals, outcomes are Over/Under (two outcomes). For spreads, two teams.
        # If more than 2 rows (rare), normalize proportionally.
        implied = sub["ImpliedBest"].values.astype(float)
        s = np.nansum(implied)
        if s > 0:
            probs = implied / s
        else:
            probs = implied
        best.loc[sub.index, "ModelProb"] = probs

    # EV
    best["EV"] = best.apply(lambda r: expected_value_units(r["ModelProb"], r["BestPrice"]), axis=1)

    # Helpful “Best” badge (always true at aggregate level)
    best["⭐ Best Book"] = "⭐"

    # Sort for display
    best = best.sort_values("EV", ascending=False)
    return best


# -----------------------------
# API: Odds API (Player Props) — separate path
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_fetch_events(sport_key: str) -> tuple[list, dict]:
    """
    Fetch events list once (daily cached).
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    params = {"apiKey": ODDS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=25)
        payload = safe_get_json(r)
        meta = {
            "endpoint": "events",
            "url": url,
            "status": r.status_code,
            "ok": r.ok,
            "params": {k: (v if k != "apiKey" else "***") for k, v in params.items()},
            "payload_type": type(payload).__name__,
            "payload_len": (len(payload) if isinstance(payload, list) else None),
            "error": payload if isinstance(payload, dict) and ("message" in payload or "error_code" in payload) else None,
        }
        return (payload if isinstance(payload, list) else [], meta)
    except Exception as e:
        return ([], {"endpoint": "events", "error": str(e), "ok": False})


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def oddsapi_fetch_props_for_event(sport_key: str, event_id: str, prop_market_key: str) -> tuple[dict, dict]:
    """
    Fetch props for ONE event (daily cached).
    This avoids the 422 you were hitting by never calling /odds with invalid markets like "player_props".
    We only request ONE valid market key per call, and only DK/FD.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": prop_market_key,
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        payload = safe_get_json(r)
        meta = {
            "endpoint": "event_odds(props)",
            "url": url,
            "status": r.status_code,
            "ok": r.ok,
            "event_id": event_id,
            "market": prop_market_key,
            "params": {k: (v if k != "apiKey" else "***") for k, v in params.items()},
            "payload_type": type(payload).__name__,
        }
        return (payload if isinstance(payload, dict) else {}, meta)
    except Exception as e:
        return ({}, {"endpoint": "event_odds(props)", "error": str(e), "ok": False, "event_id": event_id})


def normalize_player_props(event_payloads: list[dict]) -> pd.DataFrame:
    """
    Normalize event odds payloads for player props.

    Outcome schema varies by market; we defensively extract:
      - Player: outcome.description OR outcome.participant OR outcome.name (fallback)
      - Side: outcome.name (Over/Under/Yes/No)
      - Line: outcome.point
      - Price: outcome.price
    """
    rows = []
    for payload in event_payloads:
        if not isinstance(payload, dict):
            continue

        home = payload.get("home_team")
        away = payload.get("away_team")
        matchup = f"{away} @ {home}" if away and home else "Event"
        commence = payload.get("commence_time")
        event_id = payload.get("id")

        for bm in payload.get("bookmakers", []) or []:
            bm_key = bm.get("key")
            bm_title = bm.get("title") or bm_key
            if bm_key not in BOOKMAKERS:
                continue

            for m in bm.get("markets", []) or []:
                mkey = m.get("key")
                for out in m.get("outcomes", []) or []:
                    price = out.get("price", np.nan)
                    point = out.get("point", np.nan)

                    side = out.get("name")  # Over/Under or Yes/No
                    player = out.get("description") or out.get("participant")

                    # Fallbacks (some markets use name as player)
                    if not player:
                        # If side is Over/Under/Yes/No, name isn't player; but in some feeds it might be.
                        if isinstance(side, str) and side.lower() not in {"over", "under", "yes", "no"}:
                            player = side
                            side = None

                    # Ultimate fallback
                    if not player:
                        player = out.get("name")

                    rows.append(
                        {
                            "EventID": event_id,
                            "Matchup": matchup,
                            "CommenceTime": commence,
                            "Market": mkey,
                            "Player": player,
                            "Side": side,
                            "Line": point,
                            "Price": price,
                            "BookKey": bm_key,
                            "Book": bm_title,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Line"] = pd.to_numeric(df["Line"], errors="coerce")
    df["Implied"] = df["Price"].apply(american_to_implied)

    # Clean player names for joining
    df["PlayerClean"] = df["Player"].apply(clean_name)

    return df


def aggregate_best_prop_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Best price across DK/FD for each unique prop:
      (Matchup, Market, Player, Side, Line)
    """
    if df.empty:
        return df

    gcols = ["Matchup", "Market", "Player", "Side", "Line"]

    def best_idx(sub: pd.DataFrame) -> int:
        prices = sub["Price"].astype(float)
        score = prices.copy()
        neg = score < 0
        score.loc[neg] = -score.loc[neg].abs()
        return int(score.idxmax())

    idx = df.groupby(gcols, dropna=False, sort=False).apply(best_idx)
    best = df.loc[idx.values].copy()

    best = best.rename(columns={"Price": "BestPrice", "Book": "BestBook", "BookKey": "BestBookKey", "Implied": "ImpliedBest"})
    best["⭐ Best Book"] = "⭐"

    # Build a market-consensus "ModelProb" by devigging the paired outcomes when possible
    best["ModelProb"] = np.nan
    for (matchup, market, player, line), sub in best.groupby(["Matchup", "Market", "Player", "Line"], dropna=False):
        # Many props have two sides: Over/Under or Yes/No
        implied = sub["ImpliedBest"].values.astype(float)
        s = np.nansum(implied)
        if s > 0:
            probs = implied / s
            best.loc[sub.index, "ModelProb"] = probs
        else:
            best.loc[sub.index, "ModelProb"] = sub["ImpliedBest"].values

    best["EV"] = best.apply(lambda r: expected_value_units(r["ModelProb"], r["BestPrice"]), axis=1)
    best = best.sort_values("EV", ascending=False)
    return best


# -----------------------------
# API: DataGolf (PGA) — Win + Top-10
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def datagolf_outrights(market: str) -> tuple[list, dict]:
    """
    DataGolf betting tools outrights endpoint:
    https://feeds.datagolf.com/betting-tools/outrights?tour=pga&market=win|top_10&odds_format=american&file_format=json&key=...
    """
    url = "https://feeds.datagolf.com/betting-tools/outrights"
    params = {
        "tour": "pga",
        "market": market,          # win / top_10
        "odds_format": "american",
        "file_format": "json",
        "key": DATAGOLF_KEY,
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        payload = safe_get_json(r)
        meta = {
            "endpoint": "datagolf_outrights",
            "url": url,
            "status": r.status_code,
            "ok": r.ok,
            "market": market,
            "payload_type": type(payload).__name__,
        }
        # payload format can vary; we’ll return list-ish content where possible
        if isinstance(payload, dict):
            # common pattern: {"odds": [...]} or {"data": [...]}
            for k in ["odds", "data", "results"]:
                if k in payload and isinstance(payload[k], list):
                    return payload[k], meta
            # otherwise empty
            return [], {**meta, "error": payload}
        return (payload if isinstance(payload, list) else []), meta
    except Exception as e:
        return [], {"endpoint": "datagolf_outrights", "error": str(e), "ok": False}


def normalize_datagolf(rows: list) -> pd.DataFrame:
    """
    DataGolf outrights rows typically include:
      - player_name
      - dg_prob (or similar)
      - books / odds by book
    We’ll defensively parse common keys and filter to DK/FD if present.
    """
    out = []
    for r in rows or []:
        if not isinstance(r, dict):
            continue

        player = r.get("player_name") or r.get("name") or r.get("player") or r.get("golfer")
        model_prob = r.get("dg_prob") or r.get("prob") or r.get("win_prob") or r.get("top10_prob") or r.get("probability")

        # Some payloads have a list of book odds objects
        books = r.get("books") or r.get("odds") or r.get("book_odds") or []
        if isinstance(books, dict):
            books = [books]

        # Filter to DK/FD (DataGolf book keys can vary; we match by name substring)
        best_price = None
        best_book = None

        for b in books or []:
            if not isinstance(b, dict):
                continue
            bname = (b.get("book") or b.get("sportsbook") or b.get("book_name") or "").lower()
            price = b.get("odds") or b.get("american_odds") or b.get("price")
            try:
                price = float(price)
            except Exception:
                price = np.nan

            is_dk = "draftkings" in bname or bname == "dk"
            is_fd = "fanduel" in bname or bname == "fd"
            if not (is_dk or is_fd):
                continue

            # Best for American odds
            if pd.isna(price):
                continue
            if best_price is None:
                best_price = price
                best_book = "DraftKings" if is_dk else "FanDuel"
            else:
                # compare
                def score(o):
                    return o if o > 0 else -abs(o)

                if score(price) > score(best_price):
                    best_price = price
                    best_book = "DraftKings" if is_dk else "FanDuel"

        out.append(
            {
                "Player": player,
                "ModelProb": float(model_prob) if model_prob is not None and str(model_prob) != "nan" else np.nan,
                "BestBook": best_book,
                "BestPrice": best_price,
            }
        )

    df = pd.DataFrame(out).dropna(subset=["Player"])
    if df.empty:
        return df

    df["Implied"] = df["BestPrice"].apply(lambda x: american_to_implied(x) if pd.notna(x) else np.nan)
    df["EV"] = df.apply(lambda r: expected_value_units(r["ModelProb"], r["BestPrice"]) if pd.notna(r["BestPrice"]) else np.nan, axis=1)
    df["⭐ Best Book"] = np.where(df["BestBook"].notna(), "⭐", "")
    df = df.sort_values("EV", ascending=False)
    return df


# -----------------------------
# UI Rendering
# -----------------------------
if debug_on:
    st.info("Debug is ON. You’ll see API metadata + small samples where available.")

# -----------------------------
# GAME LINES
# -----------------------------
if scope == "Game Lines" and sport_label in ["NFL", "CFB", "CBB"]:
    sport_key = SPORTS[sport_label]

    with st.spinner("Loading game lines (daily cached)…"):
        raw, meta = oddsapi_fetch_game_lines(sport_key)

    if debug_on:
        st.subheader("🔎 Debug — Game Lines API")
        st.code(json.dumps(meta, indent=2))
        if isinstance(raw, list) and len(raw) > 0:
            st.caption("Sample event (truncated):")
            st.code(json.dumps(raw[0], indent=2)[:4000])

    df_raw = normalize_game_lines(raw)
    if df_raw.empty:
        st.warning("No game line rows were normalized from the API response.")
        st.stop()

    df_best = best_price_aggregate(df_raw)
    if df_best.empty:
        st.warning("No best-price game line rows available after aggregation.")
        st.stop()

    # Auto-ranked top 2–5
    top_bets = pick_top_n_by_ev(df_best, min_n=2, max_n=5)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader(f"🏈 {sport_label} — Game Lines (DK/FD)")
        st.write("**Auto-Ranked Top Bets (2–5)** — ranked by EV (units per 1u stake)")
        show_cols = ["Type", "Matchup", "Outcome", "Line", "BestPrice", "BestBook", "ModelProb", "ImpliedBest", "EV", "⭐ Best Book"]
        show_cols = [c for c in show_cols if c in top_bets.columns]
        st.dataframe(top_bets[show_cols], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("📊 Snapshot — Top 25 by EV")
        snap = df_best.sort_values("EV", ascending=False).head(25)
        show_cols = ["Type", "Matchup", "Outcome", "Line", "BestPrice", "BestBook", "ModelProb", "ImpliedBest", "EV", "⭐ Best Book"]
        show_cols = [c for c in show_cols if c in snap.columns]
        st.dataframe(snap[show_cols], use_container_width=True, hide_index=True)

# -----------------------------
# PLAYER PROPS (NFL only)
# -----------------------------
elif scope == "Player Props":
    if sport_label != "NFL":
        st.warning("Player Props are currently enabled for NFL only. Switch Sport to NFL.")
        st.stop()

    sport_key = SPORTS["NFL"]
    prop_market_key = NFL_PROP_MARKETS.get(prop_label)

    st.subheader(f"🧩 NFL — Player Props: {prop_label} (DK/FD)")
    st.caption("Props are loaded via separate endpoints: events → event odds. This prevents 422 errors from invalid market calls.")

    # Keep monthly credits sane: default small cap, still daily cached.
    max_events = st.slider("Max events to scan (daily cached)", 1, 10, 3, help="Higher scans more games but increases API usage. Cached daily.")

    with st.spinner("Loading NFL events (daily cached)…"):
        events, meta_e = oddsapi_fetch_events(sport_key)

    if debug_on:
        st.subheader("🔎 Debug — Events API")
        st.code(json.dumps(meta_e, indent=2))
        st.caption(f"Events returned: {len(events) if isinstance(events, list) else 0}")

    if not isinstance(events, list) or len(events) == 0:
        st.warning("No events returned. Try again later.")
        st.stop()

    # Filter to upcoming ~7 days to avoid stale
    upcoming = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ct = e.get("commence_time")
        try:
            # parse ISO
            dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except Exception:
            dt = None
        if dt is None:
            continue
        if dt < datetime.now(timezone.utc) - timedelta(hours=2):
            continue
        if dt > datetime.now(timezone.utc) + timedelta(days=7):
            continue
        upcoming.append(e)

    if len(upcoming) == 0:
        st.warning("No upcoming NFL events found in the next 7 days.")
        st.stop()

    # take a small set to limit calls
    upcoming = sorted(upcoming, key=lambda x: x.get("commence_time", ""))[:max_events]
    event_ids = [e.get("id") for e in upcoming if e.get("id")]

    if debug_on:
        st.caption(f"Scanning event_ids (count={len(event_ids)}): {event_ids}")

    # Fetch props per event (each call daily-cached)
    payloads = []
    metas = []
    with st.spinner("Loading player props (daily cached)…"):
        for eid in event_ids:
            payload, m = oddsapi_fetch_props_for_event(sport_key, eid, prop_market_key)
            metas.append(m)
            # Skip empties/errors safely
            if isinstance(payload, dict) and payload.get("bookmakers"):
                payloads.append(payload)

    if debug_on:
        st.subheader("🔎 Debug — Props API (per event)")
        st.code(json.dumps(metas[:3], indent=2)[:4000])
        st.caption(f"Prop payloads with bookmakers: {len(payloads)} / {len(event_ids)}")

    df_props_raw = normalize_player_props(payloads)
    if df_props_raw.empty:
        st.warning("No player props were normalized (DK/FD may not be offering this market for scanned events).")
        st.stop()

    df_props_best = aggregate_best_prop_price(df_props_raw)
    if df_props_best.empty:
        st.warning("No best-price player props available after aggregation.")
        st.stop()

    # Auto-ranked top 2–5 + snapshot top 25
    top_props = pick_top_n_by_ev(df_props_best, min_n=2, max_n=5)
    snap_props = df_props_best.sort_values("EV", ascending=False).head(25)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("✅ Auto-Ranked Top Props (2–5) by EV")
        show_cols = ["Matchup", "Player", "Side", "Line", "BestPrice", "BestBook", "ModelProb", "ImpliedBest", "EV", "⭐ Best Book"]
        show_cols = [c for c in show_cols if c in top_props.columns]
        st.dataframe(top_props[show_cols], use_container_width=True, hide_index=True)
    with c2:
        st.subheader("📌 Snapshot — Top 25 Props by EV")
        show_cols = ["Matchup", "Player", "Side", "Line", "BestPrice", "BestBook", "ModelProb", "ImpliedBest", "EV", "⭐ Best Book"]
        show_cols = [c for c in show_cols if c in snap_props.columns]
        st.dataframe(snap_props[show_cols], use_container_width=True, hide_index=True)

# -----------------------------
# PGA (DataGolf) — Win + Top-10
# -----------------------------
elif sport_label == "PGA":
    st.subheader("⛳ PGA — DataGolf Picks (DK/FD if available)")
    pick_market = st.selectbox("Pick type", ["Win", "Top-10"], index=0)
    dg_market = "win" if pick_market == "Win" else "top_10"

    with st.spinner("Loading DataGolf outrights (daily cached)…"):
        rows, meta = datagolf_outrights(dg_market)

    if debug_on:
        st.subheader("🔎 Debug — DataGolf API")
        st.code(json.dumps(meta, indent=2))
        st.caption(f"Rows returned: {len(rows)}")
        if len(rows) > 0:
            st.code(json.dumps(rows[0], indent=2)[:3500])

    df_golf = normalize_datagolf(rows)
    if df_golf.empty:
        st.warning("No PGA rows normalized. This usually means the DataGolf response format changed or no DK/FD odds were present.")
        st.stop()

    top5 = df_golf.dropna(subset=["EV"]).sort_values("EV", ascending=False).head(5)
    st.subheader("🏆 Top 5 Picks by EV")
    st.dataframe(top5[["Player", "BestPrice", "BestBook", "ModelProb", "Implied", "EV", "⭐ Best Book"]], use_container_width=True, hide_index=True)

    st.subheader("📌 Snapshot — Top 25 by EV")
    snap = df_golf.dropna(subset=["EV"]).sort_values("EV", ascending=False).head(25)
    st.dataframe(snap[["Player", "BestPrice", "BestBook", "ModelProb", "Implied", "EV", "⭐ Best Book"]], use_container_width=True, hide_index=True)

else:
    st.info("Select a Sport + Scope from the left.")
