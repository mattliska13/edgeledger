import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# -------------------------
# Config
# -------------------------
API_KEY = st.secrets["ODDS_API_KEY"]
st.write("DEBUG: API_KEY loaded successfully")
st.set_page_config(layout="wide")
st.title("EdgeLedger — Game Lines + Player Props (Best Price + EV)")

SPORT_MAP = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "UFC": "mma_mixed_martial_arts"
}

BOOK_WEIGHTS = {
    "DraftKings": 1.0,
    "FanDuel": 1.0,
    "Pinnacle": 1.2,
    "Circa Sports": 1.15
}

PLAYER_PROP_TYPES = {
    "Anytime TD": "player_pass_rush_reception_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_rec_yds"
}

# -------------------------
# Helper Functions
# -------------------------
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

def fetch_odds_api(url, params):
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json() if resp.status_code == 200 else []
        if not data:
            st.warning(f"No data returned from API for url: {url}")
        st.write(f"DEBUG API response ({url}):", data)  # debug log
        return data
    except Exception as e:
        st.error(f"API request failed: {e}")
        return []

def fetch_game_odds(sport):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/odds"
    # UFC only supports h2h
    markets = ["spreads", "totals", "h2h"] if sport != "UFC" else ["h2h"]
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }
    return fetch_odds_api(url, params)

def normalize_game(raw):
    rows = []
    if not raw:
        return pd.DataFrame()
    for g in raw:
        matchup = f"{g.get('away_team', 'Unknown')} @ {g.get('home_team', 'Unknown')}"
        bookmakers = g.get("bookmakers", [])
        if not bookmakers:
            continue  # skip empty
        for b in bookmakers:
            book = b.get("title", "Unknown Book")
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({
                        "entity": matchup,
                        "market": m.get("key"),
                        "side": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price"),
                        "book": book,
                        "matchup": matchup
                    })
    return pd.DataFrame(rows)

def fetch_events(sport):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/events"
    params = {"apiKey": API_KEY, "regions": "us"}
    return fetch_odds_api(url, params)

def fetch_event_props(sport, event_id, prop_market):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/events/{event_id}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": prop_market,
        "oddsFormat": "american"
    }
    return fetch_odds_api(url, params)

def normalize_props(raw):
    rows = []
    if not raw:
        return pd.DataFrame()
    if isinstance(raw, dict):
        raw = [raw]
    for ev in raw:
        matchup = f"{ev.get('away_team', 'Unknown')} @ {ev.get('home_team', 'Unknown')}"
        bookmakers = ev.get("bookmakers", [])
        if not bookmakers:
            continue
        for b in bookmakers:
            book = b.get("title", "Unknown Book")
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({
                        "entity": o.get("description", o.get("name")),
                        "market": m.get("key"),
                        "prop_side": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price"),
                        "book": book,
                        "matchup": matchup
                    })
    return pd.DataFrame(rows)

def best_price(df, group_cols):
    if df.empty:
        return df
    idx = df.groupby(group_cols)["odds"].idxmax()
    best = df.loc[idx].copy()
    best = best.rename(columns={"odds": "best_odds", "book": "best_book"})
    return best.reset_index(drop=True)

def display_df(df, scope):
    if df.empty:
        st.warning("No data available to display.")
        return
    df["implied_prob"] = df["best_odds"].apply(american_to_prob)
    df["model_prob"] = 0.52 if scope == "Game Lines" else 0.55
    df["ev"] = (df["model_prob"] - df["implied_prob"]) * df["best_odds"].abs()

    if scope == "Game Lines":
        cols = ["matchup", "entity", "side", "line", "best_odds", "best_book", "model_prob", "implied_prob", "ev"]
    else:
        cols = ["matchup", "entity", "prop_side", "line", "best_odds", "best_book", "model_prob", "implied_prob", "ev"]

    cols = [c for c in cols if c in df.columns]
    st.dataframe(df.sort_values("ev", ascending=False)[cols], use_container_width=True)

# -------------------------
# Sidebar
# -------------------------
sport = st.sidebar.selectbox("Sport", list(SPORT_MAP.keys()))
scope = st.sidebar.radio("Scope", ["Game Lines", "Player Props"])

# -------------------------
# Game Lines
# -------------------------
if scope == "Game Lines":
    raw = fetch_game_odds(sport)
    if not raw:
        st.stop()
    df = normalize_game(raw)
    if df.empty:
        st.warning("No normalized game lines available for this sport right now.")
        st.stop()
    df = best_price(df, ["entity", "market", "side"])
    st.subheader(f"{sport} — Game Lines (Top by EV)")
    display_df(df, scope)

    # Auto-Ranked Top 2-5
    top_n = min(max(2, int(len(df)*0.1)), 5)
    auto_bets = df.sort_values("ev", ascending=False).head(top_n)
    st.subheader(f"Auto-Ranked Top {top_n} Game Lines by EV")
    display_df(auto_bets, scope)

# -------------------------
# Player Props
# -------------------------
else:
    events = fetch_events(sport)
    if not events:
        st.warning("No upcoming events available for player props.")
        st.stop()

    event_options = {f"{e.get('away_team','Unknown')} @ {e.get('home_team','Unknown')}": e["id"] for e in events}
    selected_event = st.sidebar.selectbox("Event", list(event_options.keys()))
    event_id = event_options[selected_event]

    # Dropdown for prop type
    prop_type_label = st.sidebar.selectbox("Prop Type", list(PLAYER_PROP_TYPES.keys()))
    prop_market = [PLAYER_PROP_TYPES[prop_type_label]]  # API expects a list

    raw = fetch_event_props(sport, event_id, prop_market)
    if not raw:
        st.warning("No player props returned for selected event/prop type.")
        st.stop()

    df = normalize_props(raw)
    if df.empty:
        st.warning("No normalized player props available for this event/prop type.")
        st.stop()
    df = best_price(df, ["entity", "market", "prop_side", "line"])
    st.subheader(f"{sport} — Player Props — {selected_event} — {prop_type_label} (Top by EV)")
    display_df(df, scope)

    # Auto-Ranked Top 2-5 Player Props
    top_n = min(max(2, int(len(df)*0.1)), 5)
    auto_bets = df.sort_values("ev", ascending=False).head(top_n)
    st.subheader(f"Auto-Ranked Top {top_n} Player Props by EV")
    display_df(auto_bets, scope)
