import streamlit as st
import pandas as pd
import numpy as np
import requests

API_KEY = st.secrets["ODDS_API_KEY"]

st.set_page_config(layout="wide")
st.title("EdgeLedger — Game Lines + Player Props")

SPORT_MAP = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "UFC": "mma_mixed_martial_arts"
}

GAME_MARKETS = {
    "Spread": "spreads",
    "Total": "totals",
    "Moneyline": "h2h"
}

PLAYER_PROP_MARKETS = [
    "player_pass_tds",
    "player_rush_yds",
    "player_pass_rush_reception_tds",
    "player_pass_yds"
]

BOOK_WEIGHTS = {
    "DraftKings": 1.0,
    "FanDuel": 1.0,
    "Pinnacle": 1.2,
    "Circa Sports": 1.15
}


def fetch_game_odds(sport, market):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": market,
            "oddsFormat": "american"
        },
        timeout=10
    )
    return resp.json() if resp.status_code == 200 else []


def normalize_game(raw):
    rows = []
    for g in raw:
        matchup = f\"{g.get("away_team")} @ {g.get("home_team")}\"
        for b in g.get("bookmakers", []):
            book = b.get("title")
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({
                        "matchup": matchup,
                        "market": m.get("key"),
                        "side": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price"),
                        "book": book
                    })
    return pd.DataFrame(rows)


def fetch_events(sport):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/events"
    resp = requests.get(url, params={"apiKey": API_KEY, "regions": "us"}, timeout=10)
    return resp.json() if resp.status_code == 200 else []


def fetch_event_odds(sport, event_id, markets):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/events/{event_id}/odds"
    resp = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american"
        },
        timeout=10
    )
    return resp.json() if resp.status_code == 200 else []


def normalize_props(raw, event_id):
    rows = []
    if not raw:
        return pd.DataFrame()

    # raw should be a list with 1 event
    ev = raw[0]
    matchup = f\"{ev.get('away_team')} @ {ev.get('home_team')}\"

    for b in ev.get("bookmakers", []):
        book = b.get("title")
        for m in b.get("markets", []):
            for o in m.get("outcomes", []):
                rows.append({
                    "event_id": event_id,
                    "matchup": matchup,
                    "market": m.get("key"),
                    "player": o.get("description", o.get("name")),
                    "prop_side": o.get("name"),
                    "line": o.get("point"),
                    "odds": o.get("price"),
                    "book": book
                })
    return pd.DataFrame(rows)


# ==============================
# SIDEBAR
# ==============================

sport = st.sidebar.selectbox("Sport", list(SPORT_MAP.keys()))

bet_scope = st.sidebar.radio("Select Scope", ["Game Lines", "Player Props"])

# =========== Game Lines ===========
if bet_scope == "Game Lines":
    game_market = st.sidebar.selectbox("Market", list(GAME_MARKETS.keys()))

    raw_odds = fetch_game_odds(sport, GAME_MARKETS[game_market])
    df_games = normalize_game(raw_odds)

    if df_games.empty:
        st.warning("No game lines available.")
    else:
        df_games["implied_prob"] = df_games["odds"].apply(lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100))
        df_games["model_prob"] = 0.52
        df_games["ev"] = (df_games["model_prob"] - df_games["implied_prob"]) * df_games["odds"].abs()

        st.subheader(f"{sport} — {game_market}")
        st.dataframe(df_games, use_container_width=True)

# ========== Player Props ===========
else:
    st.sidebar.write("Select Event")

    events = fetch_events(sport)

    if not events:
        st.warning("No upcoming events found.")
        st.stop()

    event_options = {f\"{e['away_team']} @ {e['home_team']}\": e["id"] for e in events}
    selected_event = st.sidebar.selectbox("Event", list(event_options.keys()))

    event_id = event_options[selected_event]

    props_data = fetch_event_odds(sport, event_id, PLAYER_PROP_MARKETS)
    df_props = normalize_props(props_data, event_id)

    if df_props.empty:
        st.warning("No player props found for this event.")
    else:
        df_props["implied_prob"] = df_props["odds"].apply(lambda o: 100 / (o + 100) if o > 0 else -o / (-o + 100))
        df_props["model_prob"] = 0.55
        df_props["ev"] = (df_props["model_prob"] - df_props["implied_prob"]) * df_props["odds"].abs()

        st.subheader(f"{sport} — Props — {selected_event}")
        st.dataframe(df_props, use_container_width=True)

