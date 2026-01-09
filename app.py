import streamlit as st
import pandas as pd
import numpy as np
import requests

API_KEY = st.secrets["ODDS_API_KEY"]

st.set_page_config(layout="wide")
st.title("EdgeLedger — Game Lines + Player Props (Best Price)")

# -------------------------
# Config
# -------------------------
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

# -------------------------
# Helpers
# -------------------------
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

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
        matchup = f"{g.get('away_team')} @ {g.get('home_team')}"
        for b in g.get("bookmakers", []):
            book = b.get("title")
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({
                        "entity": matchup,
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
    ev = raw[0]
    matchup = f"{ev.get('away_team')} @ {ev.get('home_team')}"
    for b in ev.get("bookmakers", []):
        book = b.get("title")
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
    """Return best odds & book per group"""
    idx = df.groupby(group_cols)["odds"].idxmax()
    best = df.loc[idx].copy()
    best = best.rename(columns={"odds": "best_odds", "book": "best_book"})
    return best.reset_index(drop=True)

# -------------------------
# Sidebar
# -------------------------
sport = st.sidebar.selectbox("Sport", list(SPORT_MAP.keys()))
scope = st.sidebar.radio("Scope", ["Game Lines", "Player Props"])

# -------------------------
# Game Lines
# -------------------------
if scope == "Game Lines":
    market = st.sidebar.selectbox("Market", list(GAME_MARKETS.keys()))
    raw = fetch_game_odds(sport, GAME_MARKETS[market])
    df = normalize_game(raw)

    if df.empty:
        st.warning("No game lines available.")
    else:
        df = best_price(df, ["entity", "market", "side"])
        df["implied_prob"] = df["best_odds"].apply(american_to_prob)
        df["model_prob"] = 0.52
        df["ev"] = (df["model_prob"] - df["implied_prob"]) * df["best_odds"].abs()

        st.subheader(f"{sport} — {market} (Top Lines by EV)")
        st.dataframe(
            df.sort_values("ev", ascending=False)[
                ["matchup", "entity", "side", "line", "best_odds", "best_book", "model_prob", "implied_prob", "ev"]
            ],
            use_container_width=True
        )

# -------------------------
# Player Props
# -------------------------
else:
    events = fetch_events(sport)
    if not events:
        st.warning("No upcoming events found.")
        st.stop()

    event_options = {f"{e['away_team']} @ {e['home_team']}": e["id"] for e in events}
    selected_event = st.sidebar.selectbox("Event", list(event_options.keys()))
    event_id = event_options[selected_event]

    raw = fetch_event_odds(sport, event_id, PLAYER_PROP_MARKETS)
    df = normalize_props(raw, event_id)

    if df.empty:
        st.warning("No player props found for this event.")
    else:
        df = best_price(df, ["entity", "market", "prop_side", "line"])
        df["implied_prob"] = df["best_odds"].apply(american_to_prob)
        df["model_prob"] = 0.55
        df["ev"] = (df["model_prob"] - df["implied_prob"]) * df["best_odds"].abs()

        st.subheader(f"{sport} — Player Props — {selected_event} (Top by EV)")
        st.dataframe(
            df.sort_values("ev", ascending=False)[
                ["matchup", "entity", "prop_side", "line", "best_odds", "best_book", "model_prob", "implied_prob", "ev"]
            ],
            use_container_width=True
        )
