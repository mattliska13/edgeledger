# EdgeLedger — Advanced Betting Engine
# Supports Game Lines + Player Props with Independent Dropdowns

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# ======================
# CONFIG
# ======================
API_KEY = "6a5d08e7c2407da6fb95b86ad9619bf0"

st.set_page_config(layout="wide")
st.title("EdgeLedger — Advanced Betting Engine")

# ======================
# SPORT + MARKET MAPS
# ======================
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

PLAYER_PROP_MARKETS = {
    "Anytime TD": "player_anytime_td",
    "Receiving Yards": "player_receptions_yards",
    "Rushing Yards": "player_rushing_yards",
    "Passing Yards": "player_passing_yards"
}

BOOK_WEIGHTS = {
    "Pinnacle": 1.25,
    "Circa Sports": 1.2,
    "DraftKings": 1.0,
    "FanDuel": 1.0
}

# ======================
# HELPERS
# ======================
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)

def fetch_odds(sport, market_key):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/odds"
    r = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": market_key,
            "oddsFormat": "american"
        },
        timeout=10
    )
    return r.json() if r.status_code == 200 else []

def normalize(raw, is_player_prop=False):
    rows = []

    if not isinstance(raw, list):
        return pd.DataFrame()

    for g in raw:
        matchup = f"{g.get('away_team')} @ {g.get('home_team')}"

        for b in g.get("bookmakers", []):
            book = b.get("title")

            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append({
                        "matchup": matchup,
                        "book": book,
                        "entity": o.get("name") if is_player_prop else matchup,
                        "side": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price")
                    })

    return pd.DataFrame(rows)

def add_best_price(df):
    idx = df.groupby(["entity", "line", "side"])["odds"].idxmax()
    best = df.loc[idx].copy()

    best = best.rename(columns={
        "odds": "best_odds",
        "book": "best_book"
    })

    return best.reset_index(drop=True)

# ======================
# SIMPLE MODELS (PLACEHOLDERS)
# ======================
def game_model_prob():
    return 0.525  # replace later with efficiency-based model

def player_model_prob():
    return 0.55

# ======================
# SIDEBAR CONTROLS
# ======================
sport = st.sidebar.selectbox("Sport", list(SPORT_MAP.keys()))

bet_type = st.sidebar.radio(
    "Bet Type",
    ["Game Lines", "Player Props"]
)

if bet_type == "Game Lines":
    market_label = st.sidebar.selectbox("Game Market", list(GAME_MARKETS.keys()))
    market_key = GAME_MARKETS[market_label]
    is_player_prop = False

else:
    market_label = st.sidebar.selectbox("Player Prop", list(PLAYER_PROP_MARKETS.keys()))
    market_key = PLAYER_PROP_MARKETS[market_label]
    is_player_prop = True

# ======================
# DATA PIPELINE
# ======================
raw = fetch_odds(sport, market_key)
df = normalize(raw, is_player_prop=is_player_prop)

if df.empty:
    st.warning("No markets available.")
    st.stop()

df = add_best_price(df)

# ======================
# MODEL + EV
# ======================
if bet_type == "Game Lines":
    df["model_prob"] = game_model_prob()
else:
    df["model_prob"] = player_model_prob()

df["implied_prob"] = df["best_odds"].apply(american_to_prob)
df["book_weight"] = df["best_book"].map(BOOK_WEIGHTS).fillna(1.0)

df["ev"] = (
    (df["model_prob"] - df["implied_prob"])
    * df["book_weight"]
    * 100
)

# ======================
# DISPLAY
# ======================
st.subheader(f"Top 25 {bet_type} — {market_label}")

top = df.sort_values("ev", ascending=False).head(25)

st.dataframe(
    top[
        [
            "matchup",
            "entity",
            "side",
            "line",
            "best_odds",
            "best_book",
            "model_prob",
            "implied_prob",
            "ev"
        ]
    ],
    use_container_width=True
)

# ======================
# BET TRACKER
# ======================
if "bets" not in st.session_state:
    st.session_state.bets = []

if st.button("Log Top Bet"):
    r = top.iloc[0]
    st.session_state.bets.append({
        "date": datetime.now(),
        "sport": sport,
        "bet_type": bet_type,
        "market": market_label,
        "entity": r["entity"],
        "side": r["side"],
        "line": r["line"],
        "odds": r["best_odds"],
        "book": r["best_book"]
    })

bets_df = pd.DataFrame(st.session_state.bets)

st.subheader("Logged Bets")
st.dataframe(bets_df, use_container_width=True)

# ======================
# EXPORT
# ======================
if not bets_df.empty:
    fname = f"bets_{datetime.now().date()}.csv"
    bets_df.to_csv(fname, index=False)
    st.success(f"Exported: {fname}")
