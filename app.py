# EdgeLedger — Advanced Player Props Engine
# Clean, stable version with Best Price per Player/Line

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
st.title("EdgeLedger — Advanced Player Props Engine")

SPORT_MAP = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "UFC": "mma_mixed_martial_arts"
}

MARKET_MAP = {
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

def fetch_odds(sport, market):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/odds"
    r = requests.get(
        url,
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": MARKET_MAP[market],
            "oddsFormat": "american",
        },
        timeout=10
    )

    if r.status_code != 200:
        return []

    return r.json()

def normalize(raw):
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
                        "player": o.get("name"),
                        "line": o.get("point"),
                        "odds": o.get("price")
                    })

    return pd.DataFrame(rows)

def add_best_price(df):
    """
    Reduce to best odds per player + line
    """
    idx = df.groupby(["player", "line"])["odds"].idxmax()
    best = df.loc[idx].copy()

    best = best.rename(columns={
        "odds": "best_odds",
        "book": "best_book"
    })

    return best.reset_index(drop=True)

# ======================
# SIMPLE MODELS (PLACEHOLDERS)
# ======================
def project_receiving(targets):
    return targets * 0.65 * 11

def project_rushing(carries):
    return carries * 4.4

def anytime_td_prob(targets):
    return min(0.85, targets / 20)

# ======================
# UI CONTROLS
# ======================
sport = st.sidebar.selectbox("Sport", list(SPORT_MAP.keys()))
market = st.sidebar.selectbox("Prop Type", list(MARKET_MAP.keys()))

# ======================
# PIPELINE
# ======================
raw = fetch_odds(sport, market)
df = normalize(raw)

if df.empty:
    st.warning("No props available.")
    st.stop()

df = add_best_price(df)

# ======================
# VOLUME ASSUMPTIONS (TEMP)
# ======================
df["targets"] = np.random.randint(4, 10, len(df))
df["carries"] = np.random.randint(8, 20, len(df))

# ======================
# MODEL LOGIC
# ======================
if market == "Receiving Yards":
    df["projection"] = df["targets"].apply(project_receiving)
    df["model_prob"] = 0.55

elif market == "Rushing Yards":
    df["projection"] = df["carries"].apply(project_rushing)
    df["model_prob"] = 0.54

elif market == "Anytime TD":
    df["projection"] = np.nan
    df["model_prob"] = df["targets"].apply(anytime_td_prob)

else:
    df["projection"] = np.nan
    df["model_prob"] = 0.52

# ======================
# PRICING + EV
# ======================
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
st.subheader("Top 25 Player Props (Best Price, Highest EV)")

top_props = df.sort_values("ev", ascending=False).head(25)

st.dataframe(
    top_props[
        [
            "matchup",
            "player",
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

if st.button("Log Top Prop"):
    r = top_props.iloc[0]
    st.session_state.bets.append({
        "date": datetime.now(),
        "player": r["player"],
        "market": market,
        "line": r["line"],
        "odds": r["best_odds"],
        "book": r["best_book"]
    })

bets_df = pd.DataFrame(st.session_state.bets)

st.subheader("Logged Props")
st.dataframe(bets_df, use_container_width=True)

# ======================
# EXPORT
# ======================
if not bets_df.empty:
    fname = f"weekly_props_{datetime.now().date()}.csv"
    bets_df.to_csv(fname, index=False)
    st.success(f"Weekly props exported: {fname}")
