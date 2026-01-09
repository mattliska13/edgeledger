import streamlit as st
import pandas as pd
import requests

st.title("🏈 Game Lines")

# -----------------------------
# User Controls
# -----------------------------
sport = st.selectbox("Sport", ["NFL", "CFB"])
market = st.selectbox("Market", ["Spread", "Total", "Moneyline"])

# -----------------------------
# Odds API Setup
# -----------------------------
SPORT_MAP = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf"
}

MARKET_MAP = {
    "Spread": "spreads",
    "Total": "totals",
    "Moneyline": "h2h"
}

api_key = st.secrets["ODDS_API_KEY"]

url = f"https://api.the-odds-api.com/v4/sports/{SPORT_MAP[sport]}/odds"

params = {
    "apiKey": api_key,
    "regions": "us",
    "markets": MARKET_MAP[market],
    "oddsFormat": "american"
}

# -----------------------------
# Fetch Data
# -----------------------------
try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    games = response.json()
except Exception as e:
    st.error("Failed to load odds data")
    st.stop()

# -----------------------------
# Parse Odds
# -----------------------------
rows = []

for game in games:
    home = game["home_team"]
    away = game["away_team"]

    for book in game.get("bookmakers", []):
        book_name = book["title"]

        for market_data in book.get("markets", []):
            for outcome in market_data.get("outcomes", []):
                rows.append({
                    "Sport": sport,
                    "Book": book_name,
                    "Home Team": home,
                    "Away Team": away,
                    "Side": outcome.get("name"),
                    "Line": outcome.get("point"),
                    "Odds": outcome.get("price")
                })

df = pd.DataFrame(rows)

# -----------------------------
# Display
# -----------------------------
if df.empty:
    st.warning("No odds available")
else:
    st.subheader("Live Odds")
    st.dataframe(
        df.sort_values("Odds"),
        use_container_width=True
    )

    st.caption("Live odds powered by The Odds API")
