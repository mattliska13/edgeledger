import streamlit as st
import pandas as pd
import requests
import numpy as np

st.title("🏈 Game Lines — Best Price & EV")

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
# Helper Functions
# -----------------------------
def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def implied_to_decimal(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def expected_value(model_p, odds):
    dec = implied_to_decimal(odds)
    return round((model_p * dec) - 1, 3)

# -----------------------------
# Fetch Data
# -----------------------------
try:
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    games = r.json()
except Exception:
    st.error("Failed to load odds")
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
                    "Matchup": f"{away} @ {home}",
                    "Side": outcome.get("name"),
                    "Line": outcome.get("point"),
                    "Odds": outcome.get("price"),
                    "Book": book_name
                })

df = pd.DataFrame(rows)

if df.empty:
    st.warning("No odds available")
    st.stop()

# -----------------------------
# Best Price Per Side
# -----------------------------
best = (
    df.sort_values("Odds", ascending=False)
      .groupby(["Matchup", "Side", "Line"], as_index=False)
      .first()
)

# -----------------------------
# Model Probability (Placeholder)
# Replace later with efficiency metrics
# -----------------------------
from modules.team_efficiency import matchup_probability

def calc_model_prob(row):
    away, home = row["Matchup"].split(" @ ")
    return matchup_probability(
        team_a=row["Side"],
        team_b=home if row["Side"] == away else away,
        home=(row["Side"] == home)
    )

best["Model_Prob"] = best.apply(calc_model_prob, axis=1)
)

best["Implied_Prob"] = best["Odds"].apply(american_to_implied)
best["EV"] = best.apply(lambda r: expected_value(r["Model_Prob"], r["Odds"]), axis=1)

best["Decision"] = np.where(best["EV"] > 0, "BET", "NO BET")

# -----------------------------
# Display Top 25 by EV
# -----------------------------
st.subheader("📈 Top 25 Bets by Expected Value")

top = (
    best.sort_values("EV", ascending=False)
        .head(25)
        .reset_index(drop=True)
)

st.dataframe(
    top.style.applymap(
        lambda x: "background-color: #d4f7d4" if isinstance(x, float) and x > 0 else ""
    ),
    use_container_width=True
)

st.caption("EV calculated using best available price across books")


