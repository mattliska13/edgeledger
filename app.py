import streamlit as st
import pandas as pd
import requests
import logging

# -----------------------
# CONFIG & LOGGING
# -----------------------
st.set_page_config(page_title="EdgeLedger", layout="wide")
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

API_KEY = st.secrets.get("ODDS_API_KEY")
if not API_KEY:
    st.error("API_KEY not found in secrets.toml")
    st.stop()
logging.debug("API_KEY loaded successfully")

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def fetch_odds(sport_key, markets):
    """Fetch odds from The Odds API."""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }
    logging.debug(f"Fetching {sport_key} with markets={markets}")
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

def american_to_decimal(odds):
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def implied_probability(odds):
    return 1 / american_to_decimal(odds)

def calc_ev(row):
    """Calculate expected value based on best price."""
    prob = row["ImpliedProb"]
    price = row["DecimalOdds"]
    return round(prob * price - 1, 4)

def highlight_best(cell, col, df):
    """Highlight best bookmaker per line/outcome."""
    subset = df[df["Line"] == col]
    if not subset.empty:
        if cell == subset["DecimalOdds"].max():
            return "background-color: #b3ffb3"  # light green
    return ""

# -----------------------
# APP SIDEBAR
# -----------------------
st.sidebar.title("EdgeLedger — Sports Betting")
sport_name = st.sidebar.selectbox("Sport", ["NFL", "CFB", "UFC"])
scope = st.sidebar.selectbox("Scope", ["Game Lines", "Player Props"])

player_prop_type = None
if scope == "Player Props":
    if sport_name in ["NFL", "CFB"]:
        player_prop_type = st.sidebar.selectbox(
            "Player Prop Type",
            ["passing", "rushing", "receiving", "touchdowns"]
        )
    else:
        st.warning("Player props not available for this sport.")
        st.stop()

# -----------------------
# MAP SPORT KEYS
# -----------------------
SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_college_football",
    "UFC": "mma_ufc"
}
sport_key = SPORT_KEYS[sport_name]
logging.debug(f"Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")

# -----------------------
# BUILD MARKETS
# -----------------------
markets = []
if scope == "Game Lines":
    markets = ["h2h", "spreads", "totals"]
elif scope == "Player Props":
    if player_prop_type:
        markets = [f"player_props:{player_prop_type}"]

# -----------------------
# FETCH DATA
# -----------------------
try:
    events = fetch_odds(sport_key, markets)
    logging.debug(f"API returned {len(events)} events for {sport_name}")
except requests.HTTPError as e:
    st.error(f"HTTPError: {e}")
    st.stop()

# -----------------------
# NORMALIZE DATA
# -----------------------
rows = []
for event in events:
    event_name = event.get("home_team") + " vs " + event.get("away_team") if "home_team" in event else event.get("name", "")
    for market in event.get("bookmakers", []):
        for outcome in market.get("markets", []):
            for o in outcome.get("outcomes", []):
                decimal_odds = american_to_decimal(o["price"])
                rows.append({
                    "Event": event_name,
                    "Bookmaker": market["title"],
                    "Line": o.get("name", o.get("label", "")),
                    "DecimalOdds": decimal_odds,
                    "ImpliedProb": implied_probability(o["price"]),
                    "EV": calc_ev({
                        "DecimalOdds": decimal_odds,
                        "ImpliedProb": implied_probability(o["price"])
                    }),
                    "Outcome": o["name"]
                })

if not rows:
    st.warning("No normalized data available.")
    st.stop()

df = pd.DataFrame(rows)

# -----------------------
# TOP BETS RANKING
# -----------------------
top_bets = df.sort_values(by="EV", ascending=False).head(5)

# -----------------------
# DISPLAY
# -----------------------
st.title(f"EdgeLedger — {scope} ({sport_name})")
st.subheader("Top Bets Ranked by EV")

# Highlight best bookmaker
styled = top_bets.style.applymap(lambda x: highlight_best(x, top_bets["Line"], top_bets), subset=["DecimalOdds"])
st.dataframe(styled, use_container_width=True)

# Show all available lines for transparency
st.subheader("All Available Lines")
st.dataframe(df, use_container_width=True)
