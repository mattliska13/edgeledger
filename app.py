import streamlit as st
import requests
import pandas as pd
from datetime import datetime

# ------------------------
# CONFIG
# ------------------------
st.set_page_config(page_title="EdgeLedger — Game Lines + Player Props", layout="wide")

# Load API key from secrets
API_KEY = st.secrets.get("ODDS_API_KEY")
if not API_KEY:
    st.error("API key not found. Please add it to Streamlit secrets.")
    st.stop()
st.sidebar.write("DEBUG: API_KEY loaded successfully")

# ------------------------
# UI: Sport + Scope + Player Prop Type
# ------------------------
sport_options = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaaf",
    "UFC": "mma_ufc"
}
sport_name = st.sidebar.selectbox("Sport", list(sport_options.keys()))
sport_key = sport_options[sport_name]

scope_options = ["Game Lines", "Player Props"]
scope = st.sidebar.selectbox("Scope", scope_options)

player_prop_type = None
if scope == "Player Props" and sport_name in ["NFL", "CFB"]:
    player_prop_type = st.sidebar.selectbox(
        "Player Prop Type",
        ["passing", "rushing", "receiving", "touchdowns"]
    )

st.write(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")

# ------------------------
# API CALL
# ------------------------
def fetch_odds(sport_key, markets):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# Determine which markets to request
markets = []
if scope == "Game Lines":
    markets = ["h2h", "spreads", "totals"]
elif scope == "Player Props":
    markets = ["player_props"]

st.write(f"DEBUG: Requesting markets: {markets}")

try:
    data = fetch_odds(sport_key, markets)
except requests.HTTPError as e:
    st.error(f"API request failed: {e}")
    st.stop()

st.write(f"DEBUG: API returned {len(data)} events for {sport_name}")

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def american_to_prob(odds):
    """Convert American odds to true implied probability"""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def normalize_bookmakers(event, scope, player_prop_type=None):
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            # Skip irrelevant player props
            if scope == "Player Props" and player_prop_type:
                if player_prop_type not in market.get("key", ""):
                    continue
            for outcome in market.get("outcomes", []):
                price = outcome.get("price")
                prob = american_to_prob(price) if price else None
                row = {
                    "Event": f"{event.get('away_team')} @ {event.get('home_team')}",
                    "Commence": event.get("commence_time"),
                    "Market": market.get("key"),
                    "Outcome": outcome.get("name"),
                    "Price": price,
                    "Implied Prob": prob,
                    "Bookmaker": book.get("title")
                }
                rows.append(row)
    return rows

# ------------------------
# NORMALIZE DATA
# ------------------------
all_rows = []
for event in data:
    rows = normalize_bookmakers(event, scope, player_prop_type)
    if rows:
        all_rows.extend(rows)

if not all_rows:
    st.warning("No normalized data available for selected filters.")
    st.stop()

df = pd.DataFrame(all_rows)

# ------------------------
# BEST BOOKMAKER HIGHLIGHT
# ------------------------
def highlight_best(row, df):
    """Mark best bookmaker per outcome (highest price for positive EV)"""
    subset = df[(df["Event"] == row["Event"]) & (df["Outcome"] == row["Outcome"])]
    if row["Price"] == subset["Price"].max():
        return "background-color: #b6fcd5"  # green highlight
    return ""

# ------------------------
# CALCULATE EXPECTED VALUE
# ------------------------
df["EV"] = df["Implied Prob"] * df["Price"] - (1 - df["Implied Prob"])
df_sorted = df.sort_values("EV", ascending=False)

# Auto-rank top 2–5 bets
top_bets = df_sorted.head(5)

# ------------------------
# DISPLAY
# ------------------------
st.subheader(f"EdgeLedger — {scope} ({sport_name})")
st.write("Top Bets Ranked by EV")
st.dataframe(top_bets.style.applymap(lambda x: highlight_best(x, df), subset=["Price"]))
