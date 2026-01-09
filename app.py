import streamlit as st
import pandas as pd
import requests
import logging

# ------------------------
# SETUP
# ------------------------
st.set_page_config(page_title="EdgeLedger", layout="wide")
logging.basicConfig(level=logging.DEBUG)

# Load API key
API_KEY = st.secrets.get("ODDS_API_KEY")
if not API_KEY:
    st.error("API_KEY not found in Streamlit secrets.")
    st.stop()
logging.debug("DEBUG: API_KEY loaded successfully")

# ------------------------
# SPORT + SCOPE SELECTION
# ------------------------
SPORT_OPTIONS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_college",
    "UFC": "mma_ufc"
}

sport_name = st.selectbox("Sport", list(SPORT_OPTIONS.keys()))
sport_key = SPORT_OPTIONS[sport_name]

scope = st.radio("Scope", ["Game Lines", "Player Props"])

# Player prop type dropdown (only for NFL/CFB)
PLAYER_PROP_OPTIONS = {
    "NFL": ["passing", "rushing", "receiving", "touchdowns"],
    "CFB": ["passing", "rushing", "receiving", "touchdowns"],
    "UFC": []
}
player_prop_type = None
if scope == "Player Props" and PLAYER_PROP_OPTIONS[sport_name]:
    player_prop_type = st.selectbox("Player Prop Type", PLAYER_PROP_OPTIONS[sport_name])

# ------------------------
# API REQUEST
# ------------------------
def fetch_odds(sport_key, scope, player_prop_type=None):
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    markets = []

    if scope == "Game Lines":
        markets = ["h2h", "spreads", "totals"]
    elif scope == "Player Props":
        markets = [f"player_props:{player_prop_type}"]

    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }

    logging.debug(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")
    logging.debug(f"DEBUG: Requesting markets: {markets}")

    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()
    logging.debug(f"DEBUG: API returned {len(data)} events for {sport_name}")
    return data

# ------------------------
# DATA NORMALIZATION
# ------------------------
def normalize_data(events):
    rows = []
    for event in events:
        event_name = event.get("home_team", "") + " vs " + event.get("away_team", "")
        if "bookmakers" not in event or not event["bookmakers"]:
            continue

        for bookmaker in event["bookmakers"]:
            for market in bookmaker["markets"]:
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    # Calculate implied probability and EV
                    implied_prob = 100 / (price if price > 0 else 1)
                    ev = implied_prob - 0.5  # simple EV metric (can adjust as needed)
                    rows.append({
                        "Event": event_name,
                        "Market": market.get("key"),
                        "Outcome": outcome.get("name"),
                        "Price": price,
                        "Bookmaker": bookmaker.get("title"),
                        "EV": ev
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No normalized data available.")
    return df

# ------------------------
# BEST BOOKMAKER HIGHLIGHT
# ------------------------
def highlight_best_row(row, df):
    subset = df[(df["Event"] == row["Event"]) & (df["Outcome"] == row["Outcome"])]
    is_best = row["Price"] == subset["Price"].max()
    return ["background-color: #b6fcd5" if col == "Price" and is_best else "" for col in row.index]

# ------------------------
# MAIN
# ------------------------
try:
    events = fetch_odds(sport_key, scope, player_prop_type)
    df = normalize_data(events)

    if not df.empty:
        # Rank top bets by EV
        top_bets = df.sort_values("EV", ascending=False).head(10).reset_index(drop=True)

        st.subheader(f"EdgeLedger — {scope} ({sport_name})")
        st.write("Top Bets Ranked by EV")

        st.dataframe(
            top_bets.style.apply(lambda row: highlight_best_row(row, df), axis=1)
        )
except requests.exceptions.HTTPError as e:
    st.error(f"HTTPError: {e}")
    logging.error(f"HTTPError: {e}")
except Exception as e:
    st.error(f"This app has encountered an error. Check logs for details.")
    logging.exception("Unhandled Exception")
