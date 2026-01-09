import streamlit as st
import requests
import pandas as pd
import logging

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="EdgeLedger — Sports EV", layout="wide")
logging.basicConfig(level=logging.DEBUG)

# Load API Key
try:
    API_KEY = st.secrets["ODDS_API_KEY"]
    logging.debug("DEBUG: API_KEY loaded successfully")
except Exception as e:
    st.error("API Key not found in Streamlit secrets.")
    st.stop()

# -------------------------
# UI SELECTIONS
# -------------------------
sport_options = ["NFL", "CFB", "UFC"]
scope_options = ["Game Lines", "Player Props"]
player_prop_options = ["passing", "rushing", "receiving", "touchdowns"]

sport_name = st.selectbox("Sport", sport_options)
scope = st.selectbox("Scope", scope_options)

player_prop_type = None
if scope == "Player Props" and sport_name in ["NFL", "CFB"]:
    player_prop_type = st.selectbox("Player Prop Type", player_prop_options)

# -------------------------
# SPORT KEYS MAPPING
# -------------------------
SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_ncaa",
    "UFC": "mma_mixed_martial_arts"
}
sport_key = SPORT_KEYS[sport_name]

# -------------------------
# FETCH ODDS
# -------------------------
def fetch_odds(sport_key, scope):
    base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    markets = []

    if scope == "Game Lines":
        markets = ["h2h", "spreads", "totals"]
    elif scope == "Player Props":
        markets = ["player_props"]  # ONLY player_props to avoid 422

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

# -------------------------
# NORMALIZE DATA
# -------------------------
def normalize_data(events, scope, player_prop_type=None):
    rows = []
    for event in events:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        event_name = f"{home_team} vs {away_team}" if home_team and away_team else event.get("name", "")

        if "bookmakers" not in event or not event["bookmakers"]:
            continue

        for bookmaker in event["bookmakers"]:
            for market in bookmaker["markets"]:
                # For Player Props, filter locally
                if scope == "Player Props" and player_prop_type:
                    filtered_outcomes = [
                        o for o in market.get("outcomes", [])
                        if player_prop_type.lower() in o.get("name", "").lower()
                    ]
                else:
                    filtered_outcomes = market.get("outcomes", [])

                for outcome in filtered_outcomes:
                    price = outcome.get("price")
                    if price is None:
                        continue

                    # Convert American odds to implied probability
                    if price > 0:
                        implied_prob = 100 / (price + 100)
                    else:
                        implied_prob = -price / (-price + 100)

                    # EV (expected value) assuming true probability = 0.5 as placeholder
                    ev = implied_prob - 0.5

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

# -------------------------
# BEST BOOKMAKER HIGHLIGHT
# -------------------------
def highlight_best(val, df, row):
    # Highlight the bookmaker offering the best price for this event/outcome
    subset = df[(df["Event"] == row["Event"]) & (df["Outcome"] == row["Outcome"])]
    if subset.empty:
        return ""
    max_price = subset["Price"].max()
    if val == max_price:
        return "background-color: #b3ffb3"  # green highlight
    return ""

# -------------------------
# MAIN
# -------------------------
try:
    events = fetch_odds(sport_key, scope)
    df = normalize_data(events, scope, player_prop_type)

    if df.empty:
        st.info("No odds data available for the selected filters.")
    else:
        # Rank top bets by EV
        top_bets = df.sort_values(by="EV", ascending=False).head(5)

        # Display table with best bookmaker highlighted
        def style_func(row):
            return [highlight_best(row["Price"], df, row) if col == "Price" else "" for col in df.columns]

        st.subheader(f"EdgeLedger — {scope} ({sport_name})")
        st.dataframe(top_bets.style.apply(style_func, axis=1))

except requests.HTTPError as e:
    st.error(f"HTTP Error: {e}")
    logging.error(f"HTTP Error: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    logging.error(f"Unexpected error: {e}")
