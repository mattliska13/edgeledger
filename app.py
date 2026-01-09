import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="EdgeLedger — Game Lines & Player Props", layout="wide")

# -----------------------
# CONFIG
# -----------------------
API_KEY = st.secrets["ODDS_API_KEY"]
REGION = "us"
ODDS_FORMAT = "american"

SPORT_OPTIONS = ["NFL", "CFB"]
SCOPE_OPTIONS = ["Game Lines", "Player Props"]

PLAYER_PROP_TYPES = {
    "passing": "player_pass_yards",
    "passing TDs": "player_pass_tds",
    "rushing": "player_rush_yards",
    "rushing TDs": "player_rush_tds",
    "receiving": "player_receive_yards",
    "receiving TDs": "player_receive_tds"
}

# -----------------------
# UTILITY FUNCTIONS
# -----------------------
def american_to_implied(price):
    try:
        price = float(price)
    except:
        return np.nan
    if price > 0:
        return 100 / (price + 100)
    else:
        return -price / (-price + 100)

def compute_ev(df):
    df["Implied"] = df["Price"].apply(american_to_implied)
    df["EV"] = df["Implied"]  # For simplicity, EV = implied probability for demo
    return df

def highlight_best_bookmaker(df):
    """Return a dataframe of styles highlighting the best bookmaker per Event/Outcome"""
    style_df = pd.DataFrame("", index=df.index, columns=df.columns)
    for event in df["Event"].unique():
        event_df = df[df["Event"] == event]
        for outcome in event_df["Outcome"].unique():
            subset = event_df[event_df["Outcome"] == outcome]
            if subset.empty:
                continue
            max_price = subset["Price"].max()
            best_idx = subset[subset["Price"] == max_price].index
            style_df.loc[best_idx, "Price"] = "background-color: lightgreen"
    return style_df

# -----------------------
# DATA FETCHING
# -----------------------
def fetch_game_lines(sport_key):
    markets = ["h2h", "spreads", "totals"]
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": API_KEY, "regions": REGION, "markets": ",".join(markets), "oddsFormat": ODDS_FORMAT}
    r = requests.get(url, params=params)
    r.raise_for_status()
    events = r.json()

    data = []
    for event in events:
        event_name = event["home_team"] + " vs " + event["away_team"]
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    data.append({
                        "Event": event_name,
                        "Outcome": outcome["name"],
                        "Bookmaker": bookmaker["title"],
                        "Price": outcome["price"]
                    })
    df = pd.DataFrame(data)
    return compute_ev(df)

def fetch_player_props(sport_key, player_prop_type):
    # Step 1: fetch events
    url_events = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    r = requests.get(url_events, params={"apiKey": API_KEY, "regions": REGION})
    r.raise_for_status()
    events = r.json()
    st.text(f"DEBUG: API returned {len(events)} events for {sport_key}")

    all_props = []

    # Step 2: fetch props per event
    for event in events:
        event_id = event["id"]
        url_props = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": API_KEY,
            "regions": REGION,
            "markets": player_prop_type,
            "oddsFormat": ODDS_FORMAT
        }
        try:
            r_prop = requests.get(url_props, params=params)
            r_prop.raise_for_status()
            event_props = r_prop.json()

            if "bookmakers" not in event_props:
                continue

            event_name = event_props.get("home_team", "") + " vs " + event_props.get("away_team", "")

            for bookmaker in event_props["bookmakers"]:
                for market in bookmaker.get("markets", []):
                    if market["key"] != player_prop_type:
                        continue
                    for outcome in market.get("outcomes", []):
                        all_props.append({
                            "Event": event_name,
                            "Outcome": outcome["name"],
                            "Bookmaker": bookmaker["title"],
                            "Price": outcome["price"]
                        })
        except requests.HTTPError as e:
            st.warning(f"Skipping event {event_id} due to HTTPError: {e}")
    df = pd.DataFrame(all_props)
    return compute_ev(df)

# -----------------------
# UI
# -----------------------
st.title("EdgeLedger — Game Lines & Player Props")

sport_name = st.selectbox("Sport", SPORT_OPTIONS)
scope = st.selectbox("Scope", SCOPE_OPTIONS)

sport_key = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_college"
}[sport_name]

player_prop_type = None
if scope == "Player Props":
    prop_choice = st.selectbox("Player Prop Type", list(PLAYER_PROP_TYPES.keys()))
    player_prop_type = PLAYER_PROP_TYPES[prop_choice]

# -----------------------
# FETCH DATA
# -----------------------
if scope == "Game Lines":
    st.text(f"DEBUG: Using sport_key={sport_key}, scope={scope}")
    df = fetch_game_lines(sport_key)
elif scope == "Player Props" and player_prop_type:
    st.text(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")
    df = fetch_player_props(sport_key, player_prop_type)
else:
    df = pd.DataFrame()

# -----------------------
# RANKING TOP BETS
# -----------------------
if not df.empty:
    top_bets = df.sort_values("EV", ascending=False).head(5)
    st.subheader(f"Top Bets Ranked by EV — {scope}")
    st.dataframe(top_bets.style.apply(lambda _: highlight_best_bookmaker(top_bets), axis=None))
else:
    st.info("No data available for selected scope and sport.")
