import streamlit as st
import requests
import pandas as pd
import numpy as np

st.set_page_config(page_title="EdgeLedger", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = st.secrets["ODDS_API_KEY"]
REGION = "us"
ODDS_FORMAT = "american"

# Supported sports
SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_college"
}

# Player prop types
PLAYER_PROPS = {
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yds",
    "Rushing Yards": "player_rush_yds",
    "Receiving Yards": "player_rec_yds",
    "Anytime TD": "player_anytime_td"
}

# -----------------------------
# UTILITIES
# -----------------------------
def american_to_decimal(odds):
    if odds > 0:
        return odds / 100 + 1
    else:
        return 100 / -odds + 1

def calc_ev(american_odds, probability):
    dec_odds = american_to_decimal(american_odds)
    return probability * (dec_odds - 1) - (1 - probability)

def highlight_best(val, col_vals):
    return "background-color: #b6fcd5" if val == col_vals.max() else ""

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("EdgeLedger")

sport_name = st.selectbox("Sport", list(SPORTS.keys()))
scope = st.selectbox("Scope", ["Game Lines", "Player Props"])

sport_key = SPORTS[sport_name]

player_prop_type = None
if scope == "Player Props":
    player_prop_name = st.selectbox("Player Prop Type", list(PLAYER_PROPS.keys()))
    player_prop_type = PLAYER_PROPS[player_prop_name]

st.text(f"DEBUG: API_KEY loaded successfully")
st.text(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_game_lines():
    markets = ["h2h", "spreads", "totals"]
    st.text(f"DEBUG: Requesting markets: {markets}")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": ",".join(markets),
        "oddsFormat": ODDS_FORMAT
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    st.text(f"DEBUG: API returned {len(data)} events for {sport_name}")
    return data

def fetch_player_props():
    # Step 1: fetch events
    url_events = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
    r = requests.get(url_events, params={"apiKey": API_KEY, "regions": REGION})
    r.raise_for_status()
    events = r.json()
    st.text(f"DEBUG: API returned {len(events)} events for {sport_name}")

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
            props = r_prop.json()
            for p in props:
                for market in p.get("bookmakers", []):
                    for outcome in market.get("markets", []):
                        if outcome["key"] == player_prop_type:
                            for o in outcome["outcomes"]:
                                all_props.append({
                                    "Event": p["home_team"] + " vs " + p["away_team"],
                                    "Outcome": o["name"],
                                    "Bookmaker": market["title"],
                                    "Price": o["price"]
                                })
        except requests.HTTPError as e:
            st.warning(f"Skipping event {event_id} due to HTTPError: {e}")
    return pd.DataFrame(all_props)

# -----------------------------
# PROCESS & DISPLAY
# -----------------------------
if scope == "Game Lines":
    raw_data = fetch_game_lines()
    # Flatten for display
    rows = []
    for event in raw_data:
        event_name = event["home_team"] + " vs " + event["away_team"]
        for bookmaker in event["bookmakers"]:
            for market in bookmaker["markets"]:
                for outcome in market["outcomes"]:
                    rows.append({
                        "Event": event_name,
                        "Outcome": outcome["name"],
                        "Bookmaker": bookmaker["title"],
                        "Price": outcome["price"]
                    })
    df = pd.DataFrame(rows)
elif scope == "Player Props":
    df = fetch_player_props()

if df.empty:
    st.warning("No data found for this selection.")
else:
    # Calculate implied probability and EV
    df["Decimal"] = df["Price"].apply(american_to_decimal)
    df["ImpliedProb"] = 1 / df["Decimal"]
    df["EV"] = df.apply(lambda x: calc_ev(x["Price"], x["ImpliedProb"]), axis=1)

    # Rank top bets
    top_bets = df.sort_values("EV", ascending=False).head(5)

    # Highlight best bookmaker per line/outcome
    styled = top_bets.style.apply(
        lambda x: [highlight_best(v, top_bets[top_bets["Outcome"]==x["Outcome"]]["Price"]) for v in x], axis=1
    )

    st.subheader(f"Top Bets Ranked by EV — {scope}")
    st.dataframe(styled, use_container_width=True)
