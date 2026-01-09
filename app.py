import streamlit as st
import requests
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="EdgeLedger Dashboard", layout="wide")
st.title("🏈 EdgeLedger Dashboard")

# -------------------------------
# API KEY
# -------------------------------
API_KEY = st.secrets.get("ODDS_API_KEY", "")
if not API_KEY:
    st.error("API_KEY not found. Please set ODDS_API_KEY in Streamlit secrets.")
else:
    st.success("API_KEY loaded successfully")

# -------------------------------
# SIDEBAR - DASHBOARD
# -------------------------------
st.sidebar.header("Dashboard")

sport_key = st.sidebar.selectbox("Select Sport", ["americanfootball_nfl", "americanfootball_ncaaf"])
scope = st.sidebar.radio("Scope", ["Game Lines", "Player Props"])

# Player prop type dropdown (only if Player Props selected)
prop_key = None
if scope == "Player Props":
    prop_map = {
        "Passing TDs": "player_pass_tds",
        "Passing Yards": "player_pass_yds",
        "Rushing Yards": "player_rush_yds",
        "Receiving Yards": "player_rec_yds",
        "Total TDs": "player_total_tds"
    }
    prop_selection = st.sidebar.selectbox("Player Prop Type", list(prop_map.keys()))
    prop_key = prop_map[prop_selection]

# -------------------------------
# UTILITY FUNCTIONS
# -------------------------------
def american_to_implied(odds):
    """Convert American odds to implied probability"""
    try:
        odds = float(odds)
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
    except:
        return None

def compute_ev(df):
    """Compute expected value"""
    df["Implied"] = df["Price"].apply(american_to_implied)
    df["EV"] = (df["Probability"] - df["Implied"]) * 100
    return df

def highlight_best(price, df, row):
    """Highlight the best bookmaker price"""
    subset = df[(df["Event"] == row["Event"]) & (df["Outcome"] == row["Outcome"])]
    if price == subset["Price"].max():
        return "background-color: #b6fcd5; font-weight: bold;"
    return ""

# -------------------------------
# FETCH GAME LINES
# -------------------------------
def fetch_game_lines():
    st.sidebar.info(f"DEBUG: Using sport_key={sport_key}, scope=Game Lines")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals",
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    events = r.json()
    rows = []
    for e in events:
        for m in e.get("bookmakers", []):
            for market in m.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "Event": e["home_team"] + " vs " + e["away_team"],
                        "Outcome": outcome["name"],
                        "Price": outcome["price"],
                        "Bookmaker": m["title"],
                        "Market": market["key"],
                        "Probability": outcome.get("probability", 0)
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = compute_ev(df)
        df.sort_values("EV", ascending=False, inplace=True)
    return df

# -------------------------------
# FETCH PLAYER PROPS
# -------------------------------
def fetch_player_props():
    st.sidebar.info(f"DEBUG: Using sport_key={sport_key}, scope=Player Props, player_prop_type={prop_key}")
    
    # Step 1: fetch events
    url_events = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params_events = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "player_props",
        "oddsFormat": "american"
    }
    r = requests.get(url_events, params=params_events)
    r.raise_for_status()
    events = r.json()

    rows = []
    # Step 2: filter props per selected prop_key
    for e in events:
        for m in e.get("bookmakers", []):
            for market in m.get("markets", []):
                if market.get("key") != "player_props":
                    continue
                for outcome in market.get("outcomes", []):
                    # only include selected prop type
                    if outcome.get("name_key") != prop_key:
                        continue
                    rows.append({
                        "Event": e["home_team"] + " vs " + e["away_team"],
                        "Player": outcome.get("name", ""),
                        "Outcome": outcome.get("label", ""),
                        "Price": outcome.get("price", 0),
                        "Bookmaker": m["title"],
                        "Probability": outcome.get("probability", 0)
                    })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = compute_ev(df)
        df.sort_values("EV", ascending=False, inplace=True)
    return df

# -------------------------------
# MAIN DISPLAY
# -------------------------------
if scope == "Game Lines":
    df = fetch_game_lines()
    if df.empty:
        st.warning("No Game Lines available.")
    else:
        st.subheader(f"Game Lines — {sport_key.upper()}")
        st.dataframe(df.style.apply(lambda x: [highlight_best(p, df, x) for p in x["Price"]], axis=1))
        st.info("Top 2–5 bets are ranked by EV.")

elif scope == "Player Props":
    df = fetch_player_props()
    if df.empty:
        st.warning("No Player Props available for this type.")
    else:
        st.subheader(f"Player Props — {prop_selection} ({sport_key.upper()})")
        st.dataframe(df.style.apply(lambda x: [highlight_best(p, df, x) for p in x["Price"]], axis=1))
        st.info("Top 2–5 props ranked by EV.")

# -------------------------------
# FOOTER / DEBUG
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.text("EdgeLedger Dashboard v1.0")
