# app.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="EdgeLedger Dashboard",
    page_icon="🎯",
    layout="wide",
)

st.markdown("""
    <style>
    .css-1d391kg {font-family: 'Source Sans Pro', sans-serif;}
    .stSidebar {background-color: #f0f2f6;}
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# CONFIG
# -------------------------------
API_KEY = st.secrets.get("ODDS_API_KEY", "")
HEADERS = {"User-Agent": "EdgeLedgerApp"}

SPORTS = {
    "NFL": "americanfootball_nfl",
    "CFB": "americanfootball_college"
}

PLAYER_PROP_TYPES = {
    "Passing Yards": "player_pass_yards",
    "Passing TDs": "player_pass_tds",
    "Rushing Yards": "player_rush_yards",
    "Rushing TDs": "player_rush_tds",
    "Receiving Yards": "player_receive_yards",
    "Receiving TDs": "player_receive_tds",
}

GAME_MARKETS = ["h2h", "spreads", "totals"]

# -------------------------------
# UTILS
# -------------------------------
def american_to_implied(odds):
    odds = int(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def compute_ev(df):
    if df.empty:
        return df
    df["Implied"] = df["Price"].apply(american_to_implied)
    df["EV"] = df["Implied"] - 0.5  # simple EV example, adjust logic as needed
    return df

def highlight_best_row(row, df):
    subset = df[(df["Event"] == row["Event"]) & (df["Outcome"] == row["Outcome"])]
    if row["Price"] == subset["Price"].max():
        return ["background-color: #b6fcd5; font-weight: bold;"] * len(row)
    return [""] * len(row)

# -------------------------------
# FETCH FUNCTIONS
# -------------------------------
def fetch_game_lines(sport_key):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(GAME_MARKETS),
        "oddsFormat": "american",
    }
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    events = r.json()
    rows = []
    for event in events:
        for market in event.get("bookmakers", []):
            for outcome in market.get("markets", []):
                if outcome["key"] in GAME_MARKETS:
                    for line in outcome.get("outcomes", []):
                        rows.append({
                            "Event": event["home_team"] + " vs " + event["away_team"],
                            "Outcome": line["name"],
                            "Price": line["price"],
                            "Bookmaker": market["title"],
                            "Market": outcome["key"]
                        })
    df = pd.DataFrame(rows)
    return compute_ev(df)

def fetch_player_props(sport_key, prop_type):
    # Two-step API call to avoid 422 errors
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "player_props",
        "oddsFormat": "american",
    }
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    events = r.json()
    rows = []
    for event in events:
        # Only fetch valid player props
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_props":
                    continue
                for outcome in market.get("outcomes", []):
                    if prop_type in outcome["name"].lower().replace(" ", "_"):
                        rows.append({
                            "Event": event["home_team"] + " vs " + event["away_team"],
                            "Outcome": outcome["name"],
                            "Price": outcome["price"],
                            "Bookmaker": bookmaker["title"],
                            "Market": market["key"]
                        })
    df = pd.DataFrame(rows)
    return compute_ev(df)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("🎯 Dashboard")
sport_choice = st.sidebar.selectbox("Select Sport", list(SPORTS.keys()))
scope_choice = st.sidebar.radio("Scope", ["Game Lines", "Player Props"])

player_prop_type = None
if scope_choice == "Player Props":
    prop_choice = st.sidebar.selectbox("Player Prop Type", list(PLAYER_PROP_TYPES.keys()))
    player_prop_type = PLAYER_PROP_TYPES[prop_choice]

# -------------------------------
# MAIN
# -------------------------------
st.title("EdgeLedger — {}".format(scope_choice))

sport_key = SPORTS[sport_choice]
df = pd.DataFrame()

try:
    if scope_choice == "Game Lines":
        df = fetch_game_lines(sport_key)
    elif scope_choice == "Player Props":
        df = fetch_player_props(sport_key, player_prop_type)

    if df.empty:
        st.info("No data available.")
    else:
        # Rank top bets by EV
        top_bets = df.sort_values("EV", ascending=False).head(5)
        st.subheader("Top Bets Ranked by EV")
        st.dataframe(top_bets.style.apply(lambda row: highlight_best_row(row, top_bets), axis=1))

except requests.exceptions.HTTPError as e:
    st.error(f"API Error: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")
