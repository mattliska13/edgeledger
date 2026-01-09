import streamlit as st
import requests
import pandas as pd

# --------------------------
# Page Config & Theme
# --------------------------
st.set_page_config(
    page_title="EdgeLedger Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's default menu, footer, and header
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-18e3th9 {padding-top: 1rem;}  /* Reduce top padding */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom CSS for fonts, colors, cards
st.markdown("""
    <style>
    body {font-family: 'Source Sans Pro', sans-serif; color: #111;}
    .stButton>button {background-color:#FF4B4B; color:white; border-radius:10px;}
    .stDataFrame {border: 1px solid #ccc; border-radius: 10px;}
    h1 {color:#FF4B4B; font-family: 'Source Sans Pro', sans-serif;}
    h2 {color:#333333; font-family: 'Source Sans Pro', sans-serif;}
    </style>
""", unsafe_allow_html=True)

st.title("🔥 EdgeLedger Dashboard")

# --------------------------
# API Key
# --------------------------
API_KEY = st.secrets.get("ODDS_API_KEY", "")
if not API_KEY:
    st.error("API_KEY not found in secrets.toml")
    st.stop()
st.sidebar.success("DEBUG: API_KEY loaded successfully")

# --------------------------
# Helper Functions
# --------------------------
def american_to_implied(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def compute_ev(df, true_prob=None):
    if df.empty:
        return df
    df["Implied"] = df["Price"].apply(american_to_implied)
    if true_prob is None:
        df["EV"] = df["Implied"] - 0.5
    else:
        df["EV"] = df.apply(lambda row: row["Implied"] - true_prob.get(row["Outcome"], 0.5), axis=1)
    return df

def highlight_best(val, df, column="Price"):
    if df.empty or column not in df.columns:
        return ""
    max_price = df.groupby(["Event", "Outcome"])[column].transform("max")
    return "background-color: yellow" if val == max_price else ""

def ev_gradient(val, min_ev, max_ev):
    if pd.isna(val):
        return ""
    if val >= 0:
        pct = (val / max_ev) * 100 if max_ev > 0 else 0
        color = f"rgba(0, 200, 0, 0.6)"
    else:
        pct = (val / min_ev) * 100 if min_ev < 0 else 0
        color = f"rgba(200, 0, 0, 0.6)"
    return f"background: linear-gradient(to right, {color} {pct}%, transparent {pct}%);"

# --------------------------
# Fetch Game Lines
# --------------------------
def fetch_game_lines(sport_key):
    markets = ["h2h", "spreads", "totals"]
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": API_KEY, "regions": "us", "markets": ",".join(markets), "oddsFormat": "american"}
    st.sidebar.debug(f"DEBUG: Using sport_key={sport_key}, scope=Game Lines")
    r = requests.get(url, params=params)
    r.raise_for_status()
    events = r.json()
    all_lines = []
    for e in events:
        event_name = e.get("home_team") + " vs " + e.get("away_team")
        for b in e.get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    all_lines.append({
                        "Event": event_name,
                        "Outcome": o.get("name"),
                        "Bookmaker": b.get("title"),
                        "Price": o.get("price"),
                    })
    df = pd.DataFrame(all_lines)
    return compute_ev(df)

# --------------------------
# Fetch Player Props
# --------------------------
PROP_TYPE_MAP = {
    "Passing TDs": "player_pass_tds",
    "Passing Yards": "player_pass_yards",
    "Rushing Yards": "player_rush_yards",
    "Receiving Yards": "player_rec_yards",
    "Touchdowns": "player_tds",
}

def fetch_player_props(sport_key, player_prop_type):
    st.sidebar.debug(f"DEBUG: Using sport_key={sport_key}, scope=Player Props, player_prop_type={player_prop_type}")
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey": API_KEY, "regions": "us", "markets": "player_props", "oddsFormat": "american"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    events = r.json()
    all_props = []
    for e in events:
        event_name = e.get("home_team") + " vs " + e.get("away_team")
        for b in e.get("bookmakers", []):
            for m in b.get("markets", []):
                if m.get("key") != player_prop_type:
                    continue
                for o in m.get("outcomes", []):
                    all_props.append({
                        "Event": event_name,
                        "Outcome": o.get("name"),
                        "Bookmaker": b.get("title"),
                        "Price": o.get("price"),
                    })
    df = pd.DataFrame(all_props, columns=["Event", "Outcome", "Bookmaker", "Price"])
    if df.empty:
        st.warning("No player props returned for selected prop type.")
        return df
    return compute_ev(df)

# --------------------------
# Dashboard Sidebar
# --------------------------
st.sidebar.header("📊 Dashboard")
sport_options = {"NFL": "americanfootball_nfl", "CFB": "americanfootball_ncaaf"}
sport_choice = st.sidebar.selectbox("Select Sport", options=list(sport_options.keys()))
sport_key = sport_options[sport_choice]

scope_options = ["Game Lines", "Player Props"]
scope_choice = st.sidebar.radio("Scope", options=scope_options)

# Player Prop dropdown (only if Player Props selected)
if scope_choice == "Player Props":
    player_prop_dropdown = st.sidebar.selectbox("Player Prop Type", list(PROP_TYPE_MAP.keys()))
    prop_key = PROP_TYPE_MAP[player_prop_dropdown]

# --------------------------
# Fetch & Display Data
# --------------------------
if scope_choice == "Game Lines":
    df = fetch_game_lines(sport_key)
    st.subheader(f"🔥 Game Lines ({sport_choice})")
elif scope_choice == "Player Props":
    df = fetch_player_props(sport_key, prop_key)
    st.subheader(f"🔥 Player Props ({sport_choice} — {player_prop_dropdown})")

if not df.empty:
    top_bets = df.sort_values("EV", ascending=False).head(5)
    min_ev, max_ev = df["EV"].min(), df["EV"].max()
    st.subheader("Top Bets Ranked by EV")
    st.dataframe(top_bets.style
                 .applymap(lambda x: highlight_best(x, df), subset=["Price"])
                 .applymap(lambda x: ev_gradient(x, min_ev, max_ev), subset=["EV"]))
else:
    st.info("No data available for the selected sport/scope/prop type.")
