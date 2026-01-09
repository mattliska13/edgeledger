import streamlit as st
import requests
import pandas as pd

# =====================
# CONFIGURATION
# =====================
API_KEY = st.secrets["ODDS_API_KEY"]  # Load securely
API_URL = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds"

# Supported sports and their valid markets
SPORT_MARKETS = {
    "americanfootball_nfl": ["h2h", "spreads", "totals"],
    "americanfootball_college": ["h2h", "spreads", "totals"],
    "mma_mixed_martial_arts": ["h2h"]
}

SPORT_DISPLAY = {
    "americanfootball_nfl": "NFL",
    "americanfootball_college": "CFB",
    "mma_mixed_martial_arts": "UFC"
}

PLAYER_PROP_TYPES = ["passing", "rushing", "receiving", "touchdowns"]

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="EdgeLedger — Best Bets + EV", layout="wide")
st.title("EdgeLedger — Game Lines + Player Props (Best Price + EV)")

sport_key = st.selectbox("Sport", list(SPORT_MARKETS.keys()), format_func=lambda x: SPORT_DISPLAY[x])
scope_options = ["Game Lines"]
if sport_key in ["americanfootball_nfl", "americanfootball_college"]:
    scope_options.append("Player Props")
scope = st.selectbox("Scope", scope_options)

player_prop_type = None
if scope == "Player Props" and sport_key in ["americanfootball_nfl", "americanfootball_college"]:
    player_prop_type = st.selectbox("Player Prop Type", PLAYER_PROP_TYPES)

st.write(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={player_prop_type}")

# =====================
# HELPER FUNCTIONS
# =====================
def american_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def calc_ev(odds, prob):
    """Calculate expected value (EV)."""
    return american_to_prob(odds) * (odds if odds > 0 else 100) / 100 - (1 - prob)

def fetch_odds(sport):
    """Fetch odds from Odds API for valid markets per sport."""
    markets = SPORT_MARKETS.get(sport, ["h2h"])
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": ",".join(markets),
        "oddsFormat": "american"
    }
    url = API_URL.format(sport_key=sport)
    r = requests.get(url, params=params)
    try:
        r.raise_for_status()
        data = r.json()
        st.write(f"DEBUG: API returned {len(data)} events for {SPORT_DISPLAY[sport]}")
        return data
    except requests.HTTPError as e:
        st.error(f"HTTPError: {e}")
        st.stop()

def normalize_bookmakers(event):
    """Flatten bookmaker data into a DataFrame with EV."""
    rows = []
    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            for outcome in market.get("outcomes", []):
                row = {
                    "event_id": event["id"],
                    "sport": event["sport_title"],
                    "commence_time": event["commence_time"],
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": book["title"],
                    "market": market["key"],
                    "outcome": outcome["name"],
                    "price": outcome["price"],
                    "point": outcome.get("point", None)
                }
                # Implied probability and EV
                prob = american_to_prob(outcome["price"])
                row["implied_prob"] = prob
                row["ev"] = calc_ev(outcome["price"], prob)
                rows.append(row)
    return pd.DataFrame(rows)

# =====================
# FETCH & PROCESS DATA
# =====================
events = fetch_odds(sport_key)
all_rows = pd.concat([normalize_bookmakers(ev) for ev in events], ignore_index=True)
if all_rows.empty:
    st.warning("No normalized data available for this selection.")
else:
    # =====================
    # BEST PRICE / EV HIGHLIGHT
    # =====================
    best_ev_idx = all_rows.groupby(["event_id", "market", "outcome"])["ev"].idxmax()
    all_rows["best_ev"] = False
    all_rows.loc[best_ev_idx, "best_ev"] = True

    # =====================
    # TOP 2–5 BETS AUTO-RANK
    # =====================
    top_bets = all_rows.sort_values("ev", ascending=False).head(5)
    
    # =====================
    # DISPLAY TABLE
    # =====================
    def highlight_best(s):
        return ["background-color: lightgreen" if v else "" for v in s]

    st.subheader("Top EV Bets")
    st.dataframe(top_bets.style.apply(highlight_best, subset=["best_ev"]))

    st.subheader("All Available Lines")
    st.dataframe(all_rows.style.apply(highlight_best, subset=["best_ev"]))

st.write("DEBUG: Data loaded successfully")
