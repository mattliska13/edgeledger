import streamlit as st
import requests
from datetime import datetime
from operator import itemgetter

# ----------------------------
# Config / API Key
# ----------------------------
API_KEY = st.secrets.get("ODDS_API_KEY")
API_URL = "https://api.the-odds-api.com/v4/sports/{sport_key}/odds"

if not API_KEY:
    st.error("API_KEY not found in Streamlit secrets!")
    st.stop()

st.sidebar.title("EdgeLedger Sports Betting")
sport_options = ["americanfootball_nfl", "americanfootball_college", "mma_mixed_martial_arts"]
sport_display = {"americanfootball_nfl": "NFL",
                 "americanfootball_college": "CFB",
                 "mma_mixed_martial_arts": "UFC"}

sport_key = st.sidebar.selectbox("Select Sport", sport_options, format_func=lambda x: sport_display.get(x))
scope_options = ["Game Lines", "Player Props"]
scope = st.sidebar.radio("Scope", scope_options)

player_prop_types = ["passing", "rushing", "receiving", "fumbles", "touchdowns", "any"]
selected_player_prop = st.sidebar.selectbox("Player Prop Type", player_prop_types)

st.write(f"DEBUG: Using sport_key={sport_key}, scope={scope}, player_prop_type={selected_player_prop}")

# ----------------------------
# Fetch Odds Data
# ----------------------------
def fetch_odds(sport):
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h,spreads,totals,player_props",
        "oddsFormat": "american"
    }
    r = requests.get(API_URL.format(sport_key=sport), params=params)
    try:
        r.raise_for_status()
        data = r.json()
        st.write(f"DEBUG: API returned {len(data)} events for {sport_display.get(sport)}")
        return data
    except requests.HTTPError as e:
        st.error(f"HTTPError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# ----------------------------
# Normalize Events
# ----------------------------
def normalize_events(events, sport_key):
    normalized_game_lines = []
    normalized_player_props = []

    for event in events:
        event_id = event.get("id")
        home = event.get("home_team")
        away = event.get("away_team")
        commence_time = event.get("commence_time")
        bookmakers = event.get("bookmakers", [])

        if not bookmakers:
            st.write(f"DEBUG: Skipping event {event_id} ({home} vs {away}) — no bookmakers")
            continue

        for bookmaker in bookmakers:
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])

                if not outcomes:
                    st.write(f"DEBUG: Skipping empty market {key} for {home} vs {away}")
                    continue

                # Game Lines
                if key in ["spreads", "totals", "h2h"]:
                    normalized_game_lines.append({
                        "event_id": event_id,
                        "sport": sport_key,
                        "market": key,
                        "bookmaker": bookmaker.get("title"),
                        "home_team": home,
                        "away_team": away,
                        "outcomes": outcomes,
                        "commence_time": commence_time
                    })

                # Player Props
                if key.startswith("player_"):
                    if selected_player_prop == "any" or selected_player_prop in key:
                        normalized_player_props.append({
                            "event_id": event_id,
                            "sport": sport_key,
                            "market": key,
                            "bookmaker": bookmaker.get("title"),
                            "outcomes": outcomes,
                            "commence_time": commence_time
                        })

    return normalized_game_lines, normalized_player_props

# ----------------------------
# Convert American Odds to Decimal
# ----------------------------
def american_to_decimal(odds):
    if odds > 0:
        return odds / 100 + 1
    else:
        return 100 / abs(odds) + 1

# ----------------------------
# Convert Decimal Odds to Implied Probability
# ----------------------------
def implied_prob(decimal_odds):
    return 1 / decimal_odds

# ----------------------------
# Compute Expected Value (EV)
# ----------------------------
def compute_ev(decimal_odds, probability):
    return round(decimal_odds * probability - 1, 3)

# ----------------------------
# Find best bookmaker per outcome
# ----------------------------
def highlight_best_bookmakers(outcomes_list):
    best_bookmakers = {}
    for outcome in outcomes_list:
        key = outcome["team"] if "team" in outcome else outcome["team/player"]
        if key not in best_bookmakers or outcome["decimal_price"] > best_bookmakers[key]["decimal_price"]:
            best_bookmakers[key] = outcome
    return best_bookmakers

# ----------------------------
# Main Logic
# ----------------------------
events = fetch_odds(sport_key)
game_lines, player_props = normalize_events(events, sport_key)

def process_markets(markets, scope_name):
    all_outcomes = []

    for m in markets:
        # Convert odds to decimal & implied probability
        total_implied = sum(american_to_decimal(o["price"])**-1 for o in m["outcomes"])
        for o in m["outcomes"]:
            dec = american_to_decimal(o["price"])
            imp_prob = implied_prob(dec) / total_implied  # normalize to sum=1
            o["decimal_price"] = dec
            o["implied_prob"] = round(imp_prob, 3)
            o["ev"] = compute_ev(dec, imp_prob)
            all_outcomes.append({
                "event": f"{m.get('home_team', m['event_id'])} vs {m.get('away_team', '')}",
                "market": m["market"],
                "bookmaker": m["bookmaker"],
                "team/player": o["name"],
                "price": o["price"],
                "decimal_price": dec,
                "implied_prob": o["implied_prob"],
                "ev": o["ev"],
                "commence_time": m["commence_time"]
            })

    # Highlight best bookmaker per outcome
    best_bookmakers = highlight_best_bookmakers(all_outcomes)
    for o in all_outcomes:
        key = o["team/player"]
        o["best_bookmaker"] = o["bookmaker"] == best_bookmakers[key]["bookmaker"]

    # Rank top 2–5 bets by EV
    ranked = sorted(all_outcomes, key=itemgetter("ev"), reverse=True)[:5]
    st.subheader(f"Top {scope_name} by EV")
    st.table(ranked)

if scope == "Game Lines":
    if not game_lines:
        st.warning("No normalized game lines available.")
    else:
        process_markets(game_lines, "Game Lines")

elif scope == "Player Props":
    if not player_props:
        st.warning("No normalized player props available.")
    else:
        process_markets(player_props, "Player Props")

st.write("DEBUG: Normalization complete")
st.write(f"DEBUG: {len(game_lines)} game lines, {len(player_props)} player props found")
