
import streamlit as st
from modules.product import check_access
from pages.game_lines import run as run_games
from pages.player_props import run as run_props
from pages.ufc_props import run as run_ufc

st.set_page_config(page_title="EdgeLedger", layout="wide")
st.title("EdgeLedger — Betting Intelligence Platform")

user = st.session_state.get("user", {"plan": "free"})
check_access(user)

page = st.sidebar.radio("Navigation", ["Game Lines", "Player Props", "UFC Props"])

if page == "Game Lines":
    run_games(user)
elif page == "Player Props":
    run_props(user)
else:
    run_ufc(user)
