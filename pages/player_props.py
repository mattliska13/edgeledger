import streamlit as st
import pandas as pd
from modules.player_props_model import (
    receiving_projection,
    rushing_projection,
    passing_projection
)

st.title("🎯 Player Props")

player = st.text_input("Player Name")
position = st.selectbox("Position", ["WR", "RB", "QB"])
snap_pct = st.slider("Snap %", 0.3, 1.0, 0.75)

if position == "WR":
    targets, yards = receiving_projection(position, snap_pct)
    st.metric("Projected Targets", targets)
    st.metric("Projected Receiving Yards", yards)

elif position == "RB":
    carries, yards = rushing_projection(position, snap_pct)
    st.metric("Projected Carries", carries)
    st.metric("Projected Rushing Yards", yards)

elif position == "QB":
    attempts, yards = passing_projection(snap_pct)
    st.metric("Projected Attempts", attempts)
    st.metric("Projected Passing Yards", yards)
