import streamlit as st

st.title("🥊 UFC Props")

st.selectbox(
    "Prop Type",
    [
        "Fight Goes Distance",
        "Method of Victory",
        "Round Betting",
    ]
)

st.info("UFC prop engine loading here")
