import streamlit as st

st.title("🎯 Player Props")

st.selectbox(
    "Prop Type",
    [
        "Anytime TD",
        "Receiving Yards O/U",
        "Rushing Yards O/U",
        "Passing Yards O/U",
    ]
)

st.info("Player prop models loading here")
