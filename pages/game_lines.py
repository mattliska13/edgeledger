import streamlit as st

st.title("🏈 Game Lines")

st.selectbox("Sport", ["NFL", "CFB"])
st.selectbox("Market", ["Spread", "Total", "Moneyline"])

st.info("Game line analytics loading here")
