
import streamlit as st
import pandas as pd

def run(user):
    st.subheader("Game Lines Engine")
    st.write("Spreads, Totals, Moneylines")
    st.dataframe(pd.DataFrame({"Example": ["Bills -3", "Over 47.5"]}))
