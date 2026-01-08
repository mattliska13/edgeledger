
import streamlit as st
import pandas as pd

def run(user):
    st.subheader("Player Props Engine")
    st.dataframe(pd.DataFrame({"Prop": ["Anytime TD", "Over 65.5 Rec Yds"]}))
