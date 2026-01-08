
import streamlit as st
import pandas as pd

def run(user):
    st.subheader("UFC Props Engine")
    st.dataframe(pd.DataFrame({"Fight": ["KO/TKO", "Over 2.5 Rounds"]}))
