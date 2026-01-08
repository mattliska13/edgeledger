
import streamlit as st

def check_access(user):
    if user.get("plan") == "free":
        st.sidebar.warning("Upgrade to Pro or Elite for full access.")
