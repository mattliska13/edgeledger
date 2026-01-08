import sys
import os
import streamlit as st

# Ensure root path is visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.product import check_access

st.set_page_config(page_title="EdgeLedger", layout="wide")

st.title("📊 EdgeLedger Betting Intelligence")

if not check_access():
    st.warning("Access restricted")
    st.stop()

st.success("App loaded successfully")

st.markdown("""
Welcome to **EdgeLedger**.

Use the sidebar to navigate:
- Game Lines
- Player Props
- UFC Props
""")
