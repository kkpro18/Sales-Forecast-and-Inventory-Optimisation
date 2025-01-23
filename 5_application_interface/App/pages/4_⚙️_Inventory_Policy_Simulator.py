import streamlit as st
import pandas as pd
# import plotly.graph_objects as go
import simpy

# to run application type this into the terminal "streamlit run 5_application_interface/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Inventory Policy Simulator",
    page_icon="⚙",
)

st.markdown("# Inventory Policy Simulator")
st.write("""Let's Begin Simulating Inventory Policies!""")

if 'uploaded_dataset' in st.session_state:
    uploaded_dataset = st.session_state["uploaded_dataset"]


else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
