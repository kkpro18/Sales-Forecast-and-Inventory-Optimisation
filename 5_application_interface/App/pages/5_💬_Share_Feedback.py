import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.pyplot import xlabel

# to run application type this into the terminal "streamlit run 5_application_interface/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Sales Forecasting and Inventory Optimisation Service",
    page_icon="📈",
)

if 'uploaded_dataset' in st.session_state:
    uploaded_dataset = st.session_state["uploaded_dataset"]