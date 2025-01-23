import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.pyplot import xlabel

# to run application type this into the terminal "streamlit run 5_application_interface/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="🏠",
)
st.sidebar.success("Application Launched Successfully")

st.write("# Welcome to this Application! 👋")

st.markdown(
    """
    This Application allows you to forecast sales for your retail store by letting you upload your sales data and train a Machine Learning Model, allowing you to find out the expected demand. 
    
    Features:
    - Predict sales at a higher precision through the use of 4 different Machine Learning Models (Time Series).
    - Simulate Inventory Policies
    
    How to use this Application? (👈 Use the Sidebar to navigate this application)
    1. Upload a dataset 📁 (Expected Columns: Date, # of Sales, Product ID (Optional))
    2. Visualise Data 🔎
    3. Forecast Sales 📈
    4. Inventory Policy Simulator ⚙️
    5. Share Feedback 💬
    """
)