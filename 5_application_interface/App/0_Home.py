import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.pyplot import xlabel

# to run application type this into the terminal "streamlit run 5_application_interface/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Sales Forecasting and Inventory Optimisation Service",
    page_icon="🏠",
)
st.sidebar.success("Application Launched Successfully")

st.write("# Welcome to this Application! 👋")

st.markdown(
    """
    This Application aims to forecast sales, allowing you to know how much demand you can expect. 
    If the data you upload is detailed with product details in the transactions, expect to see even more granular insights.
    Identify optimal levels of stock for a product.
    Predict sales at a higher precision through the choice of 4 different Machine Learning Models (Time Series).
    
    How to use this Application? (Use the Sidebar to navigate this application)
    1. Upload a dataset 📁
    2. Visualise Data 🔎
    3. Forecast Sales 📈
    4. Inventory Policy Simulator ⚙️
    5. Share Feedback 💬
    """
)