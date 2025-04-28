import streamlit as st
import atexit
from App.utils.session_manager import SessionManager

# run app by "python -m streamlit run App/0_Home.py"

st.set_page_config(
    page_title="Sales Forecasting App",
    page_icon="🏠",
    layout="wide",
)
st.sidebar.success("Application Launched Successfully")

st.write("# Sales Forecasting App ")

st.markdown(
    """
    This Application allows you to forecast sales and optimise inventory policies for your eCommerce store. Get Started by uploading your sales data!

    Functionality Included in this Application are:
    - 🤖 Autonomous Data Preprocessing Pipeline
    - 📊 Visualise Sales Data for Store Sales and Product Sales
    - 📈 Forecast Sales using Time Series Models like ARIMA, ARIMAX, SARIMA, SARIMAX and Prophet
    - 💾 Export Trained Models for later use
    - ➕ Calculate Optimal Inventory Policy for Seasonable and Non Seasonal Items using proprietary Mathematical Models
    - 🎮 Simulate Inventory Policies
    
    Key Design Features 🌟:
    - 🏗️ This Application is developed using the MVC Architecture
    - 🌐 This Application leverages Streamlit framework for the user-interface
    - ⚡ This Application leverages FastAPI framework to handle the PreProcessing and Forecasting computation.
    

    How to use this Application? (👈 Use the Sidebar to navigate this application)
    1. Upload a dataset 📁| CSV / XLSX Format | Required Data Columns : Invoice Date, Product ID, Unit Price, # Sold
    2. Preprocess Data 🧼
    3. Visualise Data 🔎
    3. Forecast Sales 📈
    4. Optimise Inventory Policies 🔧
    4. Inventory Policy Simulator ⚙️
    5. Share Feedback 💬
    """
)

if not SessionManager.is_there("preprocess_data_complete"):
    SessionManager.set_state("preprocess_data_complete", False)

st.write("Your Data: ")
st.write(st.session_state)

st.page_link("pages/1_Upload_Data.py", label="👈 First Upload Sales Data", icon="📁")

atexit.register(SessionManager.cleanup) # Clean up Files Created by the App
