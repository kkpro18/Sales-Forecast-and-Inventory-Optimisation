import streamlit as st

from utils.session_manager import SessionManager

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
    This Application allows you to forecast sales for your ecommerce store by letting you upload your sales data and train a Machine Learning Model, allowing you to find out the expected demand. 

    Features:
    - Predict sales at a higher precision through the use of 4 different Machine Learning Models (Time Series).
    - Simulate Inventory Policies

    How to use this Application? (👈 Use the Sidebar to navigate this application)
    1. Upload a dataset 📁| CSV / XLSX Format | Required Data Columns : Invoice Date, Product ID, Unit Price, # Sold
    2. Preprocess Data 🧼
    3. Visualise Data 🔎
    3. Forecast Sales 📈
    4. Inventory Policy Simulator ⚙️
    5. Share Feedback 💬
    """
)

if not SessionManager.is_there("preprocess_data_complete"):
    SessionManager.set_state("preprocess_data_complete", False)

st.write("Full Session State Variables: ")
st.write(st.session_state)

st.page_link("pages/1_Upload_Data.py", label="👈 First Upload Sales Data", icon="📁")

