import streamlit as st

# to run application type this into the terminal " streamlit run App/0_home.py "
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

st.page_link("pages/4_Forecast_Sales.py", label="👈 First Upload Sales Data", icon="📈")
