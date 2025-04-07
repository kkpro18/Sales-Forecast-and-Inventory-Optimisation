import streamlit as st
from App.utils.session_manager import (SessionManager)
from App.Controllers import data_visualisation_controller

st.set_page_config(
    page_title="Visualise Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Visualise Your Sales Data")
st.write(
    """Here you can see the data visually!""")

if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset and then Preprocess")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not SessionManager.get_state("preprocess_data_complete"):
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process your data", icon="📁")
else:
    store_data = SessionManager.get_state("train_daily_store_sales")
    product_data = SessionManager.get_state("train_daily_product_sales")
    column_mapping = SessionManager.get_state("column_mapping")

    st.header("View Sales Across the Store: ")
    data_visualisation_controller.handle_store_wide_sales_visualisation(store_data, column_mapping)

    st.header("View Each Product Sales: ")
    data_visualisation_controller.handle_product_level_sales_visualisation(product_data, column_mapping)

    st.page_link("pages/4_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")



