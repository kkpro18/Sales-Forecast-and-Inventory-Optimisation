import streamlit as st
from App.utils.session_manager import (SessionManager)
from App.utils.visualise_sales import visualise_storewide_sales, visualise_individual_product_sales

st.set_page_config(
    page_title="Visualise Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Visualise Your Sales Data")
st.write(
    """Here you can see the data visually!""")

data = SessionManager.get_state("data")
column_mapping = SessionManager.get_state("column_mapping")
preprocess_data_complete = SessionManager.get_state("preprocess_data_complete")

if data is None or column_mapping is None:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not preprocess_data_complete:
    st.warning("Dataset has not been pre-processed, 👈 Please Preprocess it ")
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    st.header("View Sales Across the Store: ")
    visualise_storewide_sales(data, column_mapping)

    st.header("View Each Product Sales: ")
    visualise_individual_product_sales(data, column_mapping)


    st.page_link("pages/4_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")



