import streamlit as st
from App.utils.session_manager import (SessionManager)
from App.utils.visualise_sales import visualise_storewide_sales, visualise_individual_product_sales

st.set_page_config(
    page_title="Visualise Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Preprocess Your Sales Data")
st.write(
    """Here you can Clean (pre-process) the data!""")

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
    if st.button("View Store-Wide Sales"):
        visualise_storewide_sales(data, column_mapping)
    if st.button("View Each Product Sales"):
        if not SessionManager.is_there("product_index"):
            SessionManager.set_state("product_index", 0)
        visualise_individual_product_sales(data, column_mapping)

    st.page_link("pages/4_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")



