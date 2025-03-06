import streamlit as st
from App.utils.session_manager import *
from App.utils.data_preprocessing import *

st.set_page_config(
    page_title="Preprocess Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Preprocess Your Sales Data")
st.write(
    """Here you can Clean (pre-process) the data!""")

data = SessionManager.get_state("data")
column_map = SessionManager.get_state("column_map")

if data is None or column_map is None:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/upload_data.py", label="👈 Upload The Dataset", icon="📁")
else:
    st.write("Processing Dates in the Correct Format")
    data = format_dates(data, column_map)

    st.write("Handling Outliers")
    data = handle_outliers(data, column_map)

    st.write("Handling Missing Values ")
    data = handle_missing_values(data, column_map)

    st.write("Numerically Encoding Product ID (Unique Identifier)")
    data = encode_product_column(data, column_map)

    SessionManager.set_state("preprocess_data_complete", True)
    st.dataframe(data)
