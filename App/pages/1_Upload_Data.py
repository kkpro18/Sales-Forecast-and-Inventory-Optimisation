import streamlit as st
from App.utils.session_manager import *
from App.utils.data_handling import *

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
    layout="wide",
)

st.markdown("# Upload Dataset")
st.write("""Here you can upload your dataset in CSV format. You can click to upload or drag a CSV file over to this page.""")

uploaded_dataset = st.file_uploader("Upload your sales data", type=["csv", "xlsx"])

if uploaded_dataset is not None:
    data = read_uploaded_data(uploaded_dataset)
    SessionManager.set_state("data", data)
    st.success("File uploaded successfully!")
    st.dataframe(data.head())

    st.subheader("Map Columns to the Expected Variables")
    column_map = map_columns_to_variables(data)
    SessionManager.set_state("column_mapping", column_map)
    st.success("Columns mapped successfully!")
    st.write(column_map)
