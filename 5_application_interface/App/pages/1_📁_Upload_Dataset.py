import pandas as pd
import streamlit as st
from functions import utils_methods as utils

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
)

st.markdown("# Upload Dataset")
st.sidebar.header("1. Upload Dataset")

st.write("""Here you can upload your dataset in CSV format. You can click to upload or drag a CSV file over to this page.""")
uploaded_dataset = st.file_uploader("Upload your sales data", type="csv")

if uploaded_dataset is not None:
    df = pd.read_csv(uploaded_dataset)
    utils.update_dataset(uploaded_dataset)
    st.write(df)
    st.write("Select Columns to Use For the Forecast")
    date_column = st.selectbox("Select the Column for Dates", df.columns)
    utils.update_date_column(date_column)
    sales_column = st.selectbox("Select the Column for Sales", df.columns.drop(date_column))
    utils.update_sales_column(sales_column)

