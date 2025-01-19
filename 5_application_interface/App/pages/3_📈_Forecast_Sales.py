import streamlit as st
import pandas as pd

from utils_methods import *

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📁",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

uploaded_dataset = pd.read_csv(get_uploaded_dataset())
date_column = select_date_column(uploaded_dataset)
sales_column = select_sales_column(uploaded_dataset, date_column)

if uploaded_dataset is not None:
    st.multiselect("Select features that can be used to predict sales", uploaded_dataset.columns.drop(date_column).drop(sales_column))

    start_button = st.button("Begin Forecasting Sales")

    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value

        # create acf and dcf plots to identify other parameters for ARIMA model (p,q)

        # fit model

        # forecast

        # calculate performance metrics
        pass

