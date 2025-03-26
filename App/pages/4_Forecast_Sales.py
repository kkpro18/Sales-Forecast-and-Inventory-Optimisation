import uuid

import joblib
import pandas as pd
import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.forecasting_sales import get_seasonality, split_training_testing_data, predict_sales
import asyncio

st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)

st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not SessionManager.get_state("preprocess_data_complete"):
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    store_wide_sales = SessionManager.get_state("daily_store_sales").head(int(len(SessionManager.get_state("daily_store_sales")) * 0.25))
    column_mapping = SessionManager.get_state("column_mapping")
    get_seasonality()

    if st.button("Begin Forecasting Sales"):
        st.markdown("### Store Wide Sales Forecasting")
        st.dataframe(store_wide_sales)
        data = store_wide_sales
        asyncio.run(predict_sales(data, column_mapping, product_name=None))

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")
