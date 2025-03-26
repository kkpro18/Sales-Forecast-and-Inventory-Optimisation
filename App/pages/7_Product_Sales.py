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
    column_mapping = SessionManager.get_state("column_mapping")
    get_seasonality()

    st.markdown("### Individual Product Sales Forecasting")

    # Product Sales
    product_grouped_sales = SessionManager.get_state("daily_product_grouped_sales")
    product_grouped_sales = product_grouped_sales.groupby(column_mapping["product_column"])
    product_names = list(product_grouped_sales.groups.keys())

    left_column, right_column = st.columns(2)
    previous_button = left_column.button("Previous", disabled=SessionManager.get_state("product_index") == 0)
    next_button = right_column.button("Next",
                                      disabled=SessionManager.get_state("product_index") >= len(product_names) - 1)

    if previous_button:
        if SessionManager.get_state("product_index") > 0:
            SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)
        product_name = product_names[SessionManager.get_state('product_index')]
        st.write(f"### {product_name} Samples: {len(product_grouped_sales.get_group(product_name))}")
        product_data = product_grouped_sales.get_group(product_name)
        asyncio.run(
            predict_sales(product_data, column_mapping, product_name=product_names[SessionManager.get_state("product_index")]))


    elif next_button:
        SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
        product_name = product_names[SessionManager.get_state('product_index')]
        st.write(f"### {product_name} Samples: {len(product_grouped_sales.get_group(product_name))}")
        product_data = product_grouped_sales.get_group(product_name)
        asyncio.run(
            predict_sales(product_data, column_mapping, product_name=product_names[SessionManager.get_state("product_index")]))

    # for product_name in product_names:
    #     st.write(f"### {product_name}")
    #     product_data = product_grouped_sales.get_group(product_name)
    #     data = product_data
    #     st.dataframe(product_data)
    #     asyncio.run(predict_sales(data, column_mapping, product_name=product_name))
    #     next_product_prediction = False
    #     while not next_product_prediction:
    #         next_product_prediction = st.button("Next Product", key=uuid.uuid4())