import pandas as pd
import streamlit as st
from utils.session_manager import SessionManager

def read_uploaded_data(uploaded_dataset):
    if uploaded_dataset.name.endswith(".csv"):
        return pd.read_csv(uploaded_dataset, encoding="unicode_escape")
    elif uploaded_dataset.name.endswith(".xlsx"):
        # elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_dataset, encoding="unicode_escape")
    return None

def map_columns_to_variables(data):
    column_mapping = {
        "date_column" : st.selectbox("Select column for date", data.columns),
        "product_column": st.selectbox("Select column for Product ID", data.columns),
        "price_column": st.selectbox("Select column for price", data.columns),
        "quantity_sold_column": st.selectbox("Select column for quantity sold", data.columns)
    }
    SessionManager.set_state("confirm_button_column_map", st.button("Confirm Column Mapping: "))
    return column_mapping