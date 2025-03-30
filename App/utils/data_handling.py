import pandas as pd
import streamlit as st
from utils.session_manager import SessionManager
import chardet

def read_uploaded_data(uploaded_dataset):

    if uploaded_dataset.name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_dataset, encoding='unicode-escape', encoding_errors='replace')
        except Exception as e:
            st.error(e)
    elif uploaded_dataset.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_dataset, sheet_name=0)
    return None

def map_columns_to_variables(data):
    columns = data.columns.tolist()
    column_mapping = {
        "date_column" : st.selectbox("Select column for date", columns),
        "product_column": st.selectbox("Select column for Product ID", columns),
        "price_column": st.selectbox("Select column for price", columns),
        "quantity_sold_column": st.selectbox("Select column for quantity sold", columns)
    }
    SessionManager.set_state("confirm_button_column_map", st.button("Confirm Column Mapping: "))
    return column_mapping