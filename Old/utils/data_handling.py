import pandas as pd
import streamlit as st
from App.utils.session_manager import SessionManager


def read_uploaded_data(uploaded_dataset):
    if uploaded_dataset.name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_dataset, encoding='unicode-escape', encoding_errors='replace')
        except Exception as e:
            st.error(e)
    elif uploaded_dataset.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_dataset, sheet_name=0)
    return None

def read_file(filepath):
    if filepath.endswith(".csv"):
        try:
            return pd.read_csv(filepath, encoding='unicode-escape', encoding_errors='replace')
        except Exception as e:
            st.error(e)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath, sheet_name=0)
    return None



def map_variables(data):
    columns = data.columns.tolist()
    column_mapping = {
        "date_column" : st.selectbox("Select column for date", columns),
        "product_column": st.selectbox("Select column for Product ID", columns),
        "price_column": st.selectbox("Select column for price", columns),
        "quantity_sold_column": st.selectbox("Select column for quantity sold", columns)
    }


    region = st.selectbox("Select the region", options=["USA", "UK", "N/A"])
    confirm_button = st.button("Confirm Selection")

    if confirm_button:
        SessionManager.set_state("confirm_button_column_map", confirm_button)
        SessionManager.set_state("region", region)

    return column_mapping