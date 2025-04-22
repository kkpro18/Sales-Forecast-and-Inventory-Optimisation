import uuid

import pandas as pd
import streamlit as st
from App.utils.session_manager import SessionManager


def read_uploaded_data(uploaded_dataset):
    if uploaded_dataset.name.endswith(".csv"):
        data = pd.read_csv(uploaded_dataset, encoding='unicode-escape', encoding_errors='replace')
        SessionManager.set_state("data", data)
        return data
    elif uploaded_dataset.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_dataset, sheet_name=0)
        SessionManager.set_state("data", data)
        return data
    return None


def map_variables(data):
    columns = data.columns.tolist()
    column_mapping = {
        "date_column": st.selectbox("Select column for date", columns),
        "product_column": st.selectbox("Select column for Product ID", columns),
        "price_column": st.selectbox("Select column for price", columns),
        "quantity_sold_column": st.selectbox("Select column for quantity sold", columns)
    }

    confirm_button = st.button("Confirm Column Map")
    if confirm_button:
        SessionManager.set_state("column_mapping", column_mapping)
        return column_mapping

def select_region():
    region = st.selectbox("Select the region", options=["USA", "UK", "N/A"])

    confirm_button = st.button("Confirm Region")
    if confirm_button:
        SessionManager.set_state("region", region)
        return region

def read_file(filepath):  # used to read other files e.g macroeconomic data
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath, encoding='unicode-escape', encoding_errors='replace')
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath, sheet_name=0)
    else:
        print("Unsupported file format. Please upload a CSV or Excel file.")
    return None