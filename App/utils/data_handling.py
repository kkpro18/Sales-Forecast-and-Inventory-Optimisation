import pandas as pd
import streamlit as st

def read_uploaded_data(uploaded_dataset):
    if uploaded_dataset.endswith(".csv"):
        return pd.read_csv(uploaded_dataset)
    elif uploaded_dataset.endswith(".xlsx"):
        # elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_dataset)
    return None


def map_columns_to_variables(data):
    column_map = {
        "date_column" : st.selectbox("Select column for date", data.columns),
        "product_column": st.selectbox("Select column for Product ID", data.columns),
        "price_column": st.selectbox("Select column for price", data.columns),
        "quantity_sold_column": st.selectbox("Select column for quantity sold", data.columns)
    }
    return column_map