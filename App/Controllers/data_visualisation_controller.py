from App.Models import data_visualisation_model
import streamlit as st

def handle_store_wide_sales_visualisation(store_data, column_mapping):
    try:
        data_visualisation_model.visualise_storewide_sales(store_data, column_mapping)
    except Exception as e:
        st.error(e)
    else:
        st.success("Store-wide sales visualised successfully.")

def handle_product_level_sales_visualisation(product_data, column_mapping):
    try:
        data_visualisation_model.visualise_individual_product_sales(product_data, column_mapping)
    except Exception as e:
        st.error(e)
    else:
        st.success("Product-level sales visualised successfully.")