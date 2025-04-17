from App.Models import data_visualisation_model
import streamlit as st

def handle_store_wide_sales_visualisation(store_data, column_mapping):
    """
    Handles the store-wide sales visualisation for the store sales and displays the visualisation.
    """
    try:
        data_visualisation_model.visualise_storewide_sales(store_data, column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        st.success("Store-wide sales visualised successfully.")
        return True

def handle_product_level_sales_visualisation(product_data, column_mapping):
    """
    Handles the product-level sales visualisation for the product sales and displays the visualisation.
    """
    try:
        data_visualisation_model.visualise_individual_product_sales(product_data, column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        st.success("Product-level sales visualised successfully.")
        return True