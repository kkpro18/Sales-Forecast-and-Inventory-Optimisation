import streamlit as st
from App.Models.data_model import read_uploaded_data, map_variables, select_region
from App.utils.session_manager import SessionManager


def handle_uploaded_file(uploaded_dataset):
    """
    Handles the uploaded file, returns the data as a pandas DataFrame.
    """

    try:
        read_uploaded_data(uploaded_dataset)
    except Exception as e:
        st.error(e)
        return None
    else:
        return SessionManager.get_state("data")

def handle_column_mapping(data):
    """
    Handles the column mapping for the uploaded data and returns the column mapping as a dictionary.
    """

    try:
        map_variables(data)
    except Exception as e:
        st.error(e)
        return None
    else:
        return SessionManager.get_state("column_mapping")

def handle_region_selection():
    """
    Handles the region selection for the uploaded data and returns the selected region.
    """

    try:
        select_region()
    except Exception as e:
        st.error(e)
        return None
    else:
        return SessionManager.get_state("region")
