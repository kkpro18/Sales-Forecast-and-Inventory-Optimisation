from App.Models import inventory_simulator_model
import streamlit as st
def handle_input_details():
    try:
        input = inventory_simulator_model.input_details()
    except Exception as e:
        st.error(e)
        return None
    else:
        return input

