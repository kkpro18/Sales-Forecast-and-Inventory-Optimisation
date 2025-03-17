import os
import shutil

import streamlit as st
import requests


class SessionManager:
    @staticmethod
    def set_state(key, value):
        # sets a global variable in the session
        st.session_state[key] = value

    @staticmethod
    def get_state(key):
        # retrieves the value of the variable
        return st.session_state[key]

    @staticmethod
    def clear_states():
        st.session_state.clear()

    @staticmethod
    def is_there(key):
        if key in st.session_state:
            return True
        else:
            return False

    @staticmethod
    def fast_api(endpoint, **kwargs):
        url = f'http://127.0.0.1:8000/{endpoint}'
        response = requests.post(
            url,
            json=kwargs
        )
        return response

    @staticmethod
    def cleanup():
        # file_paths = ["models", "product_sales", "store_sales"] # cleans up space
        file_paths = ["product_sales", "store_sales"]  # cleans up space
        for path in file_paths:
            if os.path.exists(path):
                shutil.rmtree(path) # remove directory
                os.makedirs(path)  # creates directory ready for next run