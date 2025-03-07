import streamlit as st

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