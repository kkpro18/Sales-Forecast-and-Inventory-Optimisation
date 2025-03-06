import streamlit as st

class SessionManager:
    def set_state(key, value):
        # sets a global variable in the session
        st.session_state[key] = value

    def get_state(key):
        # retrieves the value of the variable
        return st.session_state.get(key)

    def clear_states(self):
        st.session_state.clear()
    