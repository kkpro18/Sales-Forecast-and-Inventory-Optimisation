import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.pyplot import xlabel

# to run application type this into the terminal "streamlit run 5_application_draft/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Share Feedback",
    page_icon="💬",
    layout="wide",
)
st.markdown("# Provide Feedback")
st.write("""Share Your Feedback""")

feedback = st.feedback(options="stars")
feedback_text = st.text_area("Enter feedback here")

if len(feedback_text) > 0:
    st.write(f"You wrote {len(feedback_text)} characters.")

submit_button = st.button("Submit Feedback")

if submit_button:
    st.success("Feedback submitted")

