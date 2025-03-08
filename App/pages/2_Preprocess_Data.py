import time

import pandas as pd
import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.data_preprocessing import format_dates, handle_outliers, encode_product_column, handle_missing_values
from utils.session_manager import SessionManager

st.set_page_config(
    page_title="Preprocess Data",
    page_icon="🧼",
    layout="wide",
)
st.markdown("# Preprocess Your Sales Data")
st.write(
    """Here you can Clean (pre-process) the data!""")


if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
else:
    data = SessionManager.get_state("data")
    column_mapping = SessionManager.get_state("column_mapping")

    st.write("Processing Dates in the Correct Format")
    data = data.to_dict(orient="records")
    data = SessionManager.flask_api_call("format_dates_call", data, column_mapping)

    st.write("Handling Missing Values ")
    data = SessionManager.flask_api_call("handle_missing_values_call", data["processed_data"], column_mapping)

    st.write("Handling Outliers")
    data = SessionManager.flask_api_call("handle_outliers_call", data["processed_data"], column_mapping)

    st.write("Numerically Encoding Product ID (Unique Identifier)")
    data = SessionManager.flask_api_call("encode_product_column_call", data["processed_data"], column_mapping)

    SessionManager.set_state("preprocess_data_complete", True)

    st.subheader("Preprocessed Data: ")
    data = pd.DataFrame(data)
    st.dataframe(data)
    st.balloons()

    preprocessed_data_csv = data.to_csv(index=False)
    st.download_button(
        label="Download the Preprocessed Data as a CSV File",
        data=preprocessed_data_csv,
        file_name="preprocessed_data.csv",
    )
    time.sleep(1)
    st.switch_page("pages/3_Visualise_Data.py")

