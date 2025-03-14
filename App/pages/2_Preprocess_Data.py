import time
import streamlit as st
from App.utils.session_manager import SessionManager
import pandas as pd

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
    data_as_dictionary = data.to_dict(orient='records')


    json_response = SessionManager.fast_api_call("format_dates_call", data = data_as_dictionary, column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success("Successfully Formatted Dates")
    else:
        st.error(json_response.text)

    st.write("Handling Missing Values ")
    json_response = SessionManager.fast_api_call("handle_missing_values_call", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success("Successfully Handled Missing Values")
    else:
        st.error(json_response.text)

    st.write("Handling Outliers")
    json_response = SessionManager.fast_api_call("handle_outliers_call", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success("Successfully Handled Outliers")
    else:
        st.error(json_response.text)

    st.write("Numerically Encoding Product ID (Unique Identifier)")
    json_response = SessionManager.fast_api_call("encode_product_column_call", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success("Successfully Encoded Product ID")
    else:
        st.error(json_response.text)

    data = pd.DataFrame(json_response.json())
    data[column_mapping["date_column"]] = pd.to_datetime(data[column_mapping["date_column"]], errors="coerce")
    data[column_mapping["date_column"]] = data[column_mapping["date_column"]].dt.tz_localize(None)

    SessionManager.set_state("preprocess_data_complete", True)
    SessionManager.set_state("data", data)

    st.subheader("Preprocessed Data: ")
    st.dataframe(SessionManager.get_state("data"))
    st.balloons()

    preprocessed_data_csv = data.to_csv(index=False)
    st.download_button(
        label="Download the Preprocessed Data as a CSV File",
        data=preprocessed_data_csv,
        file_name="preprocessed_data.csv",
    )
    time.sleep(3)
    # st.switch_page("pages/3_Visualise_Data.py")

