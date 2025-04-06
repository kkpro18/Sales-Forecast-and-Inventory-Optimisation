import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.data_handling import read_uploaded_data, map_variables

# run app by "python -m streamlit run App/0_Home.py"

st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
    layout="wide",
)

st.markdown("# Upload Dataset")
st.write("""Here you can upload your dataset in CSV format. You can click to upload or drag a CSV file over to this page.""")

uploaded_dataset = st.file_uploader("Upload your sales data", type=["csv", "xlsx"])

if uploaded_dataset is not None:
    data = read_uploaded_data(uploaded_dataset=uploaded_dataset)
    if not SessionManager.is_there("data"):
        SessionManager.set_state("data", data)
    st.success("File uploaded successfully!")
    st.dataframe(data.head())

    st.subheader("Map Columns to the Expected Variables")
    SessionManager.set_state("confirm_button_column_map", None)
    SessionManager.set_state("region", None)
    column_mapping = map_variables(data)
    if SessionManager.get_state("confirm_button_column_map"):
        SessionManager.set_state("column_mapping", column_mapping)
        st.success("Columns mapped successfully!")
    st.write("Selected Region: ", SessionManager.get_state("region"))
    st.write("Column Mapping: ", column_mapping)


    if SessionManager.is_there("data") and SessionManager.is_there("column_mapping"):
        SessionManager.set_state("data", data[column_mapping.values()])


    st.page_link("pages/2_Preprocess_Data.py", label="👈 First Preprocess Sales Data", icon="🧼")

