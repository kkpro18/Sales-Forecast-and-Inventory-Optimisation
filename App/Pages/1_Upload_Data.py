import streamlit as st
from App.utils.session_manager import SessionManager
from App.Controllers import data_controller

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

    # initialize session state variables
    if not SessionManager.is_there("data"):
        SessionManager.set_state("data", None)
    if not SessionManager.is_there("column_mapping"):
        SessionManager.set_state("column_mapping", None)
    if not SessionManager.is_there("region"):
        SessionManager.set_state("region", None)


    data = data_controller.handle_uploaded_file(uploaded_dataset)
    if data is not None:
        st.success("File uploaded successfully!")
        st.dataframe(data.head())

    st.subheader("Map Columns to the Expected Variables")
    column_mapping = data_controller.handle_column_mapping(data)
    if column_mapping is not None:
        st.success("Columns mapped successfully!")

    region = data_controller.handle_region_selection()
    if region is not None:
        st.success("Region Selected successfully!")

    st.write("Selected Region: ", SessionManager.get_state("region"))
    st.write("Column Mapping: ", SessionManager.get_state("column_mapping"))

    if SessionManager.get_state("data") is not None and SessionManager.get_state("column_mapping") is not None:
        SessionManager.set_state("data", data[column_mapping.values()])


    st.page_link("pages/2_Preprocess_Data.py", label="👈 First Preprocess Sales Data", icon="🧼")

