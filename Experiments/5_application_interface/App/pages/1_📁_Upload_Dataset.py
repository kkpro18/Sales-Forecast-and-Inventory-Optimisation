import pandas as pd
import streamlit as st

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
)

st.markdown("# Upload Dataset")
st.write("""Here you can upload your dataset in CSV format. You can click to upload or drag a CSV file over to this page.""")

uploaded_dataset = st.file_uploader("Upload your sales data", type="csv")

if uploaded_dataset is not None:
    raw_df = pd.read_csv(uploaded_dataset)
    st.session_state['uploaded_dataset'] = raw_df
    st.success("File uploaded successfully!")

if 'uploaded_dataset' in st.session_state:
    st.write(st.session_state['uploaded_dataset'].head())

    st.write("Select Columns to Use For the Forecast")

    columns = st.session_state['uploaded_dataset'].copy().columns.tolist()
    columns.insert(len(columns), "None")

    date_column = st.selectbox("Select the Column for Dates", columns)
    st.session_state["date_column"] = date_column
    product_details_column = st.selectbox("Select the Column for Product Name / Category / Details", columns)
    st.session_state["product_details_column"] = product_details_column

    sales_column = st.selectbox("Select the Column for Quantity Sold", st.session_state['uploaded_dataset'].columns.drop(date_column))
    st.session_state["sales_column"] = sales_column



    st.write("👈 Next Stage: Visualise The Dataset")