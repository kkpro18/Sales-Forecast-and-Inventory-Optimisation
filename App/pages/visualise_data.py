import streamlit as st
from App.utils.session_manager import SessionManager

st.set_page_config(
    page_title="Visualise Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Preprocess Your Sales Data")
st.write(
    """Here you can Clean (pre-process) the data!""")

data = SessionManager.get_state("data")
column_map = SessionManager.get_state("column_map")
preprocess_data_complete = SessionManager.get_state("preprocess_data_complete")

if data is None or column_map is None:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/upload_data.py", label="👈 Upload The Dataset", icon="📁")
elif not preprocess_data_complete:
    st.warning("Dataset has not been pre-processed, 👈 Please Preprocess it ")
    st.page_link("pages/preprocess_data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    if st.button("View Store-Wide Sales"):
        visualise_storewide_sales(data, column_map)
    if st.button("View Each Product Sales"):
        visualise_individual_product_sales(data, column_map)

    st.page_link("pages/forecast_sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")



