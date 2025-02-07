from datetime import datetime
from time import sleep

import pandas as pd
import plotly.graph_objects as go
from stqdm import stqdm
import streamlit as st
from tqdm import tqdm

# to run application type this into the terminal "streamlit run experiments/5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Explore Data",
    page_icon="🔎",
    layout="wide",
)
st.markdown("# Explore Your Sales")
st.write(
    """Here you can view your dataset and also pre-process it prior to the Sales Forecasting.""")

tqdm.pandas() # for progress_apply

if 'uploaded_dataset' in st.session_state:
    uploaded_dataset = st.session_state["uploaded_dataset"]
    date_column = st.session_state["date_column"] # ensure its in correct form
    sales_column = st.session_state["units_sold_column"]
    # Pre Process
    # Fix Dates, Keep Consistent with region, remove extra details
    st.write("Processing Dates in the Correct Format")

    if st.session_state["selected_region"] == "UK":
        for _ in stqdm(range(uploaded_dataset.shape[0])):
            uploaded_dataset.at[_, date_column] = pd.to_datetime(uploaded_dataset[date_column][_]).strftime("%d/%m/%Y")
        st.success("Dates have been successfully processed")

    elif st.session_state["selected_region"] == "USA":
        for _ in stqdm(range(uploaded_dataset.shape[0])):
            uploaded_dataset.at[_, date_column] = pd.to_datetime(uploaded_dataset[date_column][_]).strftime("%m/%d/%Y")
        st.success("Dates have been successfully processed")
    else:
        st.warning("Dates were not processed correctly :/")

    st.write(uploaded_dataset[date_column].head())


    # Identify Duplicates and Merge
    # Missing Values
    # Normalisation / Standardisation ?
    # Categorical Variables to Numerical e.g OHE or Label Encoding
    # Column Specific Date to be Numerical, Product ID to be Numerical, Data to be seperated per product,
    # Outliers/anomalies



    # visualise data
    visualise_button = st.button("Visualise Current Sales")
    if visualise_button:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=uploaded_dataset[date_column],y=uploaded_dataset[sales_column]))
        figure.update_layout(
                    title_text="Current Sales Data",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    xaxis_title=date_column,
                    yaxis_title=sales_column,
                )
        st.plotly_chart(figure)

        st.page_link("pages/3_📈_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")

else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_📁_Upload_Dataset.py", label="👈 Upload The Dataset", icon="📁")
