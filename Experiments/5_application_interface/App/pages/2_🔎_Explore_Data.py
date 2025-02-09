from datetime import datetime
from time import sleep

import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
    product_column = st.session_state["product_column"]
    price_column = st.session_state["unit_price_column"]

    # Pre Process

    # Fix Dates, Keep Consistent with region, remove extra details
    st.write("Processing Dates in the Correct Format")

    # Identify Duplicates and drop
    st.write("Handling Duplicates Rows")
    duplicates_exist = uploaded_dataset.duplicated(keep=False)  # `keep=False` marks all duplicates as True
    # Print the duplicate rows
    if uploaded_dataset[duplicates_exist].shape[0] != 0:
        st.write("Duplicate rows: ", uploaded_dataset[duplicates_exist])
        uploaded_dataset = uploaded_dataset.drop_duplicates()
        duplicates_exist = uploaded_dataset.duplicated(keep=False)  # `keep=False` marks all duplicates as True
    if uploaded_dataset[duplicates_exist].shape[0] == 0:
        st.success("Duplicates have been successfully removed")

    # Missing Values
    st.write("Handling Missing Values ")
    # if no dates -> out

    if uploaded_dataset.isnull().values.any():
        rows_with_missing_values = uploaded_dataset[uploaded_dataset.isnull().any(axis=1)]
        st.write("Data contains Missing or Null values")
        st.write(rows_with_missing_values.head())

        product_imputer = SimpleImputer(strategy='most_frequent')
        sales_imputer = SimpleImputer(strategy='median')
        price_imputer = SimpleImputer(strategy='mean')

        product_imputer = SimpleImputer(strategy='most_frequent')
        sales_imputer = SimpleImputer(strategy='median')
        price_imputer = SimpleImputer(strategy='mean')

        rows_with_missing_values = uploaded_dataset[uploaded_dataset.isnull().any(axis=1)]

        # st.write(rows_with_missing_values.head())
        uploaded_dataset.dropna(subset=date_column, inplace=True)
        ct = ColumnTransformer(
            [("product_imputer", product_imputer, [product_column]),
             ("sales_imputer", sales_imputer, [sales_column]),
             ("price_imputer", price_imputer, [price_column])],
            verbose_feature_names_out=False,
            remainder="passthrough")

        ct.set_output(transform="pandas")
        uploaded_dataset = ct.fit_transform(uploaded_dataset)

        st.write(uploaded_dataset)

        st.write("FINAL CHECK \n", uploaded_dataset[uploaded_dataset.isnull().any(axis=1)])

        if uploaded_dataset.isnull().values.any():
            st.warning("Missing Values Still Exists :/")
        else:
            st.success("Missing Values Processed and Cleaned")


    else:
        st.write("Data contains no missing or null values")

    if st.session_state["selected_region"] == "UK":
        for _ in stqdm(range(uploaded_dataset.shape[0])):
            uploaded_dataset.at[_, date_column] = pd.to_datetime(uploaded_dataset[date_column][_], dayfirst=True).strftime("%d/%m/%Y")
        st.success("Dates have been successfully processed in the correct format")

    elif st.session_state["selected_region"] == "USA":
        for _ in stqdm(range(uploaded_dataset.shape[0])):
            uploaded_dataset.at[_, date_column] = pd.to_datetime(uploaded_dataset[date_column][_], dayfirst=False).strftime("%m/%d/%Y")
        st.success("Dates have been successfully processed in the correct format")
    else:
        st.warning("Dates were not processed correctly :/")


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
