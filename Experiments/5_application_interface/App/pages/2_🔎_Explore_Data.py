from datetime import datetime
from time import sleep

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.validators.box import Q1SrcValidator
from pyarrow.compute import fill_null
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
    units_sold_column = st.session_state["units_sold_column"]
    product_column = st.session_state["product_column"]
    unit_price_column = st.session_state["unit_price_column"]
    uploaded_dataset = uploaded_dataset[[date_column, product_column, units_sold_column, unit_price_column]]

    # Fix Dates, Keep Consistent with region, remove extra details
    st.write("Processing Dates in the Correct Format")

    # Identify Duplicates and drop // maybe should keep incase identicial sales occur
    st.write("Handling Duplicates Rows")
    duplicates_exist = uploaded_dataset.duplicated(keep=False)  # `keep=False` marks all duplicates as True
    # Print the duplicate rows
    if uploaded_dataset[duplicates_exist].shape[0] != 0:
        uploaded_dataset = uploaded_dataset.drop_duplicates()
        duplicates_exist = uploaded_dataset.duplicated(keep=False)  # `keep=False` marks all duplicates as True
    if uploaded_dataset[duplicates_exist].shape[0] == 0:
        st.success("Duplicates have been successfully removed")

    # if non-numeric values found, make them null in dates, sales, quantity, price

    # Missing Values

    st.write("Handling Missing Values ")
    # if no dates -> out

    if uploaded_dataset.isnull().values.any():
        rows_with_missing_values = uploaded_dataset[uploaded_dataset.isnull().any(axis=1)]
        st.write("Data contains Missing or Null values")

        product_imputer = SimpleImputer(strategy='most_frequent')
        sales_imputer = SimpleImputer(strategy='median')
        price_imputer = SimpleImputer(strategy='mean')

        rows_with_missing_values = uploaded_dataset[uploaded_dataset.isnull().any(axis=1)]

        # st.write(rows_with_missing_values.head())
        uploaded_dataset.dropna(subset=[date_column], inplace=True)
        ct = ColumnTransformer(
            [("product_imputer", product_imputer, [product_column]),
             ("sales_imputer", sales_imputer, [units_sold_column]),
             ("price_imputer", price_imputer, [unit_price_column])],
            verbose_feature_names_out=False,
            remainder="passthrough")

        ct.set_output(transform="pandas")
        uploaded_dataset = ct.fit_transform(uploaded_dataset)

        if uploaded_dataset.isnull().values.any():
            st.warning("Missing Values Still Exists :/")
        else:
            st.success("Missing Values Processed and Cleaned")
    else:
        st.success("Data contains no missing or null values")
    # run check if format ok plot
    st.write("Processing Dates into ISO8601 format")
    if st.session_state["region"]:
        uploaded_dataset[date_column] = pd.to_datetime(uploaded_dataset[date_column], errors="coerce")
        st.success("Dates have been successfully formatted!")

    else:
        st.warning("Dates were not processed correctly :/")

    st.session_state.uploaded_dataset = uploaded_dataset.sort_values(date_column).reset_index(drop=True)

    st.write("Handling Outliers")

    outlier_indices = []

    if (uploaded_dataset[units_sold_column] < 0).sum() > 0:
        st.session_state.uploaded_dataset = uploaded_dataset[uploaded_dataset[units_sold_column] >= 0].copy()
        uploaded_dataset = uploaded_dataset[uploaded_dataset[units_sold_column] >= 0].copy()

    for product, product_group in uploaded_dataset.groupby(product_column):
        Q1 = product_group[units_sold_column].quantile(0.25)
        Q3 = product_group[units_sold_column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        iqr_outlier_indices = product_group[(product_group[units_sold_column] < Q1 - threshold * IQR)
                                            |
                                            (product_group[units_sold_column] > Q3 + threshold * IQR)].index
        z_scores = np.abs(zscore(product_group[units_sold_column], nan_policy="omit")) # try Median Absolute Deviation (MAD)
        z_score_outlier_indices = product_group[abs(z_scores) > 1.5].index # test different thresholds

        outlier_indices.extend(iqr_outlier_indices)
        outlier_indices.extend(z_score_outlier_indices)

    outlier_indices = list(set(outlier_indices))
    outlier_values = uploaded_dataset[units_sold_column].loc[outlier_indices]


    st.write(f"No. Outliers in Sales Column {outlier_values.shape[0]}")
    st.write(f"Total No. Values in Sales Column {uploaded_dataset[units_sold_column].shape[0]}")
    outlier_proportion = float("%.2f" % (outlier_values.shape[0] / uploaded_dataset[units_sold_column].shape[0] * 100))
    st.write(f"Proportion of Outliers to Total Data {outlier_proportion}%")

    if outlier_proportion > 0: # greater than 10% of dataset
        uploaded_dataset.loc[outlier_indices, units_sold_column] = np.nan
        imputer = SimpleImputer(strategy='mean')

        uploaded_dataset[units_sold_column] = uploaded_dataset.groupby(product_column)[units_sold_column].transform(
            lambda x: round(x.fillna(x.mean()))
        )
    else:
        uploaded_dataset = uploaded_dataset.drop(uploaded_dataset.index[outlier_indices]) # indexing may be a categorical or dates, so this would be an index safe method
        st.success(f"{outlier_values.shape[0]} outliers have been removed")


    # Categorical Variables to Numerical e.g OHE or Label Encoding

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    one_hot_encoder.fit(uploaded_dataset[[product_column]])
    one_hot_encoded = one_hot_encoder.transform(uploaded_dataset[[product_column]])

    uploaded_dataset = uploaded_dataset.drop(columns=[product_column])  # Remove original column
    uploaded_dataset[one_hot_encoder.get_feature_names_out([product_column])] = one_hot_encoded  # Add new columns
    st.write("ONE HOT ENCODER ", uploaded_dataset)



    # Normalisation / Standardisation
    # feature scaling

    # visualise data
    visualise_button = st.button("Visualise Current Sales")
    if visualise_button:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=uploaded_dataset[date_column],y=uploaded_dataset[units_sold_column]))
        figure.update_layout(
                    title_text="Current Sales Data",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    xaxis_title=date_column,
                    yaxis_title=units_sold_column,
                )
        st.plotly_chart(figure)

        st.page_link("pages/3_📈_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")

else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_📁_Upload_Dataset.py", label="👈 Upload The Dataset", icon="📁")
