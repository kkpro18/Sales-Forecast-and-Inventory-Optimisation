import numpy as np
import pandas as pd
import plotly.graph_objects as go
# from stqdm import stqdm
import streamlit as st
from category_encoders import *
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

    # Identify Duplicates and drop // maybe should keep incase identical sales occur
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

        product_impute = SimpleImputer(strategy='most_frequent')
        sales_impute = SimpleImputer(strategy='median')
        price_impute = SimpleImputer(strategy='mean')

        # st.write(rows_with_missing_values.head())
        uploaded_dataset.dropna(subset=[date_column], inplace=True)
        ct = ColumnTransformer(
            [
             ("product_impute", product_impute, [product_column]),
             ("sales_impute", sales_impute, [units_sold_column]),
             ("price_impute", price_impute, [unit_price_column])
            ],
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

    if outlier_proportion > 10: # greater than 10% of dataset
        uploaded_dataset.loc[outlier_indices, units_sold_column] = np.nan
        impute = SimpleImputer(strategy='mean')

        uploaded_dataset[units_sold_column] = uploaded_dataset.groupby(product_column)[units_sold_column].transform(
            lambda x: round(x.fillna(x.mean()))
        )
        st.success(f"{outlier_values.shape[0]} outliers have been processed")
    else:
        uploaded_dataset = uploaded_dataset.drop(outlier_indices) # indexing may be a categorical or dates, so this would be an index safe method
        st.success(f"{outlier_values.shape[0]} outliers have been removed")


    # Categorical Variables to Numerical e.g OHE or Label Encoding

    target_encoder = TargetEncoder(cols=product_column)
    uploaded_dataset["product_encoded"] = target_encoder.fit_transform(uploaded_dataset[product_column], uploaded_dataset[units_sold_column])
    # st.write("Target Encoded Result ", uploaded_dataset.sort_values(units_sold_column, ascending=True).head(5))
    st.success("Product Column has been successfully encoded")

    # Normalisation / Standardisation # not required for arima but required for neural networks

    # feature scaling / engineering

    # visualise data
    visualise_button = st.button("Visualise Current Sales")
    # st.write(uploaded_dataset)

    if visualise_button:
        store_overall_sales_figure = go.Figure()
        store_overall_sales_figure.add_trace(go.Scatter(x=uploaded_dataset[date_column], y=uploaded_dataset[units_sold_column]))
        store_overall_sales_figure.update_layout(
            title_text=f"Current Sales Data for whole store",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column,
            yaxis_title=units_sold_column,
        )
        st.plotly_chart(store_overall_sales_figure)
    if st.button("View Individual Product Sales"):
        for product, product_group in uploaded_dataset.groupby(product_column):
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=product_group[date_column], y=product_group[units_sold_column]))
            figure.update_layout(
                title_text=f"Current Sales Data for Product {product}",
                xaxis=dict(rangeslider=dict(visible=True), type="date"),
                xaxis_title=date_column,
                yaxis_title=units_sold_column,
            )
            st.plotly_chart(figure)
    st.page_link("pages/3_📈_Forecast_Sales.py", label="👈 Next Stage: Forecast Sales", icon="📈")


else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_📁_Upload_Dataset.py", label="👈 Upload The Dataset", icon="📁")
