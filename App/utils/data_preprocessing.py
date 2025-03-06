import numpy as np
import streamlit as st
import pandas as pd
from category_encoders import TargetEncoder
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def format_dates(data, column_map):
    date_column = column_map["date_column"]
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[date_column] = data[date_column].fillna(method='ffill')

    st.success("Dates have been successfully formatted!")
    return data

def handle_outliers(data, column_map):
    outlier_indices = []
    quantity_sold_column = column_map["quantity_sold_column"]
    product_column = column_map["product_column"]
    # consider pricing as possibly typos

    if (data[quantity_sold_column] < 0).sum() > 0:
        st.session_state.data = data[data[quantity_sold_column] >= 0].copy()
        data = data[data[quantity_sold_column] >= 0].copy()

    for product, product_group in data.groupby(product_column):
        Q1 = product_group[quantity_sold_column].quantile(0.25)
        Q3 = product_group[quantity_sold_column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = 1.5
        iqr_outlier_indices = product_group[(product_group[quantity_sold_column] < Q1 - threshold * IQR)
                                            |
                                            (product_group[quantity_sold_column] > Q3 + threshold * IQR)].index
        z_scores = np.abs(
            zscore(product_group[quantity_sold_column], nan_policy="omit"))  # try Median Absolute Deviation (MAD)
        z_score_outlier_indices = product_group[abs(z_scores) > 1.5].index  # test different thresholds

        outlier_indices.extend(iqr_outlier_indices)
        outlier_indices.extend(z_score_outlier_indices)

    outlier_indices = list(set(outlier_indices))
    outlier_values = data[quantity_sold_column].loc[outlier_indices]

    st.write(f"No. Outliers in Sales Column {outlier_values.shape[0]}")
    st.write(f"Total No. Values in Sales Column {data[quantity_sold_column].shape[0]}")
    outlier_proportion = float("%.2f" % (outlier_values.shape[0] / data[quantity_sold_column].shape[0] * 100))
    st.write(f"Proportion of Outliers to Total Data {outlier_proportion}%")

    if outlier_proportion > 10:  # greater than 10% of dataset
        data.loc[outlier_indices, quantity_sold_column] = np.nan
        impute = SimpleImputer(strategy='mean')

        data[quantity_sold_column] = data.groupby(product_column)[quantity_sold_column].transform(
            lambda x: round(x.fillna(x.mean()))
        )
        st.success(f"{outlier_values.shape[0]} outliers have been processed")
    else:
        data = data.drop(outlier_indices)  # indexing may be a categorical or dates, so this would be an index safe method
        st.success(f"{outlier_values.shape[0]} outliers have been removed")

    return data

def handle_missing_values(data, column_map):
    quantity_sold_column = column_map["quantity_sold_column"]
    product_column = column_map["product_column"]
    price_column = column_map["price_column"]

    if data.isnull().values.any():
        # rows_with_missing_values = data[data.isnull().any(axis=1)]
        st.write("Data contains Missing or Null values")

        product_impute = SimpleImputer(strategy='most_frequent')
        sales_impute = SimpleImputer(strategy='median')
        price_impute = SimpleImputer(strategy='mean')

        ct = ColumnTransformer(
            [
                ("product_impute", product_impute, [product_column]),
                ("sales_impute", sales_impute, [quantity_sold_column]),
                ("price_impute", price_impute, [price_column])
            ],
            verbose_feature_names_out=False,
            remainder="passthrough")

        ct.set_output(transform="pandas")
        data = ct.fit_transform(data)

        if data.isnull().values.any():
            st.warning("Missing Values Still Exists :/")
        else:
            st.success("Missing Values Processed and Cleaned")
    else:
        st.success("Data contains no missing or null values")
    return data

def encode_product_column(data, column_map):
    product_column = column_map["product_column"]
    quantity_sold_column = column_map["quantity_sold_column"]

    target_encoder = TargetEncoder(cols=product_column)
    data["product_encoded"] = target_encoder.fit_transform(data[product_column], data[quantity_sold_column])
    # st.write("Target Encoded Result ", uploaded_dataset.sort_values(units_sold_column, ascending=True).head(5))
    st.success("Product Column has been successfully encoded")
    return data
