import numpy as np
import streamlit as st
import pandas as pd
from category_encoders import TargetEncoder
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def format_dates(data, column_mapping):
    date_column = column_mapping["date_column"]
    st.toast(f"{len(data[date_column])} dates loaded")
    data[column_mapping["date_column"]] = pd.to_datetime(data[column_mapping["date_column"]], errors="coerce")
    data[column_mapping["date_column"]] = data[column_mapping["date_column"]].dt.tz_localize(None)
    data[date_column] = data[date_column].ffill()
    if data[date_column].isna().sum() > 0:
        st.warning("Missing values in the date column after processing")
    # # Set the date column as the index
    # data.index = data[date_column]
    # data = data.drop(columns=[date_column], axis=1)
    # data = data.sort_index()

    st.success("Dates have been successfully formatted!")
    return data

def handle_outliers(data, column_mapping):
    outlier_indices = []
    quantity_sold_column = column_mapping["quantity_sold_column"]
    product_column = column_mapping["product_column"]
    # consider pricing as possibly typos

    if (data[quantity_sold_column] < 0).sum() > 0:
        st.session_state.data = data[data[quantity_sold_column] >= 0].copy()
        data = data[data[quantity_sold_column] >= 0].copy()

    for product, product_group in data.groupby(product_column):
        quartile_1 = product_group[quantity_sold_column].quantile(0.25)
        quartile_3 = product_group[quantity_sold_column].quantile(0.75)
        inter_quartile_range = quartile_3 - quartile_1
        threshold = 1.5
        iqr_outlier_indices = product_group[(product_group[quantity_sold_column] < quartile_1 - threshold * inter_quartile_range)
                                            |
                                            (product_group[quantity_sold_column] > quartile_3 + threshold * inter_quartile_range)].index
        z_scores = np.abs(zscore(product_group[quantity_sold_column], nan_policy="omit"))  # try Median Absolute Deviation (MAD)
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
        data[quantity_sold_column] = data.groupby(product_column)[quantity_sold_column].transform(lambda x: round(x.fillna(x.mean())))
        st.success(f"{outlier_values.shape[0]} outliers have been processed")
    else:
        data = data.drop(outlier_indices)  # indexing may be a categorical or dates, so this would be an index safe method
        st.success(f"{outlier_values.shape[0]} outliers have been removed")

    return data

def handle_missing_values(data, column_mapping):
    quantity_sold_column = column_mapping["quantity_sold_column"]
    product_column = column_mapping["product_column"]
    price_column = column_mapping["price_column"]

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
            st.dataframe(data[data.isnull().any(axis=1)])
        else:
            st.success("Missing Values Processed and Cleaned")
    else:
        st.success("Data contains no missing or null values")
    return data

def encode_product_column(data, column_mapping):
    product_column = column_mapping["product_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]

    target_encoder = TargetEncoder(cols=product_column)
    data["product_encoded"] = target_encoder.fit_transform(data[product_column], data[quantity_sold_column])
    # st.write("Target Encoded Result ", uploaded_dataset.sort_values(units_sold_column, ascending=True).head(5))
    st.success("Product Column has been successfully encoded")
    return data
