import numpy as np
import streamlit as st
import pandas as pd
from category_encoders import TargetEncoder
from scipy.stats import zscore
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


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


def split_training_testing_data(data, column_mapping):
    features = column_mapping.copy()
    features.pop("quantity_sold_column")
    features = features.values()

    target = column_mapping["quantity_sold_column"]

    # 70 : 30 split
    train_size = int(len(data) * 0.70)
    end_train_date = data.iloc[train_size][column_mapping["date_column"]]

    train = data[data["date"] < end_train_date]
    test = data[data["date"] >= end_train_date]

    st.success("Data has been split into training and test set 70:30 Ratio")

    return train, test

def handle_missing_values(train, test, column_mapping):
    product_column = column_mapping["product_column"]
    price_column = column_mapping["price_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]


    if train.isnull().values.any() or test.isnull().values.any():
        # rows_with_missing_values = data[data.isnull().any(axis=1)]
        st.write("Data contains Missing or Null values")

        if train[quantity_sold_column].isnull().values.any() or test[quantity_sold_column].isnull().values.any(): # if target column has missing values, no point imputing so drop
            train.dropna(subset=[quantity_sold_column], inplace=True)
            test.dropna(subset=[quantity_sold_column], inplace=True)

        product_imputer = SimpleImputer(strategy='most_frequent')
        price_imputer = SimpleImputer(strategy='mean')

        ct = ColumnTransformer(
            [
                ("product_imputer", product_imputer, [product_column]),
                ("price_imputer", price_imputer, [price_column])
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        )
        ct.set_output(transform="pandas")

        ct.fit(train)

        train = ct.transform(train)
        test = ct.transform(test)

        if train.isnull().values.any() or test.isnull().values.any():
            st.warning("Missing Values Still Exists :/")
            st.dataframe(train[train.isnull().any(axis=1)])
            st.dataframe(test[test.isnull().any(axis=1)])
        else:
            st.success("Missing Values Processed and Cleaned")
    else:
        st.success("Data contains no missing or null values")

    return train, test


def format_dates(train, test, column_mapping):
    date_column = column_mapping["date_column"]
    st.toast(f"{len(train[date_column] + test[date_column])} dates loaded")
    train[date_column] = pd.to_datetime(train[date_column], errors="coerce")
    test[date_column] = pd.to_datetime(test[date_column], errors="coerce")

    train[date_column] = train[date_column].dt.tz_localize(None)
    test[date_column] = test[date_column].dt.tz_localize(None)

    train[date_column] = train[date_column].ffill()
    test[date_column] = test[date_column].ffill()

    if train[date_column].isna().sum() + test[date_column].isna().sum() > 0:
        st.warning("Missing values in the date column after processing")

    st.success("Dates have been successfully formatted!")
    return train, test
