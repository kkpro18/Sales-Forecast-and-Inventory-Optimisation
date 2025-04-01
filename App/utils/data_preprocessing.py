import numpy as np
import streamlit as st
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from App.utils.session_manager import SessionManager
from App.utils.data_handling import read_file


def convert_to_dict(data):
    # converts to dicts while fixes any issues that causes json errors e.g NaN, Infinity, -Infinity, trailing 0s.
    data.replace(pd.NaT, None, inplace=True)
    data.replace([np.nan], None, inplace=True)
    data.replace([np.inf, -np.inf], None, inplace=True)
    # needs to handle symbols maybe

    return data.to_dict(orient='records')


def transform_data(data, column_mapping):
    sales_column = column_mapping["quantity_sold_column"]
    if data[sales_column].skew() > 1:
        data[sales_column] = np.log1p(data[sales_column])
        st.success("Sales Column has been transformed using log transformation as it is skewed")
    else:
        st.success("Sales Column is not skewed, no transformation applied")
    return data


def handle_outliers(data, column_mapping):
    outlier_indices = []
    quantity_sold_column = column_mapping["quantity_sold_column"]
    product_column = column_mapping["product_column"]
    # consider pricing as possibly typos

    if (data[quantity_sold_column] < 0).sum() > 0:
        SessionManager.set_state("data", data[data[quantity_sold_column] >= 0].reset_index(drop=True).copy())
        data = SessionManager.get_state("data")

    for product, product_group in data.groupby(product_column):
        quartile_1 = product_group[quantity_sold_column].quantile(0.25)
        quartile_3 = product_group[quantity_sold_column].quantile(0.75)
        inter_quartile_range = quartile_3 - quartile_1
        threshold = 1.5
        iqr_outlier_indices = product_group[(product_group[quantity_sold_column] < quartile_1 - threshold * inter_quartile_range)
                                            |
                                            (product_group[quantity_sold_column] > quartile_3 + threshold * inter_quartile_range)].index
        # median absolute deviation to handle outliers

        median_sales = np.median(product_group[quantity_sold_column])
        mad = np.median(np.abs(product_group[quantity_sold_column] - median_sales))
        if mad == 0:  # prevent division by zero
            mad_outlier_indices = np.zeros(len(product_group[quantity_sold_column]), dtype=bool)
        else:
            modified_z_score = 0.6745 * (product_group[quantity_sold_column] - median_sales) / mad
            mad_outlier_indices = np.where(np.abs(modified_z_score) > 3.5)[0]

        outlier_indices.extend(iqr_outlier_indices)
        outlier_indices.extend(mad_outlier_indices)

    outlier_indices = list(set(outlier_indices))
    outlier_values = data[quantity_sold_column].loc[outlier_indices]

    st.write(f"No. Outliers in Sales Column {outlier_values.shape[0]}")
    st.write(f"Total No. Values in Sales Column {data[quantity_sold_column].shape[0]}")
    outlier_proportion = round(outlier_values.shape[0] / data[quantity_sold_column].shape[0] * 100,2)
    st.write(f"Proportion of Outliers to Total Data {outlier_proportion}%")

    if outlier_proportion >= 20:
        # since outliers are more than 15% of the data, we will keep mean of the products for each product
        data.loc[outlier_indices, quantity_sold_column] = None # outliers are removed prior to calculating mean to prevent bias
        data[quantity_sold_column] = data.groupby(product_column)[quantity_sold_column].transform('mean').round()
        st.success(f"{outlier_values.shape[0]} outliers have been processed")
    else:
        data = data.drop(outlier_indices)
        st.success(f"{outlier_values.shape[0]} outliers have been removed")

    return data


def split_training_testing_data(data, column_mapping):
    # 70 : 30 split
    # convert date column to datetime
    # data[column_mapping["date_column"]] = pd.to_datetime(data[column_mapping["date_column"]], errors="coerce")
    date_column = column_mapping["date_column"]
    product_column = column_mapping["product_column"]
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")
    data[date_column] = data[date_column].dt.tz_localize(None)
    data[date_column] = data[date_column].ffill()

    data.sort_values(by=date_column, ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    train_size = int(len(data) * 0.90)
    end_train_date = data.iloc[train_size][date_column]
    st.write(f"Training End Date : {end_train_date}")

    train = data[data[date_column] < end_train_date]
    test = data[data[date_column] > end_train_date]
    test = test[test[product_column].isin(train[product_column].unique())]

    return train, test


def encode_product_column(train, test, column_mapping):
    product_column = column_mapping["product_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]

    target_encoder = TargetEncoder(cols=product_column)

    target_encoder.fit(train[product_column], train[quantity_sold_column])

    train["product_encoded"] = target_encoder.transform(train[product_column])
    test["product_encoded"] = target_encoder.transform(test[product_column])

    train_product_encoder_map = dict(zip(train[product_column], train['product_encoded']))
    test_product_encoder_map = dict(zip(test[product_column], test['product_encoded']))

    SessionManager.set_state("train_product_encoder_map", train_product_encoder_map)
    SessionManager.set_state("test_product_encoder_map", test_product_encoder_map)

    # st.write("Target Encoded Result ", uploaded_dataset.sort_values(units_sold_column, ascending=True).head(5))
    st.success("Product Column has been successfully encoded")
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

    return train, test

# scale exog X only, robust scaler to prevent outliers, MINMAX for fixed range, standard scaler for normal distribution - prioritise standard scaler mostly

def fill_missing_date_range_product_group(group, column_mapping):
    product_date_range = pd.date_range(start=group[column_mapping["date_column"]].min(),
                                       end=group[column_mapping["date_column"]].max(), freq='D')
    product_daily_dates_df = pd.DataFrame(product_date_range, columns=[column_mapping["date_column"]])
    group = pd.merge(product_daily_dates_df, group, on=column_mapping["date_column"], how='left')
    group[column_mapping["quantity_sold_column"]] = group[column_mapping["quantity_sold_column"]].fillna(0)
    group[column_mapping["price_column"]] = group[column_mapping["price_column"]].ffill()
    group[column_mapping["product_column"]] = group[column_mapping["product_column"]].ffill()

    return group

def format_dates(train, test, column_mapping):
    date_column = column_mapping["date_column"]
    train[date_column] = pd.to_datetime(train[date_column], errors="coerce")
    test[date_column] = pd.to_datetime(test[date_column], errors="coerce")

    train[date_column] = train[date_column].dt.tz_localize(None)
    test[date_column] = test[date_column].dt.tz_localize(None)

    # train[date_column] = train[date_column].ffill()
    # test[date_column] = test[date_column].ffill()
    train.dropna(subset=[date_column], inplace=True)
    test.dropna(subset=[date_column], inplace=True)

    if train[date_column].isna().sum() + test[date_column].isna().sum() > 0:
        st.warning("Missing values in the date column after processing")

    # train daily sales
    # groups by each date and sums them
    train_daily_store_sales = train.groupby(column_mapping["date_column"], as_index=False).agg(
        {column_mapping["quantity_sold_column"]: 'sum'})
    train_date_range = pd.date_range(start=train_daily_store_sales[column_mapping["date_column"]].min(), end=train_daily_store_sales[column_mapping["date_column"]].max(), freq='D')
    train_daily_dates_df = pd.DataFrame(train_date_range, columns=[column_mapping["date_column"]])
    train_daily_store_sales = pd.merge(train_daily_dates_df, train_daily_store_sales, on=column_mapping["date_column"],  how='left')
    # fills in missing columns from new addded dates
    train_daily_store_sales[column_mapping["quantity_sold_column"]] = train_daily_store_sales[column_mapping["quantity_sold_column"]].fillna(0)

    train_daily_store_sales.reset_index(drop=True, inplace=True)

    # test daily sales
    # groups by each date and sums them
    test_daily_store_sales = test.groupby(column_mapping["date_column"], as_index=False).agg(
        {column_mapping["quantity_sold_column"]: 'sum'})
    test_date_range = pd.date_range(start=test_daily_store_sales[column_mapping["date_column"]].min(), end=test_daily_store_sales[column_mapping["date_column"]].max(), freq='D')
    test_daily_dates_df = pd.DataFrame(test_date_range, columns=[column_mapping["date_column"]])
    test_daily_store_sales = pd.merge(test_daily_dates_df, test_daily_store_sales, on=column_mapping["date_column"], how='left')
    # fills in missing columns from new addded dates
    test_daily_store_sales[column_mapping["quantity_sold_column"]] = test_daily_store_sales[column_mapping["quantity_sold_column"]].fillna(0)

    test_daily_store_sales.reset_index(drop=True, inplace=True)


    # train product specific sales
    train_product_sales = train.groupby([column_mapping["product_column"], column_mapping["date_column"]],
                                        as_index=False).agg(
        {
            column_mapping["price_column"]: 'mean',
            column_mapping["quantity_sold_column"]: 'sum'
        })
    train_product_sales = train_product_sales.groupby(column_mapping["product_column"]).apply(lambda group: fill_missing_date_range_product_group(group, column_mapping))
    train_product_sales.reset_index(drop=True, inplace=True)

    # test product specific sales
    test_product_sales = test.groupby([column_mapping["product_column"], column_mapping["date_column"]],
                                      as_index=False).agg(
        {
            column_mapping["price_column"]: 'mean',
            column_mapping["quantity_sold_column"]: 'sum'
        })
    test_product_sales = test_product_sales.groupby(column_mapping["product_column"]).apply(lambda group: fill_missing_date_range_product_group(group, column_mapping))
    test_product_sales.reset_index(drop=True, inplace=True)


    return train_daily_store_sales, test_daily_store_sales, train_product_sales, test_product_sales


def concatenate_exogenous_data(selected_region, train_daily_store_sales, test_daily_store_sales, train_product_sales, test_product_sales, column_mapping):
    if selected_region == "UK":
        exogenous_data = read_file("App/AppMaintenance/raw_macro_economical_data/UK/Processed/uk_macro_economical_data.csv")
        exog_date_column = "Date"
        exogenous_data[exog_date_column] = pd.to_datetime(exogenous_data[exog_date_column], format="%Y-%m-%d",
                                                          errors="coerce")
        exogenous_data[exog_date_column] = exogenous_data[exog_date_column].dt.tz_localize(None)
        exogenous_data = exogenous_data.rename(columns={exog_date_column: column_mapping["date_column"]})

        train_daily_store_sales_with_exog = pd.merge(train_daily_store_sales, exogenous_data,
                                                     on=column_mapping["date_column"], how='left')
        test_daily_store_sales_with_exog = pd.merge(test_daily_store_sales, exogenous_data,
                                                    on=column_mapping["date_column"], how='left')
        train_product_sales_with_exog = pd.merge(train_product_sales, exogenous_data,
                                                 on=column_mapping["date_column"], how='left')
        test_product_sales_with_exog = pd.merge(test_product_sales, exogenous_data,
                                                on=column_mapping["date_column"], how='left')

        return train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog, test_product_sales_with_exog


    elif selected_region == "USA":
        exogenous_data = read_file("App/AppMaintenance/raw_macro_economical_data/USA/Processed/usa_macro_economical_data.csv")
        exog_date_column = "Date"
        exogenous_data[exog_date_column] = pd.to_datetime(exogenous_data[exog_date_column], format="%Y-%m-%d", errors="coerce")
        exogenous_data[exog_date_column] = exogenous_data[exog_date_column].dt.tz_localize(None)
        exogenous_data = exogenous_data.rename(columns={exog_date_column: column_mapping["date_column"]})

        train_daily_store_sales_with_exog = pd.merge(train_daily_store_sales, exogenous_data,
                                                     on=column_mapping["date_column"], how='left')
        test_daily_store_sales_with_exog = pd.merge(test_daily_store_sales, exogenous_data,
                                                    on=column_mapping["date_column"], how='left')
        train_product_sales_with_exog = pd.merge(train_product_sales, exogenous_data,
                                                 on=column_mapping["date_column"], how='left')
        test_product_sales_with_exog = pd.merge(test_product_sales, exogenous_data,
                                                on=column_mapping["date_column"], how='left')


        return train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog, test_product_sales_with_exog








