import os

import pandas as pd

from App.Models.data_preprocessing_model import convert_to_dict, format_dates, concatenate_exogenous_data, scale_exogenous_data, add_lag_features
import streamlit as st
from App.utils.session_manager import SessionManager


def handle_dictionary_conversion(data):
    """
    returns the data as a dictionary (JSON Format).
    """
    try:
        data_as_dictionary = convert_to_dict(data)
    except Exception as e:
        st.error(e)
    else:
        return data_as_dictionary

def handle_data_transformation(data_as_dictionary, column_mapping):
    """
    Handles the data transformation for the uploaded data and returns the transformed data.
    """
    try:
        json_response = SessionManager.fast_api("transform_data_api",
                                                data=data_as_dictionary,
                                                column_mapping=column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return json_response

def handle_outliers(data, column_mapping):
    """
    Handles the outliers for the uploaded data and returns the data with outliers handled.
    """
    try:
        json_response = SessionManager.fast_api("handle_outliers_api", data=data, column_mapping=column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return json_response

def handle_dates_and_split_product_and_overall_sales(data, column_mapping):
    """
    Handles the dates and splits the data into product sales and overall store sales.
    """
    try:
        json_response = SessionManager.fast_api("fix_dates_and_split_into_product_sales_and_daily_sales_api",
                                                data=data,
                                                column_mapping=column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return json_response

def handle_train_test_split(daily_store_sales, daily_product_sales, column_mapping):
    """
    Handles the train test split (80:20) for the uploaded data and returns the train and test data.
    """
    try:
        json_response = SessionManager.fast_api("train_test_split_api",
                                                daily_store_sales=daily_store_sales,
                                                daily_product_sales=daily_product_sales,
                                                column_mapping=column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return json_response

def handle_missing_values(train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, column_mapping):
    """
    Handles the missing values for the uploaded data and returns the data with missing values handled.
    """
    try:
        json_response = SessionManager.fast_api("handle_missing_values_api",
                                                train_daily_store_sales=train_daily_store_sales,
                                                test_daily_store_sales=test_daily_store_sales,
                                                train_daily_product_sales=train_daily_product_sales,
                                                test_daily_product_sales=test_daily_product_sales,
                                                column_mapping=column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return json_response

def handle_date_formatting(train, test, column_mapping):
    """
    Formats the dates for the uploaded data and returns the data with dates formatted.
    """
    try:

        data = format_dates(pd.DataFrame(train), pd.DataFrame(test), column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return data

def handle_inclusion_of_exogenous_variables(selected_region, train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, column_mapping):
    """
    Handles the inclusion of exogenous variables for the uploaded data and returns the data with exogenous variables included.
    """
    try:
        data = concatenate_exogenous_data(selected_region, train_daily_store_sales, test_daily_store_sales,
                                          train_daily_product_sales, test_daily_product_sales, column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return data

def handle_exogenous_scaling(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog, column_mapping):
    """
    Handles the scaling of exogenous variables for the uploaded data and returns the data with exogenous variables scaled.
    """
    try:
        data = scale_exogenous_data(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog,
                                    train_daily_product_sales_with_exog, test_daily_product_sales_with_exog,
                                    column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return data

def handle_lag_features(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog, column_mapping):
    """
    Handles the lag features for the uploaded data and returns the data with lag features.
    """
    try:
        data = add_lag_features(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog,
                                train_daily_product_sales_with_exog, test_daily_product_sales_with_exog,
                                column_mapping)
    except Exception as e:
        st.error(e)
        return None
    else:
        return data



