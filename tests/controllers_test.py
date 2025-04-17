import numpy as np
import pytest
import pandas as pd
from pydantic.v1.utils import almost_equal_floats
import os
from App.Controllers import data_preprocessing_controller
import numpy


class TestControllers:
    """
    Pre-Processing Controller Tests
    ENSURE FAST_API is Running
    """

    def test_handle_dictionary_conversion(self):
        """
        Test the dictionary conversion of data - Preparing for JSON
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02"],
            "product": ["A", "B"],
            "price": [1.12, 2.34],
            "quantity_sold": [10, 20]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        # Check if the data is converted to JSON format
        assert isinstance(data_as_json, list), "Data is not a List of Dictionaries (JSON Compatible Format)"

    def test_handle_data_transformation(self):
        """
        Test the data transformation of data - Using FastAPI Pre-Processing Route
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", "2025-01-10"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "price": [1.12, 2.24, 3.34, 2.35, 1.47, 1.21, 2.20, 5.42, 5.21, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, 2300, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }

        transformed_data_response = data_preprocessing_controller.handle_data_transformation(data_as_json,
                                                                                             column_mapping).json()
        transformed_data = transformed_data_response["data"]
        is_log_transformed = transformed_data_response["is_log_transformed"]

        # Check if the data is transformed correctly
        assert transformed_data is not None, f"Data transformation returned {type(transformed_data)}"
        assert is_log_transformed == True, f"Data transformation failed, transformation status: {is_log_transformed} "

    def test_handle_outliers(self):
        """
        Tests if Outliers are Handled/Removed Correctly - FastAPI Pre-Processing Route
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", "2025-01-10"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "price": [1.12, 2.24, 3.34, 2.35, 1.47, 1.21, 2.20, 5.42, 5.21, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, 20, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }

        data_with_outliers = data_as_json
        data_handled_outliers = data_preprocessing_controller.handle_outliers(data_with_outliers, column_mapping)

        assert len(data_handled_outliers.json()) < len(data_with_outliers), "The 1000 Sales Outlier Remains in the Data"

        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", "2025-01-10"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "price": [1.12, 2.24, 3.34, 2.35, 1.47, 1.21, 2.20, 5.42, 5.21, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, 2000, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }

        data_with_outliers = data_as_json
        data_handled_outliers = data_preprocessing_controller.handle_outliers(data_with_outliers, column_mapping)
        assert len(data_handled_outliers.json()) == len(
            data_with_outliers), "Since, there is 20% outliers, they should not be removed and rather handled accordingly, however this is not the case"

    def test_handle_dates_and_split_product_and_overall_sales(self):
        """
        Tests if data is split into product and overall sales and missing gaps are filled
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", "2025-01-12"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "price": [1.12, 2.24, 3.34, 2.35, 1.47, 1.21, 2.20, 5.42, 5.21, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, 20, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)
        daily_store_sales = pd.DataFrame(response.json()["daily_store_sales"])
        daily_product_sales = pd.DataFrame(response.json()["daily_product_sales"])

        assert daily_store_sales is not None, "Daily Store Sales Missing"
        assert daily_product_sales is not None, "Daily Product Sales Missing"
        assert len(daily_store_sales[column_mapping["date_column"]].unique()) >= len(
            mock_data[column_mapping["date_column"]].unique())
        assert len(daily_store_sales[column_mapping["date_column"]].unique()) > len(
            mock_data[column_mapping["date_column"]].unique())

    def test_handle_train_test_split(self):
        """
        Tests if data is split in the correct ratio for train test data.
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", "2025-01-12"],
            "product": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "price": [1.12, 2.24, 3.34, 2.35, 1.47, 1.21, 2.20, 5.42, 5.21, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, 20, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)
        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        assert train_daily_store_sales is not None, "Train Daily Store Sales Missing"
        assert test_daily_store_sales is not None, "Test Daily Product Sales Missing"

        assert train_daily_product_sales is not None, "Train Daily Product Sales Missing"
        assert test_daily_product_sales is not None, "Test Daily Product Sales Missing"

        product_sales_train_proportion = len(train_daily_product_sales) / (len(train_daily_product_sales) + len(
            test_daily_product_sales))
        store_sales_train_proportion = len(train_daily_store_sales) / (len(train_daily_store_sales) + len(
            test_daily_store_sales))

        # almost equal used as time-based splitting means the proportions are not exactly 80:20, hence threshold was also changed from strict default to +-0.05%
        assert almost_equal_floats(product_sales_train_proportion, 0.8,
                                   delta=0.05), "Product Sales is not split in 80:20"
        assert almost_equal_floats(store_sales_train_proportion, 0.8, delta=0.05), "Store Sales is not split in 80:20"

    def test_handle_missing_values(self):
        """
        Checks if Missing Values are Handled
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", None],
            "product": ["A", "B", "C", None, "B", "C", "A", "B", None, "A"],
            "price": [None, 2.24, 3.34, 2.35, None, 1.21, 2.20, 5.42, None, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, None, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)

        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        response = data_preprocessing_controller.handle_missing_values(
            train_daily_store_sales=train_daily_store_sales,
            test_daily_store_sales=test_daily_store_sales,
            train_daily_product_sales=train_daily_product_sales,
            test_daily_product_sales=test_daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales_handled_missing_values = response.json()['train_daily_store_sales']
        test_daily_store_sales_handled_missing_values = response.json()['test_daily_store_sales']

        train_daily_product_sales_handled_missing_values = response.json()['train_daily_product_sales']
        test_daily_product_sales_handled_missing_values = response.json()['test_daily_product_sales']

        assert not pd.DataFrame(
            train_daily_store_sales_handled_missing_values).isnull().values.any(), "Missing Values Exist"
        assert not pd.DataFrame(
            test_daily_store_sales_handled_missing_values).isnull().values.any(), "Missing Values Exist"

        assert not pd.DataFrame(
            train_daily_product_sales_handled_missing_values).isnull().values.any(), "Missing Values Exist"
        assert not pd.DataFrame(
            test_daily_product_sales_handled_missing_values).isnull().values.any(), "Missing Values Exist"

    def test_handle_date_formatting(self):
        """
        Checks if Date Column is converted to the right format
        """
        # Mock data
        mock_data = pd.DataFrame(data=
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06", "2025-01-07",
                     "2025-01-08", "2025-01-09", None],
            "product": ["A", "B", "C", None, "B", "C", "A", "B", None, "A"],
            "price": [None, 2.24, 3.34, 2.35, None, 1.21, 2.20, 5.42, None, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, None, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)

        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        response = data_preprocessing_controller.handle_missing_values(
            train_daily_store_sales=train_daily_store_sales,
            test_daily_store_sales=test_daily_store_sales,
            train_daily_product_sales=train_daily_product_sales,
            test_daily_product_sales=test_daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales_handled_missing_values = response.json()['train_daily_store_sales']
        test_daily_store_sales_handled_missing_values = response.json()['test_daily_store_sales']

        train_daily_product_sales_handled_missing_values = response.json()['train_daily_product_sales']
        test_daily_product_sales_handled_missing_values = response.json()['test_daily_product_sales']

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_store_sales_handled_missing_values, test_daily_store_sales_handled_missing_values,
            column_mapping)
        assert isinstance(train_daily_store_sales[column_mapping["date_column"]][0], pd.Timestamp)
        assert isinstance(test_daily_store_sales[column_mapping["date_column"]][0], pd.Timestamp)

        train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_product_sales_handled_missing_values, test_daily_product_sales_handled_missing_values,
            column_mapping)
        assert isinstance(train_daily_product_sales[column_mapping["date_column"]][0], pd.Timestamp)
        assert isinstance(test_daily_product_sales[column_mapping["date_column"]][0], pd.Timestamp)

    def test_handle_inclusion_of_exogenous_variables(self):
        """
        Tests if Exogenous Variables are Included
        """

        os.chdir("../") # needs access to App Dir

        # Mock data - dates are older due to exog data age
        mock_data = pd.DataFrame(data=
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07",
                     "2024-01-08", "2024-01-09", None],
            "product": ["A", "B", "C", None, "B", "C", "A", "B", None, "A"],
            "price": [None, 2.24, 3.34, 2.35, None, 1.21, 2.20, 5.42, None, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, None, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)

        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        response = data_preprocessing_controller.handle_missing_values(
            train_daily_store_sales=train_daily_store_sales,
            test_daily_store_sales=test_daily_store_sales,
            train_daily_product_sales=train_daily_product_sales,
            test_daily_product_sales=test_daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales_handled_missing_values = response.json()['train_daily_store_sales']
        test_daily_store_sales_handled_missing_values = response.json()['test_daily_store_sales']

        train_daily_product_sales_handled_missing_values = response.json()['train_daily_product_sales']
        test_daily_product_sales_handled_missing_values = response.json()['test_daily_product_sales']

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_store_sales_handled_missing_values, test_daily_store_sales_handled_missing_values,
            column_mapping)
        train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_product_sales_handled_missing_values, test_daily_product_sales_handled_missing_values,
            column_mapping)

        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog, test_product_sales_with_exog = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(
            "UK",
            train_daily_store_sales, test_daily_store_sales,
            train_daily_product_sales,  test_daily_product_sales,
            column_mapping)

        assert set(train_daily_store_sales_with_exog.columns) != set(pd.DataFrame(train_daily_store_sales).columns)
        assert set(test_daily_store_sales_with_exog.columns) != set(pd.DataFrame(test_daily_store_sales).columns)

        assert set(train_product_sales_with_exog.columns) != set(pd.DataFrame(train_daily_product_sales).columns)
        assert set(test_product_sales_with_exog.columns) != set(pd.DataFrame(test_daily_product_sales).columns)
    def test_handle_exogenous_scaling(self):
        os.chdir("../")  # needs access to App Dir

        # Mock data - dates are older due to exog data age
        mock_data = pd.DataFrame(data=
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07",
                     "2024-01-08", "2024-01-09", None],
            "product": ["A", "B", "C", None, "B", "C", "A", "B", None, "A"],
            "price": [None, 2.24, 3.34, 2.35, None, 1.21, 2.20, 5.42, None, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, None, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)

        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        response = data_preprocessing_controller.handle_missing_values(
            train_daily_store_sales=train_daily_store_sales,
            test_daily_store_sales=test_daily_store_sales,
            train_daily_product_sales=train_daily_product_sales,
            test_daily_product_sales=test_daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales_handled_missing_values = response.json()['train_daily_store_sales']
        test_daily_store_sales_handled_missing_values = response.json()['test_daily_store_sales']

        train_daily_product_sales_handled_missing_values = response.json()['train_daily_product_sales']
        test_daily_product_sales_handled_missing_values = response.json()['test_daily_product_sales']

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_store_sales_handled_missing_values, test_daily_store_sales_handled_missing_values,
            column_mapping)
        train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_product_sales_handled_missing_values, test_daily_product_sales_handled_missing_values,
            column_mapping)

        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog, test_product_sales_with_exog = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(
            "UK",
            train_daily_store_sales, test_daily_store_sales,
            train_daily_product_sales, test_daily_product_sales,
            column_mapping)

        train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_product_sales_with_exog_scaled, test_product_sales_with_exog_scaled = data_preprocessing_controller.handle_exogenous_scaling(
            train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog,
            test_product_sales_with_exog, column_mapping)

        scaled_data = [train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_product_sales_with_exog_scaled,
                       test_product_sales_with_exog_scaled]

        exog_columns = list(train_daily_store_sales_with_exog_scaled.columns.difference(train_daily_store_sales.columns))

        for data in scaled_data:
            for exog_column in exog_columns:
                assert almost_equal_floats(np.mean(data[exog_column]), 0),f"Column: {exog_column} is not scaled correctly, the mean is not removed/0"



    def test_handle_lag_features(self):
        os.chdir("../")  # needs access to App Dir

        # Mock data - dates are older due to exog data age
        mock_data = pd.DataFrame(data=
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-07",
                     "2024-01-08", "2024-01-09", None],
            "product": ["A", "B", "C", None, "B", "C", "A", "B", None, "A"],
            "price": [None, 2.24, 3.34, 2.35, None, 1.21, 2.20, 5.42, None, 7.83],
            "quantity_sold": [1, 2, 3, 5, 7, 10, 12, 12, None, 1000]
        })

        data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

        column_mapping = {
            "date_column": "date",
            "product_column": "product",
            "price_column": "price",
            "quantity_sold_column": "quantity_sold"
        }
        response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_as_json,
                                                                                                  column_mapping)

        daily_store_sales = response.json()["daily_store_sales"]
        daily_product_sales = response.json()["daily_product_sales"]

        response = data_preprocessing_controller.handle_train_test_split(
            daily_store_sales=daily_store_sales,
            daily_product_sales=daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales = response.json()['train_daily_store_sales']
        test_daily_store_sales = response.json()['test_daily_store_sales']

        train_daily_product_sales = response.json()['train_daily_product_sales']
        test_daily_product_sales = response.json()['test_daily_product_sales']

        response = data_preprocessing_controller.handle_missing_values(
            train_daily_store_sales=train_daily_store_sales,
            test_daily_store_sales=test_daily_store_sales,
            train_daily_product_sales=train_daily_product_sales,
            test_daily_product_sales=test_daily_product_sales,
            column_mapping=column_mapping)

        train_daily_store_sales_handled_missing_values = response.json()['train_daily_store_sales']
        test_daily_store_sales_handled_missing_values = response.json()['test_daily_store_sales']

        train_daily_product_sales_handled_missing_values = response.json()['train_daily_product_sales']
        test_daily_product_sales_handled_missing_values = response.json()['test_daily_product_sales']

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_store_sales_handled_missing_values, test_daily_store_sales_handled_missing_values,
            column_mapping)
        train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(
            train_daily_product_sales_handled_missing_values, test_daily_product_sales_handled_missing_values,
            column_mapping)

        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog, test_product_sales_with_exog = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(
                                                                                        "UK",
                                                                                        train_daily_store_sales, test_daily_store_sales,
                                                                                        train_daily_product_sales, test_daily_product_sales,
                                                                                        column_mapping)

        train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_product_sales_with_exog_scaled, test_product_sales_with_exog_scaled = data_preprocessing_controller.handle_exogenous_scaling(
                                                                                        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_product_sales_with_exog,
                                                                                        test_product_sales_with_exog, column_mapping)

        scaled_data = [train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled,
                                                                                       train_product_sales_with_exog_scaled,
                                                                                       test_product_sales_with_exog_scaled]

        train_daily_store_sales_with_exog_scaled_lagged, test_daily_store_sales_with_exog_scaled_lagged, train_product_sales_with_exog_scaled_lagged, test_product_sales_with_exog_scaled_lagged = (
            data_preprocessing_controller.handle_lag_features(
                                                                                            train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_product_sales_with_exog_scaled,
                                                                                            test_product_sales_with_exog_scaled, column_mapping))

        lag_data = [train_daily_store_sales_with_exog_scaled_lagged,
                    test_daily_store_sales_with_exog_scaled_lagged,
                   train_product_sales_with_exog_scaled_lagged,
                   test_product_sales_with_exog_scaled_lagged]
        lag_columns = ["-1day", "-2day", "-3day"]
        for data in lag_data:
            for lag_column in lag_columns:
                assert lag_column in data.columns.tolist()

    """
    Forecasting Controller Tests
    """

    def test_handle_store_sales_data(self):
        self.fail()

    def test_handle_product_sales_data(self):
        self.fail()

    def test_handle_seasonality_input(self):
        self.fail()

    def test_handle_arima_sarima_training_and_predictions(self):
        self.fail()

    def test_handle_arimax_sarimax_training_and_predictions(self):
        self.fail()

    def test_handle_fb_prophet_with_and_without_exog_training_and_predictions(self):
        self.fail()
