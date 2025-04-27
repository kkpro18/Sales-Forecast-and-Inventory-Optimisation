import pytest
import time
import os
import subprocess

from pydantic.v1.utils import almost_equal_floats
import pandas as pd
from App.Models import data_model
from App.Controllers import data_preprocessing_controller


@pytest.fixture(scope='session', autouse=True)
def before_all():
    """
    This function runs before all tests
    """
    print("Starting FastAPI server...")
    os.chdir("../")
    subprocess.run(["bash", "start_fast_api.sh"])
    time.sleep(3)  # delay to setup fast api
    yield  # tests will run here
    print("Stopping FastAPI server...")
    subprocess.run(["pkill", "-f", "uvicorn"])

def preprocessing_pipeline(path="App/data/kaggle.com_datasets_gabrielramos87_an-online-shop-business/Sales Transaction v.4a.csv",
                           column_mapping={
                               "date_column": "Date",
                               "product_column": "ProductName",
                               "price_column": "Price",
                               "quantity_sold_column": "Quantity"
                           }):

    mock_data = data_model.read_file(path)

    data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)
    assert isinstance(data_as_json, list), "Data is not a List of Dictionaries (JSON Compatible Format)"

    transformed_data_response = data_preprocessing_controller.handle_data_transformation(data_as_json, column_mapping)

    transformed_data = transformed_data_response.json()["data"]
    assert transformed_data is not None, f"Data transformation returned {type(transformed_data.json())} instead of a list of dictionaries"

    data_handled_outliers = data_preprocessing_controller.handle_outliers(transformed_data, column_mapping)
    response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_handled_outliers.json(), column_mapping)
    daily_store_sales = pd.DataFrame(response.json()["daily_store_sales"])
    daily_product_sales = pd.DataFrame(response.json()["daily_product_sales"])

    assert daily_store_sales is not None, "Daily Store Sales Missing"
    assert daily_product_sales is not None, "Daily Product Sales Missing"

    response = data_preprocessing_controller.handle_train_test_split(
        daily_store_sales=response.json()["daily_store_sales"],
        daily_product_sales=response.json()["daily_product_sales"],
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

    # almost equal used as time-based splitting means the proportions are not exactly 80:20, so the threshold was also changed from strict default to +-0.05%
    assert almost_equal_floats(product_sales_train_proportion, 0.8,
                               delta=0.05), "Product Sales is not split in 80:20"
    assert almost_equal_floats(store_sales_train_proportion, 0.8, delta=0.05), "Store Sales is not split in 80:20"

    response = data_preprocessing_controller.handle_missing_values(
        train_daily_store_sales=response.json()['train_daily_store_sales'],
        test_daily_store_sales=response.json()['test_daily_store_sales'],
        train_daily_product_sales=response.json()['train_daily_product_sales'],
        test_daily_product_sales=response.json()['test_daily_product_sales'],
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

    train_daily_store_sales_with_exogenous, test_daily_store_sales_with_exogenous, train_product_sales_with_exogenous, test_product_sales_with_exogenous = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(
        "UK",
        train_daily_store_sales, test_daily_store_sales,
        train_daily_product_sales, test_daily_product_sales,
        column_mapping)

    assert set(train_daily_store_sales_with_exogenous.columns) != set(pd.DataFrame(train_daily_store_sales).columns)
    assert set(test_daily_store_sales_with_exogenous.columns) != set(pd.DataFrame(test_daily_store_sales).columns)

    assert set(train_product_sales_with_exogenous.columns) != set(pd.DataFrame(train_daily_product_sales).columns)
    assert set(test_product_sales_with_exogenous.columns) != set(pd.DataFrame(test_daily_product_sales).columns)

    train_daily_store_sales_with_exogenous_scaled, test_daily_store_sales_with_exogenous_scaled, train_product_sales_with_exogenous_scaled, test_product_sales_with_exogenous_scaled = data_preprocessing_controller.handle_exogenous_scaling(
        train_daily_store_sales_with_exogenous, test_daily_store_sales_with_exogenous,
        train_product_sales_with_exogenous, test_product_sales_with_exogenous,
        column_mapping)

    train_daily_store_sales_with_exogenous_scaled_lagged, test_daily_store_sales_with_exogenous_scaled_lagged, train_product_sales_with_exogenous_scaled_lagged, test_product_sales_with_exogenous_scaled_lagged = (
        data_preprocessing_controller.handle_lag_features(
            train_daily_store_sales_with_exogenous_scaled, test_daily_store_sales_with_exogenous_scaled,
            train_product_sales_with_exogenous_scaled,
            test_product_sales_with_exogenous_scaled, column_mapping))

    lag_data = [train_daily_store_sales_with_exogenous_scaled_lagged,
                test_daily_store_sales_with_exogenous_scaled_lagged,
                train_product_sales_with_exogenous_scaled_lagged,
                test_product_sales_with_exogenous_scaled_lagged]
    lag_columns = ["-1day", "-2day", "-3day"]
    for data in lag_data:
        for lag_column in lag_columns:
            assert lag_column in data.columns.tolist()


class TestForecasting:
    """
    ENSURE FAST_API is Running
    Forecasting Controller Tests
    """

    def test_handle_store_sales_data(self):
        preprocessing_pipeline()

    def test_handle_product_sales_data(self):
        self.fail()

    def test_handle_arima_sarima_training_and_predictions(self):
        self.fail()

    def test_handle_arimax_sarimax_training_and_predictions(self):
        self.fail()

    def test_handle_fb_prophet_with_and_without_exog_training_and_predictions(self):
        self.fail()
