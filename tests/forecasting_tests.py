import pytest
import time
import os
import subprocess
from pydantic.v1.utils import almost_equal_floats
import pandas as pd
from App.Models import data_model
from App.Controllers import data_preprocessing_controller, forecasting_controller


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

def preprocessing_pipeline(path=None,
                           column_mapping=None):

    if column_mapping is None:
        column_mapping = {
            "date_column": "Date",
            "product_column": "ProductName",
            "price_column": "Price",
            "quantity_sold_column": "Quantity"
        }

    if path is None:
        path = "App/data/kaggle.com_datasets_gabrielramos87_an-online-shop-business/Sales Transaction v.4a.csv"

    mock_data = data_model.read_file(path)

    data_as_json = data_preprocessing_controller.handle_dictionary_conversion(mock_data)

    transformed_data_response = data_preprocessing_controller.handle_data_transformation(data_as_json, column_mapping)

    transformed_data = transformed_data_response.json()["data"]

    data_handled_outliers = data_preprocessing_controller.handle_outliers(transformed_data, column_mapping)
    response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data_handled_outliers.json(), column_mapping)

    response = data_preprocessing_controller.handle_train_test_split(
        daily_store_sales=response.json()["daily_store_sales"],
        daily_product_sales=response.json()["daily_product_sales"],
        column_mapping=column_mapping)

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
    train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(
        train_daily_product_sales_handled_missing_values, test_daily_product_sales_handled_missing_values,
        column_mapping)
    train_daily_store_sales_with_exogenous, test_daily_store_sales_with_exogenous, train_product_sales_with_exogenous, test_product_sales_with_exogenous = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(
        "UK",
        train_daily_store_sales, test_daily_store_sales,
        train_daily_product_sales, test_daily_product_sales,
        column_mapping)

    train_daily_store_sales_with_exogenous_scaled, test_daily_store_sales_with_exogenous_scaled, train_product_sales_with_exogenous_scaled, test_product_sales_with_exogenous_scaled = data_preprocessing_controller.handle_exogenous_scaling(
        train_daily_store_sales_with_exogenous, test_daily_store_sales_with_exogenous,
        train_product_sales_with_exogenous, test_product_sales_with_exogenous,
        column_mapping)

    train_daily_store_sales_with_exogenous_scaled_lagged, test_daily_store_sales_with_exogenous_scaled_lagged, train_product_sales_with_exogenous_scaled_lagged, test_product_sales_with_exogenous_scaled_lagged = (
        data_preprocessing_controller.handle_lag_features(
            train_daily_store_sales_with_exogenous_scaled, test_daily_store_sales_with_exogenous_scaled,
            train_product_sales_with_exogenous_scaled,
            test_product_sales_with_exogenous_scaled, column_mapping))

    return train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, train_daily_store_sales_with_exogenous_scaled_lagged, test_daily_store_sales_with_exogenous_scaled_lagged, train_product_sales_with_exogenous_scaled_lagged, test_product_sales_with_exogenous_scaled_lagged

class TestForecasting:
    """
    ENSURE FAST_API is Running
    Forecasting Controller Tests
    """
    def test_handle_arima_sarima_training_and_predictions(self):
        train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, train_daily_store_sales_with_exogenous_scaled_lagged, test_daily_store_sales_with_exogenous_scaled_lagged, train_product_sales_with_exogenous_scaled_lagged, test_product_sales_with_exogenous_scaled_lagged = preprocessing_pipeline()
        column_mapping = {
            "date_column": "Date",
            "product_column": "ProductName",
            "price_column": "Price",
            "quantity_sold_column": "Quantity"
        }
        asyncio.run(
            forecasting_controller.handle_arima_sarima_training_and_predictions(
                train_daily_store_sales,
                test_daily_store_sales,
                column_mapping,
            )
        )


    def test_handle_arimax_sarimax_training_and_predictions(self):
        self.fail()

    def test_handle_fb_prophet_with_and_without_exog_training_and_predictions(self):
        self.fail()
