import pytest
import time
import os
import subprocess

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


class TestForecasting:
    """
    ENSURE FAST_API is Running
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
