import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, List, Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, conint
from App.utils.data_preprocessing import format_dates, handle_missing_values, handle_outliers, encode_product_column
from App.utils.forecasting_sales import fit_arima_model, fit_sarima_model, predict
from datetime import datetime

"uvicorn App.utils.fast_api_app:app --port 8000 --reload"

app = FastAPI()
executor = ProcessPoolExecutor(max_workers=2) # do one less than the number of cores || no. jobs in parallel

class InputData(BaseModel):
    column_mapping: Optional[Dict[str, str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    X_train: Optional[List[Dict[str, Any]]] = None
    X_test: Optional[List[Dict[str, Any]]] = None
    y_train: Optional[Dict[int, Any]] = None
    y_test: Optional[Dict[int, Any]] = None
    seasonality: Optional[conint(gt=0)] = None
    test_forecast_steps: Optional[conint(gt=0)] = None
    model_path : Optional[str] = None


## Pre Processing
@app.post("/format_dates_api")
def format_dates_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return format_dates(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/handle_missing_values_api")
def handle_missing_values_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return handle_missing_values(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/handle_outliers_api")
def handle_outliers_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return handle_outliers(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/encode_product_column_api")
def encode_product_column_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return encode_product_column(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Fitting

# Methods to fit models
def fit_and_store_arima_model(received_data: InputData):
    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        y_train = pd.Series(received_data.y_train)

        arima_model = fit_arima_model(y_train)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        arima_model_path = f'models/arima_{date_timestamp}.pkl'
        joblib.dump(arima_model, arima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"arima_model_path": arima_model_path}

def fit_and_store_sarima_model(received_data: InputData):
    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        if received_data.seasonality is None:
            raise ValueError("seasonality is missing.")

        seasonality = received_data.seasonality
        y_train = pd.Series(received_data.y_train)

        sarima_model = fit_sarima_model(y_train, seasonality)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        sarima_model_path = f'models/sarima_{date_timestamp}.pkl'
        joblib.dump(sarima_model, sarima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"sarima_model_path": sarima_model_path}


# API Endpoint for fitting models in Parallel
@app.post("/fit_models_in_parallel_api")
async def fit_models_in_parallel_api(received_data: InputData): # https://blog.stackademic.com/fastapi-parallel-processing-1eaa67981ab9
    loop = asyncio.get_running_loop()

    fitting_arima = loop.run_in_executor(executor, fit_and_store_arima_model, received_data)
    fitting_sarima = loop.run_in_executor(executor, fit_and_store_sarima_model, received_data)

    arima_model, sarima_model = await asyncio.gather(fitting_arima, fitting_sarima)

    return {"arima" : arima_model, "sarima" : sarima_model}


# Model Predictions
@app.post("/predict_train_test_api")
def predict_train_test_api(received_data: InputData):
    try:
        test_forecast_steps = received_data.test_forecast_steps
        model_path = received_data.model_path

        y_train_prediction = predict(model_path=model_path, forecast_periods=None)
        y_test_prediction = predict(model_path=model_path, forecast_periods=test_forecast_steps)

        return {"y_train_prediction": y_train_prediction, "y_test_prediction": y_test_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, port=8000)