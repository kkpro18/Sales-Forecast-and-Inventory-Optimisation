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
executor = ProcessPoolExecutor(max_workers=2) # do one less than the number of cores

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
@app.post("/format_dates_call")
def format_dates_call(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return format_dates(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/handle_missing_values_call")
def handle_missing_values_call(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return handle_missing_values(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/handle_outliers_call")
def handle_outliers_call(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return handle_outliers(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/encode_product_column_call")
def encode_product_column_call(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return encode_product_column(data, column_mapping).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Model Fitting
# @app.post("/fit_and_store_arima_model_call")
def fit_and_store_arima_model_call(received_data: InputData):
    try:
        if not received_data.y_train:
            raise ValueError("y_train is missing.")
        y_train = pd.Series(received_data.y_train)

        arima_model = fit_arima_model(y_train)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        arima_model_path = f'models/arima_{date_timestamp}.pkl'
        joblib.dump(arima_model, arima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"arima_model_path": arima_model_path}
# @app.post("/fit_and_store_sarima_model_call")

def fit_and_store_sarima_model_call(received_data: InputData):
    try:
        if not received_data.y_train:
            raise ValueError("y_train is missing.")
        if received_data.seasonality is None:
            raise ValueError("seasonality is missing.")

        seasonality = received_data.seasonality
        print(seasonality)
        y_train = pd.Series(received_data.y_train)

        sarima_model = fit_sarima_model(y_train, seasonality)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        sarima_model_path = f'models/sarima_{date_timestamp}.pkl'
        joblib.dump(sarima_model, sarima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"sarima_model_path": sarima_model_path}


@app.post("/fit_models_in_parallel")
async def fit_models_in_parallel(received_data: InputData):
    loop = asyncio.get_running_loop()

    fitting_arima = loop.run_in_executor(executor, fit_and_store_arima_model_call, received_data)
    fitting_sarima = loop.run_in_executor(executor, fit_and_store_sarima_model_call, received_data)

    arima_model, sarima_model = await asyncio.gather(fitting_arima, fitting_sarima)

    return {"arima" : arima_model, "sarima" : sarima_model}

# Model Predictions
@app.post("/predict_train_test")
def predict_train_test(received_data: InputData):
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