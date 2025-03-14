import datetime
from typing import Dict, List, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, conint
from App.utils.data_preprocessing import format_dates, handle_missing_values, handle_outliers, encode_product_column
from App.utils.forecasting_sales import fit_arima_model, fit_sarima_model, predict
"uvicorn App.utils.fast_api_app:app --reload --port 8000"
"uvicorn App.utils.fast_api_app:app --host 127.0.0.1 --port 8000 --workers 4"
app = FastAPI()


class InputData(BaseModel):
    column_mapping: Dict[str, str]
    data: List[dict[str, Any]]
    X_train: Dict[str, List]
    X_test: Dict[str, List]
    y_train: Dict[str, List]
    y_test: Dict[str, List]
    seasonality: conint(gt=0)
    train_forecast_steps : conint(gt=0)
    test_forecast_steps: conint(gt=0)
    model_name : str


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
@app.post("/fit_and_store_arima_model_call")
def fit_and_store_arima_model_call(received_data: InputData):
    try:
        y_train = pd.DataFrame(received_data.y_train)
        arima_model = fit_arima_model(y_train)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        joblib.dump(arima_model, f'models/arima_{date_timestamp}.pkl')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "ARIMA model saved successfully"}
@app.post("/fit_and_store_sarima_model_call")
def fit_and_store_sarima_model_call(received_data: InputData):
    try:
        y_train = pd.DataFrame(received_data.y_train)
        seasonality = received_data.seasonality

        sarima_model = fit_sarima_model(y_train, seasonality)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        joblib.dump(sarima_model, f'models/sarima_{date_timestamp}.pkl')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "SARIMA model saved successfully"}

# Model Predictions
@app.post("/predict_train_test")
def predict_train_test(received_data: InputData):
    try:
        train_forecast_steps = received_data.train_forecast_steps
        test_forecast_steps = received_data.test_forecast_steps
        model_name = received_data.model_name

        y_train_prediction = predict(train_forecast_steps, model_name)
        y_test_prediction = predict(test_forecast_steps, model_name)

        return {"y_train_prediction": y_train_prediction, "y_test_prediction": y_test_prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, port=8000)