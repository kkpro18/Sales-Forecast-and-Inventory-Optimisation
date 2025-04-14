import asyncio
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional, Dict, List, Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint

from App.utils.data_preprocessing import convert_to_dict, transform_data, handle_outliers, \
    fix_dates_and_split_into_product_sales_and_daily_sales, split_training_testing_data, handle_missing_valuesxw
from App.utils.forecasting_sales import fit_arima_model, fit_sarima_model, fit_arimax_model, fit_sarimax_model, \
    fit_fb_prophet_model, fit_fb_prophet_model_with_exog, predict

"uvicorn App.utils.fast_api_app:app --port 8000 --reload"

app = FastAPI()
executor = ProcessPoolExecutor(max_workers=2) # do one less than the number of cores || no. jobs in parallel

class InputData(BaseModel):
    column_mapping: Optional[Dict[str, str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    train : Optional[List[Dict[str, Any]]] = None
    test : Optional[List[Dict[str, Any]]] = None
    train_with_exog : Optional[List[Dict[str, Any]]] = None
    test_with_exog : Optional[List[Dict[str, Any]]] = None
    daily_store_sales: Optional[List[Dict[str, Any]]] = None
    daily_product_sales: Optional[List[Dict[str, Any]]] = None
    train_daily_store_sales: Optional[List[Dict[str, Any]]] = None
    test_daily_store_sales: Optional[List[Dict[str, Any]]] = None
    train_daily_product_sales: Optional[List[Dict[str, Any]]] = None
    test_daily_product_sales: Optional[List[Dict[str, Any]]] = None
    train_data : Optional[List[Dict[str, Any]]] = None
    test_data : Optional[List[Dict[str, Any]]] = None
    X_train: Optional[List[Dict[str, Any]]] = None
    X_test: Optional[List[Dict[str, Any]]] = None
    y_train: Optional[Dict[int, Any]] = None
    y_test: Optional[Dict[int, Any]] = None
    seasonality: Optional[conint(gt=0)] = None
    test_forecast_steps: Optional[conint(gt=0)] = None
    model_name : Optional[str] = None
    model_path : Optional[str] = None
    product_name: Optional[str] = None
    model_one : Optional[str] = None
    model_two : Optional[str] = None
    is_log_transformed: Optional[bool] = None

## Pre Processing

@app.post("/transform_data_api")
def transform_data_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping
        data, is_log_transformed = transform_data(data, column_mapping)
        return {"data": convert_to_dict(data), "is_log_transformed": is_log_transformed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/handle_outliers_api")
def handle_outliers_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return convert_to_dict(handle_outliers(data, column_mapping))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/fix_dates_and_split_into_product_sales_and_daily_sales_api")
def fix_dates_and_split_into_product_sales_and_daily_sales_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        daily_store_sales, daily_product_sales = fix_dates_and_split_into_product_sales_and_daily_sales(data, column_mapping)
        # needs to be filtered i.e NaN to None as JSON does not allow this
        daily_store_sales, daily_product_sales = convert_to_dict(daily_store_sales), convert_to_dict(daily_product_sales)
        return {
            "daily_store_sales": daily_store_sales,
            "daily_product_sales": daily_product_sales,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_test_split_api")
def train_test_split_api(received_data: InputData):
    try:
        daily_store_sales = pd.DataFrame(received_data.daily_store_sales)
        daily_product_sales = pd.DataFrame(received_data.daily_product_sales)

        column_mapping = received_data.column_mapping

        train_daily_store_sales, test_daily_store_sales = split_training_testing_data(daily_store_sales, column_mapping)
        train_daily_product_sales, test_daily_product_sales = split_training_testing_data(daily_product_sales, column_mapping)

        return {
            "train_daily_store_sales": convert_to_dict(train_daily_store_sales),
            "test_daily_store_sales": convert_to_dict(test_daily_store_sales),
            "train_daily_product_sales": convert_to_dict(train_daily_product_sales),
            "test_daily_product_sales": convert_to_dict(test_daily_product_sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/handle_missing_values_api")
def handle_missing_values_api(received_data: InputData):
    try:

        train_daily_store_sales = pd.DataFrame(received_data.train_daily_store_sales)
        test_daily_store_sales = pd.DataFrame(received_data.test_daily_store_sales)
        train_daily_product_sales = pd.DataFrame(received_data.train_daily_product_sales)
        test_daily_product_sales = pd.DataFrame(received_data.test_daily_product_sales)
        column_mapping = received_data.column_mapping

        train_daily_store_sales, test_daily_store_sales = handle_missing_values(train_daily_store_sales,
                                                                                test_daily_store_sales, column_mapping)
        train_daily_product_sales, test_daily_product_sales = handle_missing_values(train_daily_product_sales,
                                                                                    test_daily_product_sales,
                                                                                    column_mapping)

        return {
            "train_daily_store_sales": convert_to_dict(train_daily_store_sales),
            "test_daily_store_sales": convert_to_dict(test_daily_store_sales),
            "train_daily_product_sales": convert_to_dict(train_daily_product_sales),
            "test_daily_product_sales": convert_to_dict(test_daily_product_sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model Fitting

# Methods to fit models
def store_arima_model(received_data: InputData):
    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        y_train = pd.Series(received_data.y_train)

        arima_model = fit_arima_model(y_train)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            arima_model_path = f'models/arima_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            arima_model_path = f'models/arima_{date_timestamp}.pkl'
        joblib.dump(arima_model, arima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"arima_model_path": arima_model_path}
def store_sarima_model(received_data: InputData):
    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        if received_data.seasonality is None:
            raise ValueError("seasonality is missing.")

        seasonality = received_data.seasonality
        y_train = pd.Series(received_data.y_train)

        sarima_model = fit_sarima_model(y_train, seasonality)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            sarima_model_path = f'models/sarima_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            sarima_model_path = f'models/sarima_{date_timestamp}.pkl'
        joblib.dump(sarima_model, sarima_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"sarima_model_path": sarima_model_path}
def store_arimax_model(received_data: InputData):
    # print(f"\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n{received_data.y_train}\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")

    try:
        if not received_data.y_train or len(received_data.y_train) == 0 or not received_data.X_train or len(received_data.X_train) == 0:
            raise ValueError("y_train or X_train is missing / empty")
        X_train = pd.DataFrame(received_data.X_train)
        y_train = pd.Series(received_data.y_train)
        arimax_model = fit_arimax_model(X_train, y_train)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            arimax_model_path = f'models/arimax_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            arimax_model_path = f'models/arimax_{date_timestamp}.pkl'
        joblib.dump(arimax_model, arimax_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"arimax_model_path": arimax_model_path}
def store_sarimax_model(received_data: InputData):

    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        if received_data.seasonality is None:
            raise ValueError("seasonality is missing.")

        seasonality = received_data.seasonality
        X_train = pd.DataFrame(received_data.X_train)
        y_train = pd.Series(received_data.y_train)
        sarimax_model = fit_sarimax_model(X_train, y_train, seasonality)
        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            sarimax_model_path = f'models/sarimax_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            sarimax_model_path = f'models/sarimax_{date_timestamp}.pkl'
        joblib.dump(sarimax_model, sarimax_model_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"sarimax_model_path": sarimax_model_path}
def store_fb_prophet_model_without_exog(received_data: InputData):
    try:
        if not received_data.train or len(received_data.train) == 0:
            raise ValueError("data is missing / empty")

        train = pd.DataFrame(received_data.train)

        prophet_model = fit_fb_prophet_model(train, received_data.column_mapping)

        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            prophet_model_path = f'models/prophet_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            prophet_model_path = f'models/prophet_{date_timestamp}.pkl'
        joblib.dump(prophet_model, prophet_model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"fb_prophet_model_path": prophet_model_path}
def store_fb_prophet_model_with_exog(received_data: InputData):
    try:
        if not received_data.train_with_exog or len(received_data.train_with_exog) == 0:
            raise ValueError("train_with_exog is missing / empty")

        train_with_exog = pd.DataFrame(received_data.train_with_exog)
        prophet_model = fit_fb_prophet_model_with_exog(train_with_exog, received_data.column_mapping)

        date_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        if received_data.product_name is not None:
            prophet_model_path = f'models/prophet_with_exog_{received_data.product_name}_{date_timestamp}.pkl'
        else:
            prophet_model_path = f'models/prophet_with_exog_{date_timestamp}.pkl'
        joblib.dump(prophet_model, prophet_model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        return {"fb_prophet_with_exog_model_path": prophet_model_path}


# API Endpoint for fitting models in Parallel
@app.post("/fit_models_in_parallel_api")
async def fit_models_in_parallel_api(received_data: InputData): # https://blog.stackademic.com/fastapi-parallel-processing-1eaa67981ab9
    loop = asyncio.get_running_loop()
    if received_data.model_one == "arima" and received_data.model_two == "sarima":
        fitted_arima = loop.run_in_executor(executor, store_arima_model, received_data)
        fitted_sarima = loop.run_in_executor(executor, store_sarima_model, received_data)
        arima, sarima = await asyncio.gather(fitted_arima, fitted_sarima)
        return {"arima": arima,"sarima": sarima}

    elif received_data.model_one == "arimax" and received_data.model_two == "sarimax":
        fitted_arimax = loop.run_in_executor(executor, store_arimax_model, received_data)
        fitted_sarimax = loop.run_in_executor(executor, store_sarimax_model, received_data)
        arimax, sarimax = await asyncio.gather(fitted_arimax, fitted_sarimax)
        return {"arimax": arimax, "sarimax": sarimax}

    elif received_data.model_one == "fb_prophet_without_exog" and received_data.model_two == "fb_prophet_with_exog":
        fitted_fb_prophet_without_exog = loop.run_in_executor(executor, store_fb_prophet_model_without_exog, received_data)
        fitted_fb_prophet_with_exog = loop.run_in_executor(executor, store_fb_prophet_model_with_exog, received_data)

        fb_prophet_without_exog, fb_prophet_with_exog = await asyncio.gather(fitted_fb_prophet_without_exog, fitted_fb_prophet_with_exog)
        return {"fb_prophet_without_exog": fb_prophet_without_exog, "fb_prophet_with_exog": fb_prophet_with_exog }


# Model Predictions
@app.post("/predict_train_test_api")
def predict_train_test_api(received_data: InputData):
    column_mapping = received_data.column_mapping
    if received_data.model_path is None:
        raise ValueError("Model path is None")
    try:
        model_path = received_data.model_path
        if received_data.model_name == "arima" or received_data.model_name == "sarima":
            test_forecast_steps = received_data.test_forecast_steps

            y_train_prediction = predict(model_path=model_path)
            y_test_prediction = predict(model_path=model_path, forecast_periods=test_forecast_steps)

        elif received_data.model_name == "arimax" or received_data.model_name == "sarimax":

            test_forecast_steps = received_data.test_forecast_steps
            if test_forecast_steps <= 0: # if -1 it means exog
                raise ValueError(f"Invalid forecast periods: {test_forecast_steps}")
            column_mapping.pop("price_column") # price is used as a exog feature
            train_exog_features = pd.DataFrame(received_data.X_train).drop(columns=column_mapping.values(), errors="ignore")
            test_exog_features = pd.DataFrame(received_data.X_test).drop(columns=column_mapping.values(), errors="ignore")
            y_train_prediction = predict(model_path=model_path, data=train_exog_features)
            y_test_prediction = predict(model_path=model_path, forecast_periods=test_forecast_steps, data=test_exog_features)

        elif received_data.model_name == "fb_prophet_with_exog" or received_data.model_name == "fb_prophet_without_exog":
            if received_data.train is None or len(received_data.train) == 0:
                raise ValueError("Train data is None")
            if received_data.test is None or len(received_data.test) == 0:
                raise ValueError("Test data is None")

            train = pd.DataFrame(received_data.train).rename(columns={column_mapping["date_column"]: 'ds', column_mapping["quantity_sold_column"]: 'y'})
            test = pd.DataFrame(received_data.test).rename(columns={column_mapping["date_column"]: 'ds', column_mapping["quantity_sold_column"]: 'y'})

            train["ds"] = pd.to_datetime(train["ds"], errors="coerce")
            test["ds"] = pd.to_datetime(test["ds"], errors="coerce")

            y_train_prediction = predict(model_path=model_path, model_name=received_data.model_name, data=train)
            y_test_prediction = predict(model_path=model_path, model_name=received_data.model_name, data=test)

            if y_train_prediction is None or len(y_train_prediction) == 0:
                raise ValueError("Y Train is Empty")
            if y_test_prediction is None or len(y_test_prediction) == 0:
                raise ValueError("Y Test is Empty")

        if received_data.is_log_transformed is True:
            y_train_prediction = np.round(np.expm1(y_train_prediction))
            y_test_prediction = np.round(np.expm1(y_test_prediction))
        else:
            y_train_prediction = np.round(y_train_prediction)
            y_test_prediction = np.round(y_test_prediction)

        if y_train_prediction.isna().any():
            raise ValueError("y_train_prediction contains NaNs")
        if y_test_prediction.isna().any():
            raise ValueError("y_test_prediction contains NaNs")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"y_train_prediction": y_train_prediction, "y_test_prediction": y_test_prediction}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)