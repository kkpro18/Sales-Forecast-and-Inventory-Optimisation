import asyncio
from concurrent.futures import ProcessPoolExecutor
from fastapi import APIRouter, HTTPException
from App.utils.fastapi.schemas.input_data_model import InputData
from App.Models import data_forecasting_model
import pandas as pd
from datetime import datetime
import joblib
import numpy as np

router = APIRouter()
executor = ProcessPoolExecutor(max_workers=2)

# Model Fitting

# Methods to fit models
def store_arima_model(received_data: InputData):
    try:
        if not received_data.y_train or len(received_data.y_train) == 0:
            raise ValueError("y_train is missing / empty")
        y_train = pd.Series(received_data.y_train)

        arima_model = data_forecasting_model.fit_arima_model(y_train)
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

        sarima_model = data_forecasting_model.fit_sarima_model(y_train, seasonality)
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
        if not received_data.y_train or len(received_data.y_train) == 0 or not received_data.X_train or len(
                received_data.X_train) == 0:
            raise ValueError("y_train or X_train is missing / empty")
        X_train = pd.DataFrame(received_data.X_train)
        y_train = pd.Series(received_data.y_train)
        arimax_model = data_forecasting_model.fit_arimax_model(X_train, y_train)
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
        sarimax_model = data_forecasting_model.fit_sarimax_model(X_train, y_train, seasonality)
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

        prophet_model = data_forecasting_model.fit_fb_prophet_model(train, received_data.column_mapping)

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
        prophet_model = data_forecasting_model.fit_fb_prophet_model_with_exog(train_with_exog,
                                                                              received_data.column_mapping)

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
@router.post("/fit_all_models_in_parallel_api")
async def fit_all_models_in_parallel_api(received_data: InputData):  # https://blog.stackademic.com/fastapi-parallel-processing-1eaa67981ab9
    loop = asyncio.get_running_loop()
    if received_data.model_one == "arima" and received_data.model_two == "sarima":
        fitted_arima = loop.run_in_executor(executor, store_arima_model, received_data)
        fitted_sarima = loop.run_in_executor(executor, store_sarima_model, received_data)
        arima, sarima = await asyncio.gather(fitted_arima, fitted_sarima)
        return {"arima": arima, "sarima": sarima}

    elif received_data.model_one == "arimax" and received_data.model_two == "sarimax":
        fitted_arimax = loop.run_in_executor(executor, store_arimax_model, received_data)
        fitted_sarimax = loop.run_in_executor(executor, store_sarimax_model, received_data)
        arimax, sarimax = await asyncio.gather(fitted_arimax, fitted_sarimax)
        return {"arimax": arimax, "sarimax": sarimax}

    elif received_data.model_one == "fb_prophet_without_exog" and received_data.model_two == "fb_prophet_with_exog":
        fitted_fb_prophet_without_exog = loop.run_in_executor(executor, store_fb_prophet_model_without_exog,
                                                              received_data)
        fitted_fb_prophet_with_exog = loop.run_in_executor(executor, store_fb_prophet_model_with_exog, received_data)

        fb_prophet_without_exog, fb_prophet_with_exog = await asyncio.gather(fitted_fb_prophet_without_exog,
                                                                             fitted_fb_prophet_with_exog)
        return {"fb_prophet_without_exog": fb_prophet_without_exog, "fb_prophet_with_exog": fb_prophet_with_exog}


# Model Predictions
@router.post("/predict_train_test_api")
def predict_train_test_api(received_data: InputData):
    column_mapping = received_data.column_mapping
    try:
        if received_data.model_path is None:
            raise ValueError("Model path is None")
        model_path = received_data.model_path
        if received_data.model_name == "arima" or received_data.model_name == "sarima":
            test_forecast_steps = received_data.test_forecast_steps

            y_train_prediction = data_forecasting_model.predict(model_path=model_path)
            y_test_prediction = data_forecasting_model.predict(model_path=model_path, forecast_periods=test_forecast_steps)
        elif received_data.model_name == "arimax" or received_data.model_name == "sarimax":

            test_forecast_steps = received_data.test_forecast_steps
            if test_forecast_steps <= 0:  # if -1 it means exog
                raise ValueError(f"Invalid forecast periods: {test_forecast_steps}")
            column_mapping.pop("price_column")  # price is used as a exog feature
            train_exog_features = pd.DataFrame(received_data.X_train).drop(columns=column_mapping.values(),
                                                                           errors="ignore")
            test_exog_features = pd.DataFrame(received_data.X_test).drop(columns=column_mapping.values(),
                                                                         errors="ignore")
            y_train_prediction = data_forecasting_model.predict(model_path=model_path, data=train_exog_features)
            y_test_prediction = data_forecasting_model.predict(model_path=model_path, forecast_periods=test_forecast_steps,
                                        data=test_exog_features)


        elif received_data.model_name == "fb_prophet_with_exog" or received_data.model_name == "fb_prophet_without_exog":
            if received_data.train is None or len(received_data.train) == 0:
                raise ValueError("Train data is None")
            if received_data.test is None or len(received_data.test) == 0:
                raise ValueError("Test data is None")

            train = pd.DataFrame(received_data.train).rename(
                columns={column_mapping["date_column"]: 'ds', column_mapping["quantity_sold_column"]: 'y'})
            test = pd.DataFrame(received_data.test).rename(
                columns={column_mapping["date_column"]: 'ds', column_mapping["quantity_sold_column"]: 'y'})

            train["ds"] = pd.to_datetime(train["ds"], errors="coerce")
            test["ds"] = pd.to_datetime(test["ds"], errors="coerce")

            y_train_prediction = data_forecasting_model.predict(model_path=model_path, model_name=received_data.model_name, data=train)
            y_test_prediction = data_forecasting_model.predict(model_path=model_path, model_name=received_data.model_name, data=test)

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