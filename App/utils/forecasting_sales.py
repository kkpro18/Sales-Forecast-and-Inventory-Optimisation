import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
import prophet
from prophet import Prophet
import streamlit as st
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error
import uuid

from sktime.utils.mlflow_sktime import load_model

from App.utils.session_manager import SessionManager

def get_seasonality():
    seasonality_frequency = [4, 7, 12, 365]

    selected_seasonality = st.radio(label="Select the seasonality that applies to your store (4 - Quarterly, 7 - Weekly,  12 - Monthly, 365 - Yearly)", options=seasonality_frequency)
    SessionManager.set_state("selected_seasonality", selected_seasonality)
    if st.button("Confirm Selection"):
        st.write(f"You Selected {SessionManager.get_state('selected_seasonality')}.")

        return SessionManager.get_state('selected_seasonality')

def fit_arima_model(y_train):
    st.write("ARIMA")
    arima_model = pm.auto_arima(y_train,
                                # feed in just one variable - uni variate model - learn trends from sales
                                seasonal=False, trace=True,
                                error_action='ignore',  # don't need to know if an order does not work
                                suppress_warnings=False,  # don't want convergence warnings
                                stepwise=True,  # set to stepwise which is quicker,
                                scoring='mae',
                                )

    # st.write(arima_model.summary())
    return arima_model

def fit_sarima_model(y_train, seasonality):
    sarima_model = pm.auto_arima(y_train,
                                 seasonal=True,
                                 m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 scoring='mae',
                                 )

    return sarima_model

# TBD

def fit_arimax_model(X_train, y_train):
    arimax_model = pm.auto_arima(y=y_train,
                                 X=X_train,
                                 seasonal=False, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 scoring='mae',
                                 )

    return arimax_model

def fit_sarimax_model(X_train, y_train, seasonality):

    sarima_model = pm.auto_arima(y=y_train,
                                 X=X_train,
                                 seasonal=True, m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 scoring='mae',
                                 )

    return sarima_model


def fit_fb_prophet_model(full_data, column_mapping):
    data = full_data.rename(columns={column_mapping["date_column"]: 'ds', column_mapping["quantity_sold_column"]: 'y'}) # passing in expected format
    prophet_model = Prophet()

    prophet_model.fit(data)

    return prophet_model

def fit_fb_prophet_model_with_exog(full_data, column_mapping):
    data = full_data.rename(  # passing in expected format
        columns={
            column_mapping["date_column"]: 'ds',
            column_mapping["quantity_sold_column"]: 'y'
        })
    prophet_model = Prophet()

    exog_features = full_data.columns.difference(column_mapping)

    for exog_feature in exog_features:
        prophet_model.add_regressor(exog_feature)

    prophet_model.fit(data)

    return prophet_model

def fit_lstm_model():
    pass

def predict(model_path, forecast_periods=None, model_name="", data=None, exog=None):
    if model_path is None or len(model_path) == 0:
        st.error("No model path provided.")
    if forecast_periods is None:
        predictions = joblib.load(model_path).predict_in_sample(X=exog)  # Train
    else:
        if exog is not None:
            predictions = joblib.load(model_path).predict(n_periods=forecast_periods, X=exog)  # Test / Predict Future
        else:
            predictions = joblib.load(model_path).predict(n_periods=forecast_periods)  # Test / Predict Future

    return predictions

async def predict_sales_univariate(train, test, column_mapping, product_name=None):

    features = column_mapping["date_column"]
    target = column_mapping["quantity_sold_column"]

    X_train, X_test, y_train, y_test = train[features], test[features], train[target], test[target]

    json_response = SessionManager.fast_api("fit_models_in_parallel_api", model_one="arima", model_two="sarima", y_train=y_train.to_dict(), seasonality=SessionManager.get_state('selected_seasonality'), product_name=product_name)
    if json_response.status_code == 200:

        arima_model_path = json_response.json()["arima"]["arima_model_path"]
        sarima_model_path = json_response.json()["sarima"]["sarima_model_path"]

        # Predict ARIMA
        json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                model_path=arima_model_path)
        if json_response.status_code == 200:
            y_train_prediction_arima = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_arima = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### ARIMA Model:")
            st.write(joblib.load(arima_model_path).summary())
            # st.write(joblib.load(arima_model_path).params)

            print_performance_metrics(y_train, y_train_prediction_arima, y_test,
                                      y_test_prediction_arima)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arima, column_mapping)
        else:
            st.error(json_response.text)

        # Predict SARIMA
        json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                model_path=sarima_model_path)

        if json_response.status_code == 200:
            y_train_prediction_sarima = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_sarima = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### SARIMA Model:")
            st.write(joblib.load(sarima_model_path).summary())
            # st.write(joblib.load(sarima_model_path).params)

            print_performance_metrics(y_train, y_train_prediction_sarima, y_test,
                                      y_test_prediction_sarima)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarima, column_mapping)
        else:
            st.error(json_response.text)


        # fb_prophet_model_path = json_response.json()["fb_prophet"]["fb_prophet_model_path"]
        # # Predict fb_prophet
        # json_response = SessionManager.fast_api("predict_train_test_api",
        #                                         model_path=fb_prophet_model_path,
        #                                         model_name="fb_prophet")
        #
        # if json_response.status_code == 200:
        #     y_train_prediction_fb_prophet = pd.Series(json_response.json()["y_train_prediction"])
        #     y_test_prediction_fb_prophet = pd.Series(json_response.json()["y_test_prediction"])
        #
        #     st.markdown("### FB-Prophet Model:")
        #     # st.write(load(fb_prophet_model_path).summary())
        #     print_performance_metrics(fb_prophet_model_path, y_train,
        #                               y_train_prediction_fb_prophet, y_test,
        #                               y_test_prediction_fb_prophet)
        #     plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_fb_prophet,
        #                     column_mapping)
        # else:
        #     st.error(json_response.text)

    else:
        st.error(json_response.text)
async def predict_sales_multivariate(train, test, column_mapping, product_name=None):
    features = train.columns.tolist()
    features.remove(column_mapping["quantity_sold_column"])
    features.remove(column_mapping["date_column"])
    exog_features = features.copy()

    plot_features = column_mapping["date_column"]


    target = column_mapping["quantity_sold_column"]

    X_train_exog, X_test_exog, X_train, X_test, y_train, y_test = train[exog_features], test[exog_features], train[plot_features], test[plot_features], train[target], test[target]

    # st.write("exog features are", exog_features)
    # st.write("plot features are", plot_features)
    # st.write("x test", len(X_test))
    # st.write("y test", len(y_test))

    json_response = SessionManager.fast_api("fit_models_in_parallel_api",
                                            y_train=y_train.to_dict(),
                                            X_train=X_train_exog.to_dict(orient='records'),
                                            seasonality=SessionManager.get_state('selected_seasonality'),
                                            product_name=product_name, model_one = "arimax", model_two = "sarimax")
    if json_response.status_code == 200:

        arimax_model_path = json_response.json()["arimax"]["arimax_model_path"]
        sarimax_model_path = json_response.json()["sarimax"]["sarimax_model_path"]
        # fb_prophet_with_exog_model_path = json_response.json()["fb_prophet"]["fb_prophet_with_exog_model_path"]

        # Predict ARIMAX
        json_response = SessionManager.fast_api("predict_train_test_api",
                                                test_forecast_steps=len(X_test_exog),
                                                model_path=arimax_model_path,
                                                model_name="ARIMAX",
                                                X_train=X_train_exog.to_dict(orient='records'),
                                                X_test=X_test_exog.to_dict(orient='records'),
                                                column_mapping=column_mapping)
        if json_response.status_code == 200:
            y_train_prediction_arimax = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_arimax = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### ARIMAX Model:")
            st.write(joblib.load(arimax_model_path).summary())
            # st.write(joblib.load(arimax_model_path).params)

            print_performance_metrics(y_train, y_train_prediction_arimax, y_test, y_test_prediction_arimax)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arimax, column_mapping)
        else:
            st.error(json_response.text)

        # Predict SARIMA
        json_response = SessionManager.fast_api("predict_train_test_api",
                                                test_forecast_steps=len(X_test_exog),
                                                model_path=sarimax_model_path,
                                                model_name="SARIMAX",
                                                X_train=X_train_exog.to_dict(orient='records'),
                                                X_test=X_test_exog.to_dict(orient='records'),
                                                column_mapping=column_mapping)

        if json_response.status_code == 200:
            y_train_prediction_sarimax = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_sarimax = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### SARIMAX Model:")
            st.write(joblib.load(sarimax_model_path).summary())
            # st.write(joblib.load(sarimax_model_path).params)

            print_performance_metrics(y_train, y_train_prediction_sarimax, y_test,
                                      y_test_prediction_sarimax)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarimax, column_mapping)
        else:
            st.error(json_response.text)

        # # Predict fb_prophet with exog
        # json_response = SessionManager.fast_api("predict_train_test_api", model_path=fb_prophet_with_exog_model_path, model_name="fb_prophet_with_exog")
        #
        # if json_response.status_code == 200:
        #     y_train_prediction_fb_prophet_with_exog = pd.Series(json_response.json()["y_train_prediction"])
        #     y_test_prediction_fb_prophet_with_exog = pd.Series(json_response.json()["y_test_prediction"])
        #
        #     st.markdown("### FB-Prophet Model With Exog Features:")
        #     st.write(joblib.load(fb_prophet_with_exog_model_path).summary())
        #     print_performance_metrics(fb_prophet_with_exog_model_path, y_train, y_train_prediction_fb_prophet_with_exog, y_test,
        #                               y_test_prediction_fb_prophet_with_exog)
        #     plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_fb_prophet_with_exog, column_mapping)
        # else:
        #     st.error(json_response.text)

    else:
        st.error(json_response.text)



def print_performance_metrics(y_train, y_train_prediction, y_test, y_test_prediction):

    y_train, y_train_prediction = np.array(y_train), np.array(y_train_prediction)
    y_test, y_test_prediction = np.array(y_test), np.array(y_test_prediction)

    y_train_filtered, y_train_prediction_filtered = y_train[y_train != 0], y_train_prediction[y_train != 0]
    y_test_filtered, y_test_prediction_filtered = y_test[y_test != 0], y_test_prediction[y_test != 0]


    performance_metrics = {
        "Train: Mean Absolute Percentage Error (MAPE)": round(mean_absolute_percentage_error(y_train_filtered, y_train_prediction_filtered), 4),
        "Test: Mean Absolute Percentage Error (MAPE)": round(mean_absolute_percentage_error(y_test_filtered, y_test_prediction_filtered), 4),
        "Train: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_train_filtered, y_train_prediction_filtered), 4),
        "Test: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_test_filtered, y_test_prediction_filtered), 4),
        "Train: Mean Absolute Scaled Error (MASE)":round(mean_absolute_scaled_error(y_train_filtered, y_train_prediction_filtered, y_train=y_train_filtered), 4),
        "Test: Mean Absolute Scaled Error (MASE)": round(mean_absolute_scaled_error(y_test_filtered, y_test_prediction_filtered, y_train=y_train_filtered), 4),
    }
    left, right = st.columns(2)
    for metric,value in performance_metrics.items():
        if metric.startswith("Train"):
            left.metric(label=metric, value=round(value, 4))
        else:
            right.metric(label=metric, value=round(value, 4))

def plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction, column_mapping, multivariate=False):
    if multivariate:
        X_train = X_train[column_mapping["date_column"]]
        X_test = X_test[column_mapping["date_column"]]


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train, y=y_train, mode='lines', name='Training Data', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=X_test, y=y_test, mode='lines', name='Testing Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=X_test, y=y_test_prediction, mode='lines', name='Predicted Sales', line=dict(color='blue')))

    frames = [
        go.Frame(
            data=[
                go.Scatter(x=X_train, y=y_train, mode='lines', line=dict(color='red')),  # Static
                go.Scatter(x=X_test[:i + 1], y=y_test[:i + 1], mode='lines', line=dict(color='green')),
                go.Scatter(x=X_test[:i + 1], y=y_test_prediction[:i + 1], mode='lines', line=dict(color='blue'))
            ]
        )
        for i in range(len(X_test))
    ]

    fig.frames = frames

    fig.update_layout(
        updatemenus=[
            dict(type='buttons',
                 buttons=[
                     dict(label='Play', method='animate',
                          args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)]),
                     dict(label='Pause', method='animate',
                          args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])
                          ]
                 )
        ]
    )

    st.plotly_chart(fig, key=uuid.uuid4())