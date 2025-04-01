import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
import streamlit as st
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_percentage_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error
import uuid
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
    )

    # st.write(arima_model.summary())
    return arima_model

def fit_sarima_model(y_train, seasonality):
    sarima_model = pm.auto_arima(y_train,
                                 seasonal=True, m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
    )

    return sarima_model

# TBD

def fit_arimax_model(X_train, y_train):
    arimax_model = pm.auto_arima(y_train,
                                 exogenous=X_train,
                                 seasonal=False, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 )

    return arimax_model

def fit_sarimax_model(X_train, y_train, seasonality):

    sarima_model = pm.auto_arima(y_train,
                                 exogenous=X_train,
                                 seasonal=True, m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 )

    return sarima_model

def fit_lstm_model():
    pass

def fit_fb_prophet_model():
    pass

def predict(model_path, forecast_periods=None):
    if model_path is None or len(model_path) == 0:
        st.error("No model path provided.")
    if forecast_periods is None:
        predictions = joblib.load(model_path).predict_in_sample()  # Train
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

            print_performance_metrics(arima_model_path, y_train, y_train_prediction_arima, y_test,
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
            print_performance_metrics(sarima_model_path, y_train, y_train_prediction_sarima, y_test,
                                      y_test_prediction_sarima)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarima, column_mapping)
        else:
            st.error(json_response.text)

    else:
        st.error(json_response.text)




async def predict_sales_multivariate(train, test, column_mapping, product_name=None):
    features = train.columns.tolist()
    st.write("features are", features)
    features.remove(column_mapping["quantity_sold_column"])
    features.remove(column_mapping["date_column"])

    target = column_mapping["quantity_sold_column"]

    X_train, X_test, y_train, y_test = train[features], test[features], train[target], test[target]

    json_response = SessionManager.fast_api("fit_models_in_parallel_api",
                                            y_train=y_train.to_dict(),
                                            X_train=X_train.to_dict(orient='records'),
                                            seasonality=SessionManager.get_state('selected_seasonality'),
                                            product_name=product_name, model_one = "arimax", model_two = "sarimax")
    if json_response.status_code == 200:

        arimax_model_path = json_response.json()["arimax"]["arimax_model_path"]
        sarimax_model_path = json_response.json()["sarimax"]["sarimax_model_path"]

        # Predict ARIMA
        json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                model_path=arimax_model_path)
        if json_response.status_code == 200:
            y_train_prediction_arimax = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_arimax = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### ARIMAX Model:")
            st.write(joblib.load(arimax_model_path).summary())

            print_performance_metrics(arimax_model_path, y_train, y_train_prediction_arimax, y_test,
                                      y_test_prediction_arimax)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arimax, column_mapping)
        else:
            st.error(json_response.text)

        # Predict SARIMA
        json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                model_path=sarimax_model_path)

        if json_response.status_code == 200:
            y_train_prediction_sarimax = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_sarimax = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### SARIMAX Model:")
            st.write(joblib.load(sarimax_model_path).summary())
            print_performance_metrics(sarimax_model_path, y_train, y_train_prediction_sarimax, y_test,
                                      y_test_prediction_sarimax)
            plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarimax, column_mapping)
        else:
            st.error(json_response.text)

    else:
        st.error(json_response.text)

def print_performance_metrics(model_path, y_train, y_train_prediction, y_test, y_test_prediction):

    model = joblib.load(model_path)
    performance_metrics = {
        "Train: Mean Absolute Scaled Error (MASE)":round(
            mean_absolute_scaled_error(y_train, y_train_prediction, y_train=y_train), 4),
        "Test: Mean Absolute Scaled Error (MASE)": round(
            mean_absolute_scaled_error(y_test, y_test_prediction, y_train=y_train), 4),
        "Train: Mean Absolute Percentage Error (MAPE)": round(
            mean_absolute_percentage_error(y_train, y_train_prediction), 4),
        "Test: Mean Absolute Percentage Error (MAPE)": round(mean_absolute_percentage_error(y_test, y_test_prediction), 4),
        "Train: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_train, y_train_prediction), 4),
        "Test: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_test, y_test_prediction), 4),
        "Train: Mean Absolute Error (MAE)": round(mean_absolute_error(y_train, y_train_prediction), 4),
        "Test: Mean Absolute Error (MAE)": round(mean_absolute_error(y_test, y_test_prediction), 4),
        "Train: R Squared (R²)": round(r2_score(y_train, y_train_prediction), 4),
        "Test: R Squared (R²)": round(r2_score(y_test, y_test_prediction), 4),
        "Akaike Information Criterion (AIC)": round(model.aic(), 4),
        "Bayesian Information Criterion (BIC)": round(model.bic(), 4),
    }
    left, right = st.columns(2)
    for i, (metric, value) in enumerate(performance_metrics.items()):
        if i < 5:
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