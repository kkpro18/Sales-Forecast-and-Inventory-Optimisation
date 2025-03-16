import joblib
import numpy as np
import pandas as pd
import streamlit as st

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import root_mean_squared_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error
import pmdarima as pm
import plotly.graph_objects as go

from App.utils.session_manager import SessionManager


def split_training_testing_data(data, column_mapping):
    # st.write(column_mapping)
    features = column_mapping.copy()
    features.pop("quantity_sold_column")
    features = features.values()

    target = column_mapping["quantity_sold_column"]

    # 70 : 30 split
    train_size = int(len(data) * 0.70)

    train = data[:train_size]
    test = data[train_size:]

    X_train, X_test = train[features], test[features]
    y_train, y_test = train[target], test[target]
    st.toast("Data has been split into training and test set 70:30 Ratio")

    return X_train, X_test, y_train, y_test

def get_seasonality():
    seasonality_frequency = [7, 12, 365]

    selected_seasonality = st.radio(label="Select the seasonality that applies to your store (7 - Weekly,  12 - Monthly, 365 - Yearly)", options=seasonality_frequency)
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

    # st.write(sarima_model.summary())
    return sarima_model

# TBD

def fit_sarimax_model():
    pass

def fit_lstm_model():
    pass

def fit_fb_prophet_model():
    pass


def predict(model_path, forecast_periods):
    if forecast_periods is None:
        predictions = joblib.load(model_path).predict_in_sample()  # Train
    else:
        predictions = joblib.load(model_path).predict(n_periods=forecast_periods)  # Test / Predict Future
    return predictions


def print_performance_metrics(model_path, y_train, y_train_prediction, y_test, y_test_prediction):
    # st.write(f"y_train length: {len(y_train)}, y_train_prediction length: {len(y_train_prediction)}")
    # st.write(f"y_test length: {len(y_test)}, y_test_prediction length: {len(y_test_prediction)}")

    model = joblib.load(model_path)
    performance_metrics = {
        "Train: Mean Absolute Scaled Error (MASE)": mean_absolute_scaled_error(y_train, y_train_prediction, y_train=y_train),
        "Test: Mean Absolute Scaled Error (MASE)": mean_absolute_scaled_error(y_test, y_test_prediction, y_train=y_train),
        "Train: Root Mean Squared Error (RMSE)": root_mean_squared_error(y_train, y_train_prediction),
        "Test: Root Mean Squared Error (RMSE)": root_mean_squared_error(y_test, y_test_prediction),
        "Train: Mean Absolute Error (MAE)": mean_absolute_error(y_train, y_train_prediction),
        "Test: Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_test_prediction),
        "Train: R Squared (R²)": r2_score(y_train, y_train_prediction),
        "Test: R Squared (R²)": r2_score(y_test, y_test_prediction),
        "Akaike Information Criterion (AIC)": model.aic(),
        "Bayesian Information Criterion (BIC)": model.bic(),
    }
    left, right = st.columns(2)
    for i, (metric, value) in enumerate(performance_metrics.items()):
        if i < 5:
            left.metric(label=metric, value=round(value, 4))
        else:
            right.metric(label=metric, value=round(value, 4))


def plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction, column_mapping):
    X_train_dates = X_train[column_mapping["date_column"]]
    X_test_dates = X_test[column_mapping["date_column"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train_dates, y=y_train, mode='lines', name='Training Data', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=X_test_dates, y=y_test, mode='lines', name='Testing Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=X_test_dates, y=y_test_prediction, mode='lines', name='Predicted Sales', line=dict(color='blue')))

    frames = [
        go.Frame(
            data=[
                go.Scatter(x=X_train_dates, y=y_train, mode='lines', line=dict(color='red')),  # Static
                go.Scatter(x=X_test_dates[:i + 1], y=y_test[:i + 1], mode='lines', line=dict(color='green')),
                go.Scatter(x=X_test_dates[:i + 1], y=y_test_prediction[:i + 1], mode='lines', line=dict(color='blue'))
            ]
        )
        for i in range(len(X_test_dates))
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

    st.plotly_chart(fig)



    st.dataframe(X_train_dates)
