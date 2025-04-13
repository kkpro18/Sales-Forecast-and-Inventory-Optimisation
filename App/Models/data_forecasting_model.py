import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
from prophet import Prophet
import streamlit as st
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from permetrics.regression import RegressionMetric
from App.utils.session_manager import SessionManager
import uuid

def get_seasonality():
    seasonality_frequency = [4, 7, 12, 365]

    selected_seasonality = st.radio(label="Select the seasonality that applies to your store (4 - Quarterly, 7 - Weekly,  12 - Monthly, 365 - Yearly)", options=seasonality_frequency)
    SessionManager.set_state("selected_seasonality", selected_seasonality)
    if st.button("Confirm Selection"):
        st.write(f"You Selected {SessionManager.get_state('selected_seasonality')}.")

        return SessionManager.get_state('selected_seasonality')

def fit_arima_model(y_train):
    arima_model = pm.auto_arima(y_train,
                                # feed in just one variable - uni variate model - learn trends from sales
                                seasonal=False, trace=True,
                                error_action='ignore',  # don't need to know if an order does not work
                                suppress_warnings=False,  # don't want convergence warnings
                                stepwise=True,  # set to stepwise which is quicker,
                                scoring='mae',
                                )
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

def fit_arimax_model(X_exog, y_train):
    arimax_model = pm.auto_arima(y=y_train,
                                 X=X_exog,
                                 seasonal=False, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 scoring='mae',
                                 )

    return arimax_model

def fit_sarimax_model(X_exog, y_train, seasonality):
    sarimax_model = pm.auto_arima(y=y_train,
                                 X=X_exog,
                                 seasonal=True, m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker
                                 scoring='mae',
                                 )

    return sarimax_model

def fit_fb_prophet_model(full_data, column_mapping):
    data = full_data.rename(columns={column_mapping['date_column']: 'ds', column_mapping['quantity_sold_column']: 'y'}) # passing in expected format
    data['ds'] = pd.to_datetime(data['ds'])
    prophet_model = Prophet()

    prophet_model.fit(data)

    return prophet_model
def fit_fb_prophet_model_with_exog(full_data, column_mapping):
    data = full_data.rename(columns={column_mapping['date_column']: 'ds',
                                     column_mapping['quantity_sold_column']: 'y'})  # passing in expected format
    data['ds'] = pd.to_datetime(data['ds'])
    prophet_model = Prophet()

    exog_features = full_data.columns.difference(column_mapping.values())

    for exog_feature in exog_features:
        prophet_model.add_regressor(exog_feature)

    prophet_model.fit(data)

    return prophet_model

def fit_lstm_model():
    pass

def predict(model_path, forecast_periods=None, model_name=None, data=None):
    if model_path is None or len(model_path) == 0:
        st.error("No model path provided.")
    model = joblib.load(model_path)
    if model_name == "fb_prophet_without_exog" or model_name == "fb_prophet_with_exog":
        return model.predict(data)['yhat']
    if forecast_periods is None:
        return model.predict_in_sample(X=data)  # Train
    else:
        if data is not None:
            return model.predict(n_periods=forecast_periods, X=data)  # Test / Predict Future
        else:
            return model.predict(n_periods=forecast_periods)  # Test / Predict Future

def mean_direction_accuracy(y_true, y_predicted):
    """
    function inspired by https://datasciencestunt.com/mean-directional-accuracy-of-time-series-forecast/
    """
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)

    # calculate the signs of the differences between consecutive values
    y_true_diff = np.diff(y_true)
    y_true_signs = np.sign(y_true_diff)
    y_predicted_diff = np.diff(y_predicted)
    y_predicted_signs = np.sign(y_predicted_diff)

    # count the number of times the signs are the same
    num_correct_signs = np.sum(y_true_signs == y_predicted_signs)

    # calculate the MDA value
    mean_accuracy = num_correct_signs / (len(y_true) - 1)

    return mean_accuracy

def print_performance_metrics(y_train, y_train_prediction, y_test, y_test_prediction):

    y_train, y_train_prediction = np.array(y_train), np.array(y_train_prediction)
    y_test, y_test_prediction = np.array(y_test), np.array(y_test_prediction)

    performance_metrics = {
        "Train: Mean Arc-Tangent Absolute Percentage Error (MAAPE) in Radians": round(RegressionMetric(y_train, y_train_prediction).mean_arctangent_absolute_percentage_error(), 4),
        "Test: Mean Arc-Tangent Absolute Percentage Error (MAAPE) in Radians": round(RegressionMetric(y_test, y_test_prediction).mean_arctangent_absolute_percentage_error(), 4),
        "Train: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_train, y_train_prediction), 4),
        "Test: Root Mean Squared Error (RMSE)": round(root_mean_squared_error(y_test, y_test_prediction), 4),
        "Train: Mean Absolute Scaled Error (MASE)":round(mean_absolute_scaled_error(y_train, y_train_prediction, y_train=y_train), 4),
        "Test: Mean Absolute Scaled Error (MASE)": round(mean_absolute_scaled_error(y_test, y_test_prediction, y_train=y_train), 4),
        "Train: Mean Direction Accuracy (MDA)": round(mean_direction_accuracy(y_train, y_train_prediction), 4),
        "Test: Mean Direction Accuracy (MDA)": round(mean_direction_accuracy(y_test, y_test_prediction), 4),
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