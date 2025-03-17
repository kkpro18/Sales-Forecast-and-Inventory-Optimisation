import joblib
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
import streamlit as st
from sklearn.metrics import root_mean_squared_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error, mean_absolute_error
import uuid
from App.utils.session_manager import SessionManager


def split_training_testing_data(data, column_mapping, univariate=False):
    # st.write(column_mapping)
    if univariate:
        features = column_mapping["date_column"]
    else:
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
def predict(model_path, forecast_periods=None):
    if model_path is None or len(model_path) == 0:
        st.error("No model path provided.")
    if forecast_periods is None:
        predictions = joblib.load(model_path).predict_in_sample()  # Train
    else:
        predictions = joblib.load(model_path).predict(n_periods=forecast_periods)  # Test / Predict Future
    return predictions

def predict_store_wide_sales(X_train, X_test, y_train, y_test, column_mapping):

    json_response = SessionManager.fast_api("fit_models_in_parallel_api", y_train=y_train.to_dict(),
                                            seasonality=SessionManager.get_state('selected_seasonality'))
    if json_response.status_code == 200:
        arima_model_path = json_response.json()["arima"]["arima_model_path"]
        sarima_model_path = json_response.json()["sarima"]["sarima_model_path"]
    else:
        st.error(json_response.text)

    # Predict ARIMA
    json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test), model_path=arima_model_path)
    if json_response.status_code == 200:
        y_train_prediction_arima = pd.Series(json_response.json()["y_train_prediction"])
        y_test_prediction_arima = pd.Series(json_response.json()["y_test_prediction"])
    else:
        st.error(json_response.text)

    # Predict SARIMA
    json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                            model_path=sarima_model_path)

    if json_response.status_code == 200:
        y_train_prediction_sarima = pd.Series(json_response.json()["y_train_prediction"])
        y_test_prediction_sarima = pd.Series(json_response.json()["y_test_prediction"])
    else:
        st.error(json_response.text)

    st.markdown("### ARIMA Model:")
    st.write(joblib.load(arima_model_path).summary())
    print_performance_metrics(arima_model_path, y_train, y_train_prediction_arima, y_test, y_test_prediction_arima)
    plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arima, column_mapping, univariate=True)


    st.markdown("### SARIMA Model:")
    st.write(joblib.load(sarima_model_path).summary())
    print_performance_metrics(sarima_model_path, y_train, y_train_prediction_sarima, y_test, y_test_prediction_sarima)
    plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarima, column_mapping, univariate=True)




def print_performance_metrics(model_path, y_train, y_train_prediction, y_test, y_test_prediction):

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

def plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction, column_mapping, univariate=False):
    if univariate:
        X_train_dates = X_train
        X_test_dates = X_test
    else:
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

    st.plotly_chart(fig, key=uuid.uuid4())