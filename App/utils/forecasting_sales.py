import time
import streamlit as st
from sklearn.metrics import root_mean_squared_error, r2_score
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error
from App.utils.session_manager import SessionManager
import pmdarima as pm
import plotly.graph_objects as go

def split_training_testing_data(data, column_mapping):
    features = column_mapping.copy().pop("quantity_sold_column")
    target = column_mapping["quantity_sold_column"]
    # 70 : 30 split
    train_size = int(len(data) * 0.70)

    train = data[:train_size]
    test = data[train_size:]

    X_train, X_test = train[features.values()], test[features.values()]
    y_train, y_test = train[target], test[target]

    return X_train, X_test, y_train, y_test
def print_performance_metrics(model, y_train, y_train_prediction, y_test, y_test_prediction):
    performance_metrics = {
        "Train: Mean Absolute Scaled Error (MASE)": mean_absolute_scaled_error(y_test, y_train_prediction, y_train),
        "Test: Mean Absolute Scaled Error (MASE)": mean_absolute_scaled_error(y_test, y_test_prediction, y_train),
        "Train: Root Mean Squared Error (RMSE)": root_mean_squared_error(y_train, y_train_prediction),
        "Test: Mean Absolute Error (MAE)": root_mean_squared_error(y_test, y_test_prediction),
        "Train: Akaike Information Criterion (AIC)": model.aic,
        "Test: Bayesian Information Criterion (BIC)": model.bic,
        "Train: R Squared (R²)": r2_score(y_train, y_train_prediction),
        "Test: R Squared (R²)": r2_score(y_test, y_test_prediction)
    }
    for metric, value in performance_metrics.items():
        st.metric(label=metric, value=round(value, 4))
    # print(performance_metrics)

def get_seasonality():
    seasonality_frequency = [7, 12, 365, 4]
    selected_seasonality = st.radio(label="Enter the region where your store is based?", options=seasonality_frequency)
    SessionManager.set_state("selected_seasonality", selected_seasonality)

    st.markdown(f"You Selected {SessionManager.get_state(selected_seasonality)}.")
    time.sleep(4)

    return SessionManager.get_state(selected_seasonality)

def fit_arima_model(y_train):
    st.write("ARIMA")
    arima_model = pm.auto_arima(y_train,
                                # feed in just one variable - uni variate model - learn trends from sales
                                seasonal=False, trace=True,
                                error_action='ignore',  # don't need to know if an order does not work
                                suppress_warnings=False,  # don't want convergence warnings
                                stepwise=True,  # set to stepwise which is quicker,
                                n_jobs=-1)  # n_jobs uses all processor cores for faster build

    # st.write(arima_model.summary())
    return arima_model

def fit_sarima_model(y_train, seasonality):
    sarima_model = pm.auto_arima(y_train,
                                 seasonal=True, m=seasonality,
                                 trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True,  # set to stepwise for quicker - false does grid search takes longer
                                 n_jobs=-1)  # uses all cpu cores

    # st.write(sarima_model.summary())
    return sarima_model

# TBD

def fit_sarimax_model():
    pass

def fit_lstm_model():
    pass

def fit_fb_prophet_model():
    pass

def plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_train, y=y_train, mode='lines', name='Training Data', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=X_test, y=y_test, mode='lines', name='Testing Data', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=X_test, y=y_test_prediction, mode='lines', name='Predicted Sales', line=dict(color='blue')))

    fig.frames = []
    for i in range(len(X_test)):
        fig.frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=X_train[:i + 1], y=y_train[:i + 1], mode='lines', line=dict(color='red')),
                    go.Scatter(x=X_test[:i + 1], y=y_test[:i + 1], mode='lines', line=dict(color='green')),
                    go.Scatter(x=X_test[:i + 1], y=y_test_prediction[:i + 1], mode='lines', line=dict(color='blue')),
                ],
                name=str(i)
            )
        )

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
