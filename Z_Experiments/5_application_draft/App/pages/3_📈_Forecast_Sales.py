import time

import sklearn.model_selection
import streamlit as st
import statsmodels
import plotly.graph_objects as go
from matplotlib import animation
from matplotlib.dates import DateFormatter
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# to run application type this into the terminal "streamlit run 5_application_draft/App/0_Home.py"
st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

if 'uploaded_dataset' in st.session_state:
    date_column = st.session_state["date_column"]  # ensure its in correct form
    units_sold_column = st.session_state["units_sold_column"]
    product_column = st.session_state["product_column"]
    unit_price_column = st.session_state["unit_price_column"]
    """
    using 25% to speed up model fitting
    """
    #                                                                                                                              change below to make full data
    uploaded_dataset = st.session_state["uploaded_dataset"][
        [date_column, "product_encoded", units_sold_column, unit_price_column]].head(int(0.05 * st.session_state["uploaded_dataset"].shape[0]))
    # uploaded_dataset.set_index(date_column, inplace=True)
    # uploaded_dataset.sort_index(inplace=True)

    start_button = st.button("Begin Forecasting Sales")


    def animate_data(X_test, y_test, y_test_prediction):
        # Create the plot with True and Predicted values
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=X_test, y=y_test, mode='lines', name='True'))
        fig.add_trace(go.Scatter(x=X_test, y=y_test_prediction, mode='lines', name='Predicted'))

        # Animation frames
        fig.frames = [go.Frame(data=[go.Scatter(x=X_test[:i + 1], y=y_test[:i + 1], mode='lines'),
                                     go.Scatter(x=X_test[:i + 1], y=y_test_prediction[:i + 1], mode='lines')],
                               name=str(i)) for i in range(len(X_test))]

        # Play/Pause buttons
        fig.update_layout(updatemenus=[dict(type='buttons', buttons=[
            dict(label='Play', method='animate',
                 args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label='Pause', method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate')])
        ])])

        return fig


    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value
        # ADF Test

        result = adfuller(uploaded_dataset[units_sold_column].values)  # as data is huge for now we test with head
        print("ADF statistic:", result[0])
        print("p-value:", result[1])
        print("Critical values:", result[4])

        if result[1] < 0.05:  # less than significance level, so is stationary
            st.write("The data is stationary, so we reject the null hypothesis and our d value is 0")
            # warnings.filterwarnings("ignore")
            # """
            # Daily Data m=7 weekly, m=14 biweekly two weeks for pay,  m=30 - monthly , m=365 - yearly
            # Weekly Data m=2 - biweekly, m=4 - 4 weeks, m=52 - yearly depending on how seasonality repeats
            # Monthly Data m=3 - quarterly, m=12 - yearly
            # Quarterly Data m=4 yearly
            # """

            train_size = int(len(uploaded_dataset) * 0.70)
            # train_split_index = uploaded_dataset.index[train_size]
            train = uploaded_dataset[:train_size]
            print("Train: ", train)

            test = uploaded_dataset[train_size:]
            print("Test Date: ", test.index)

            X_train, X_test = train, test
            y_train, y_test = train[units_sold_column], test[units_sold_column]

            st.write("ARIMA")
            stepwise_fit_ARIMA = pm.auto_arima(y_train,
                                               # feed in just one variable - uni variate model - learn trends from sales
                                               seasonal=False, trace=True,
                                               error_action='ignore',  # don't need to know if an order does not work
                                               suppress_warnings=False,  # don't want convergence warnings
                                               stepwise=True,
                                               n_jobs=-1)  # set to stepwise which skips some combinations, becomes quicker, n_jobs uses all processor cores for faster build

            st.write(stepwise_fit_ARIMA.summary())

            y_test_prediction_ARIMA = stepwise_fit_ARIMA.predict(len(X_test))

            mean_absolute_error = mean_absolute_error(y_test, y_test_prediction_ARIMA)
            st.write(mean_absolute_error)

            root_mean_squared_error = root_mean_squared_error(y_test, y_test_prediction_ARIMA)
            st.write(root_mean_squared_error)
            fig = animate_data(X_test, y_test, y_test_prediction_ARIMA)
            st.plotly_chart(fig)

            seasonality_frequency = [7, 12, 365, 4]
            if 'selected_seasonality' not in st.session_state:
                st.session_state.selected_seasonality = 0
            selected_seasonality = st.session_state.selected_seasonality
            seasonality = st.radio(label="Enter the region where your store is based?", options=seasonality_frequency,
                                   index=st.session_state.selected_seasonality)
            st.session_state["seasonality"] = seasonality
            st.session_state.selected_seasonality = seasonality_frequency.index(seasonality)
            st.markdown(f"You Selected {st.session_state.seasonality}.")
            time.sleep(5)
            stepwise_fit_SARIMA = pm.auto_arima(y_train,
                                                seasonal=True, m=365,
                                                trace=True,
                                                error_action='ignore',  # don't want to know if an order does not work
                                                suppress_warnings=True,  # don't want convergence warnings
                                                stepwise=True)  # set to stepwise

            st.write(stepwise_fit_SARIMA.summary())

        else:
            st.write("The data is non-stationary, so we need to apply differencing until the data becomes stationary.")
            # make it stationary

        # fit model

        # forecast

        # calculate performance metrics
    st.page_link("pages/4_⚙️_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                 icon="⚙️")
else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")