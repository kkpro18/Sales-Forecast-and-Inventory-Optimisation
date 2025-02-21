import sklearn.model_selection
import streamlit as st
import statsmodels
from attr.filters import exclude
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import pandas as pd
import warnings

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
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
    uploaded_dataset = st.session_state["uploaded_dataset"][[date_column, "product_encoded", units_sold_column, unit_price_column]]
    uploaded_dataset.set_index(date_column, inplace=True).sort_index()


    start_button = st.button("Begin Forecasting Sales")

    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value
        # ADF Test
        sales_data = pd.DataFrame(uploaded_dataset[units_sold_column].head(10000), index=uploaded_dataset.index)

        result = adfuller(sales_data.head(100)) # as data is huge for now we test with head
        print("ADF statistic:", result[0])
        print("p-value:", result[1])
        print("Critical values:", result[4])

        if result[1] < 0.05:  # less than significance level, so is stationary
            st.write("The data is stationary, so we reject the null hypothesis and our d value is 0")
            warnings.filterwarnings("ignore")
            """
            Daily Data m=7 weekly, m=14 biweekly two weeks for pay,  m=30 - monthly , m=365 - yearly
            Weekly Data m=2 - biweekly, m=4 - 4 weeks, m=52 - yearly depending on how seasonality repeats
            Monthly Data m=3 - quarterly, m=12 - yearly
            Quarterly Data m=4 yearly
            """

            X_train, Y_train, x_test, y_test = sklearn.model_selection.train_test_split(X=uploaded_dataset[date_column])
            st.write("ARIMA")
            stepwise_fit_ARIMA = pm.auto_arima(uploaded_dataset,
                                         start_p=1, start_q=1,
                                         max_p=3, max_q=3,
                                         seasonal=False,
                                         d=None, trace=True,
                                         error_action='ignore',  # don't want to know if an order does not work
                                         suppress_warnings=True,  # don't want convergence warnings
                                         stepwise=True)  # set to stepwise

            st.write(stepwise_fit_ARIMA.summary())

            st.write("SARIMA")
            stepwise_fit_SARIMA = pm.auto_arima(uploaded_dataset,
                                         start_p=1, start_q=1,
                                         max_p=3, max_q=3,
                                         seasonal=True, m=365,
                                         start_P=0, start_Q=0,
                                         max_P=2, max_Q=2,
                                         d=None, D=None, trace=True,
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
    st.page_link("pages/4_⚙️_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy", icon="⚙️")
else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")