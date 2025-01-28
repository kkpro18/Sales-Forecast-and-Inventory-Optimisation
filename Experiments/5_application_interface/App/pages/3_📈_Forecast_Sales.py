import streamlit as st
import statsmodels
from statsmodels.tsa.stattools import adfuller
from sktime.forecasting.arima import AutoARIMA
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
    uploaded_dataset = st.session_state["uploaded_dataset"]
    date_column = st.session_state["date_column"]
    sales_column = st.session_state["units_sold_column"]
    st.multiselect("Select features that can be used to predict sales", uploaded_dataset.columns.drop(date_column).drop(sales_column))

    start_button = st.button("Begin Forecasting Sales")

    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value
        # ADF Test
        sales_data = pd.DataFrame(uploaded_dataset[sales_column].head(10000), index=uploaded_dataset.index)

        result = adfuller(sales_data.head(100)) # as data is huge for now we test with head
        print("ADF statistic:", result[0])
        print("p-value:", result[1])
        print("Critical values:", result[4])

        if result[1] < 0.05:  # less than significance level, so is stationary
            st.write("The data is stationary, so we reject the null hypothesis and our d value is 0")
            warnings.filterwarnings("ignore")
            #
            # m1 = auto_arima(y=uploaded_dataset[sales_column].head(10000),X=uploaded_dataset[date_column].head(10000), d=0,seasonal=True,stationary=True,m=12,trace=True)
            forecaster = AutoARIMA(
                sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True
            )
            forecaster.fit(sales_data)

            st.write(forecaster.summary())
            # )
            # fig, axes = plt.subplots(1, 2, figsize=(12, 8))
            # x = np.arange(uploaded_dataset.shape[0])
            #
            # # Plot m=1
            # axes[0].scatter(x, uploaded_dataset, marker='x')
            # axes[0].plot(x, m1.predict(n_periods=uploaded_dataset.shape[0]))
            # plt.show()

        else:
            st.write("The data is non-stationary, so we need to apply differencing until the data becomes stationary.")

        # create acf and dcf plots to identify other parameters for ARIMA model (p,q)

        # fit model

        # forecast

        # calculate performance metrics
        pass
    st.page_link("pages/4_⚙️_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy", icon="⚙️")
else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")