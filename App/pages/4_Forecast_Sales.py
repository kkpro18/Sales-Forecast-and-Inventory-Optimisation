import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.forecasting_sales import split_training_testing_data, fit_arima_model, print_performance_metrics, get_seasonality, fit_sarima_model

st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

data = SessionManager.get_state("data")
column_mapping = SessionManager.get_state("column_mapping")
preprocess_data_complete = SessionManager.get_state("preprocess_data_complete")

if data is None or column_mapping is None:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not preprocess_data_complete:
    st.warning("Dataset has not been pre-processed, 👈 Please Preprocess it ")
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    data = data[column_mapping.values()]

    if st.button("Begin Forecasting Sales"):
        X_train, X_test, y_train, y_test = split_training_testing_data(data, column_mapping)

        arima_model = fit_arima_model(y_train)
        st.write(arima_model.summary())
        y_train_prediction_ARIMA = arima_model.predict(len(X_train)-1)
        y_test_prediction_ARIMA = arima_model.predict(len(X_test))
        print_performance_metrics(y_test_prediction_ARIMA, y_test)

        sales_seasonality = get_seasonality()
        sarima_model = fit_sarima_model(y_train, seasonality = sales_seasonality)
        st.write(sarima_model.summary())
        y_train_prediction_sarima = sarima_model.predict(len(X_train) - 1)
        y_test_prediction_sarima = sarima_model.predict(len(X_test))
        print_performance_metrics(y_test_prediction_sarima, y_test)

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")