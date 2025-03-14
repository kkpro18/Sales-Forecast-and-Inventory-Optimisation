import json

import joblib
import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.forecasting_sales import split_training_testing_data, print_performance_metrics, get_seasonality

st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not SessionManager.get_state("preprocess_data_complete"):
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    data = SessionManager.get_state("data").head(int(len(SessionManager.get_state("data"))))
    column_mapping = SessionManager.get_state("column_mapping")

    if st.button("Begin Forecasting Sales"):
        X_train, X_test, y_train, y_test = split_training_testing_data(data, column_mapping)

        st.write("""### Training Data""")
        st.dataframe(X_train)
        st.dataframe(y_train)
        st.write("""### Testing Data""")
        st.dataframe(X_test)
        st.dataframe(y_test)

        json_response = SessionManager.fast_api_call("fit_and_store_arima_model_call",
                                                      y_train = y_train.to_dict())
        if json_response.status_code == 200:
            st.write(joblib.load('arima.pkl').summary())
        else:
            st.error(json_response.text)

        json_response = SessionManager.fast_api_call("predict_train_test",
                                                      train_forecast_steps = len(X_train) - 1,
                                                      test_forecast_steps = len(X_test),
                                                      model_name = 'arima')

        if json_response.status_code == 200:
            y_train_prediction_arima = json_response.json()["y_train_prediction"]
            y_test_prediction_arima = json_response.json()["y_train_prediction"]
        else:
            st.error(json_response.text)
        print_performance_metrics(y_train_prediction_arima, y_train)
        print_performance_metrics(y_test_prediction_arima, y_test)



        sales_seasonality = get_seasonality()
        SessionManager.fast_api_call("fit_and_store_sarima_model_call",
                                      y_train = y_train.to_dict(),
                                      seasonality = sales_seasonality)
        st.write(joblib.load('sarima.pkl').summary())
        json_response = SessionManager.fast_api_call("predict_train_test",
                                                      train_forecast_steps=len(X_train) - 1,
                                                      test_forecast_steps=len(X_test),
                                                      model_name='sarima')
        y_train_prediction_sarima = json_response.json()["y_train_prediction"]
        y_test_prediction_sarima = json_response.json()["y_test_prediction"]
        print_performance_metrics(y_train_prediction_sarima, y_train)
        print_performance_metrics(y_test_prediction_sarima, y_test)

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")