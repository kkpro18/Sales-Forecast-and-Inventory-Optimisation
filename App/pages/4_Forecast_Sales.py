import joblib
import pandas as pd
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
    data = SessionManager.get_state("data").head(int(len(SessionManager.get_state("data")) * 0.005))
    column_mapping = SessionManager.get_state("column_mapping")
    get_seasonality()
    if st.button("Begin Forecasting Sales"):
        X_train, X_test, y_train, y_test = split_training_testing_data(data, column_mapping)
        #
        # st.write("""### Training Data""")
        # st.dataframe(X_train.head())
        # st.dataframe(y_train.head())
        # st.write("""### Testing Data""")
        # st.dataframe(X_test.head())
        # st.dataframe(y_test.head())

        json_response = SessionManager.fast_api_call("fit_and_store_arima_model_call",
                                                     y_train=y_train.to_dict())
        if json_response.status_code == 200:
            arima_model_path = json_response.json()["arima_model_path"]
            st.write(joblib.load(arima_model_path).summary())
        else:
            st.error(json_response.text)

        json_response = SessionManager.fast_api_call("predict_train_test",
                                                     test_forecast_steps=len(X_test),
                                                     model_path=arima_model_path)

        if json_response.status_code == 200:
            y_train_prediction_arima = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_arima = pd.Series(json_response.json()["y_test_prediction"])
        else:
            st.error(json_response.text)
        # st.write(y_train.shape)
        # st.write(y_train_prediction_arima.shape)
        # st.write(y_test.shape)
        # st.write(y_test_prediction_arima.shape)

        print_performance_metrics(arima_model_path, y_train, y_train_prediction_arima, y_test, y_test_prediction_arima)

        json_response = SessionManager.fast_api_call("fit_and_store_sarima_model_call",
                                     y_train=y_train.to_dict(),
                                     seasonality=SessionManager.get_state('selected_seasonality'))

        if json_response.status_code == 200:
            sarima_model_path = json_response.json()["sarima_model_path"]
            st.write(joblib.load(sarima_model_path).summary())
        else:
            st.error(json_response.text)

        json_response = SessionManager.fast_api_call("predict_train_test",
                                                     test_forecast_steps=len(X_test),
                                                     model_path=sarima_model_path)
        if json_response.status_code == 200:
            y_train_prediction_sarima = json_response.json()["y_train_prediction"]
            y_test_prediction_sarima = json_response.json()["y_test_prediction"]
        else:
            st.error(json_response.text)

        print_performance_metrics(sarima_model_path, y_train, y_train_prediction_sarima, y_test, y_test_prediction_sarima)

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")
