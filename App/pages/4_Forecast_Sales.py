import joblib
import pandas as pd
import streamlit as st
from App.utils.session_manager import SessionManager
from App.utils.forecasting_sales import split_training_testing_data, print_performance_metrics, get_seasonality, plot_prediction

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
    data = SessionManager.get_state("data").head(int(len(SessionManager.get_state("data")) * 0.05))
    column_mapping = SessionManager.get_state("column_mapping")
    get_seasonality()
    if st.button("Begin Forecasting Sales"):

        ## Seperate Data daily sales instead for whole store
        ## do product level forecasting and reformat data so it contains total sold per day of that product
        X_train, X_test, y_train, y_test = split_training_testing_data(data, column_mapping)
        #
        # st.write("""### Training Data""")
        # st.dataframe(X_train.head())
        # st.dataframe(y_train.head())
        # st.write("""### Testing Data""")
        # st.dataframe(X_test.head())
        # st.dataframe(y_test.head())

        # # st.write(y_train.shape)
        # # st.write(y_train_prediction_arima.shape)
        # # st.write(y_test.shape)
        # # st.write(y_test_prediction_arima.shape)
        #

        json_response = SessionManager.fast_api("fit_models_in_parallel_api", y_train=y_train.to_dict(), seasonality=SessionManager.get_state('selected_seasonality'))
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
        json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test), model_path=sarima_model_path)
        if json_response.status_code == 200:
            y_train_prediction_sarima = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_sarima = pd.Series(json_response.json()["y_test_prediction"])
        else:
            st.error(json_response.text)

        st.subheader("# ARIMA Model:")
        st.write(joblib.load(arima_model_path).summary())
        print_performance_metrics(arima_model_path, y_train, y_train_prediction_arima, y_test, y_test_prediction_arima)
        plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arima, column_mapping)
        st.write(X_train.head())
        st.write(X_test.head())

        #
        # st.subheader("# SARIMA Model:")
        # st.write(joblib.load(sarima_model_path).summary())
        # print_performance_metrics(sarima_model_path, y_train, y_train_prediction_sarima, y_test, y_test_prediction_sarima)
        # plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarima, column_mapping)

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")
