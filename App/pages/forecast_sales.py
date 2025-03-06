import streamlit as st

from App.utils.session_manager import SessionManager

st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

data = SessionManager.get_state("data")
column_map = SessionManager.get_state("column_map")
preprocess_data_complete = SessionManager.get_state("preprocess_data_complete")

if data is None or column_map is None:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/upload_data.py", label="👈 Upload The Dataset", icon="📁")
elif not preprocess_data_complete:
    st.warning("Dataset has not been pre-processed, 👈 Please Preprocess it ")
    st.page_link("pages/preprocess_data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    data = data[column_map.date_column]

    if st.button("Begin Forecasting Sales"):
        X_train, X_test, y_train, y_test = split_training_testing_data(data, column_map)

        st.write("ARIMA")
        arima_model = fit_arima_model(y_train)
        st.write(arima_model.summary())
        y_train_prediction_ARIMA = arima_model.predict(len(X_train)-1)
        y_test_prediction_ARIMA = arima_model.predict(len(X_test))
        print_performance_metrics(y_test_prediction_ARIMA, y_test)

        st.write("SARIMA")
        sales_seasonality = get_seasonality()
        sarima_model = fit_sarima_model(y_train, seasonality = sales_seasonality)
        st.write(sarima_model.summary())
        y_train_prediction_sarima = sarima_model.predict(len(X_train) - 1)
        y_test_prediction_sarima = sarima_model.predict(len(X_test))
        print_performance_metrics(y_test_prediction_sarima, y_test)

        st.page_link("pages/inventory_policy_simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")