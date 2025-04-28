import streamlit as st
from App.Models import data_forecasting_model
from App.utils.session_manager import SessionManager
import pandas as pd

def handle_store_sales_data(column_mapping):
    date_column = column_mapping["date_column"]

    train_daily_store_sales = SessionManager.get_state("train_daily_store_sales")
    train_daily_store_sales[date_column] = train_daily_store_sales[date_column].astype(str)

    test_daily_store_sales = SessionManager.get_state("test_daily_store_sales")
    test_daily_store_sales[date_column] = test_daily_store_sales[date_column].astype(str)

    train_daily_store_sales_with_exog = SessionManager.get_state("train_daily_store_sales_with_exog")
    train_daily_store_sales_with_exog[date_column] = train_daily_store_sales_with_exog[date_column].astype(str)

    test_daily_store_sales_with_exog = SessionManager.get_state("test_daily_store_sales_with_exog")
    test_daily_store_sales_with_exog[date_column] = test_daily_store_sales_with_exog[date_column].astype(str)

    return train_daily_store_sales, test_daily_store_sales, train_daily_store_sales_with_exog, test_daily_store_sales_with_exog

def handle_product_sales_data(column_mapping):
    train_product_sales_raw = SessionManager.get_state("train_daily_product_sales")
    train_product_sales_raw[column_mapping['date_column']] = train_product_sales_raw[
        column_mapping['date_column']].astype(str)
    train_product_sales_grouped = train_product_sales_raw.groupby(
        column_mapping["product_column"])  # groups by product column
    train_product_names = list(train_product_sales_grouped.groups.keys())  # gets each product names

    test_product_sales_raw = SessionManager.get_state("test_daily_product_sales")
    test_product_sales_raw[column_mapping['date_column']] = test_product_sales_raw[
        column_mapping['date_column']].astype(str)
    test_product_sales_grouped = test_product_sales_raw.groupby(
        column_mapping["product_column"])  # groups by product column
    test_product_names = list(test_product_sales_grouped.groups.keys())  # gets each product name

    train_product_sales_raw_with_exog = SessionManager.get_state("train_daily_product_sales_with_exog")
    train_product_sales_raw_with_exog[column_mapping['date_column']] = train_product_sales_raw_with_exog[
        column_mapping['date_column']].astype(str)
    train_product_sales_with_exog_grouped = train_product_sales_raw_with_exog.groupby(
        column_mapping["product_column"])  # groups by product column
    train_product_with_exog_names = list(train_product_sales_with_exog_grouped.groups.keys())  # gets each product names

    test_product_sales_with_exog_raw = SessionManager.get_state("test_daily_product_sales_with_exog")
    test_product_sales_with_exog_raw[column_mapping['date_column']] = test_product_sales_with_exog_raw[
        column_mapping['date_column']].astype(str)
    test_product_sales_with_exog_grouped = test_product_sales_with_exog_raw.groupby(
        column_mapping["product_column"])  # groups by product column
    test_product_with_exog_names = list(test_product_sales_with_exog_grouped.groups.keys())  # gets each product names

    return train_product_sales_grouped, train_product_names, test_product_sales_grouped, test_product_names, train_product_sales_with_exog_grouped, train_product_with_exog_names, test_product_sales_with_exog_grouped, test_product_with_exog_names


def handle_seasonality_input():
    """
    Function to get the seasonality value from the user input.
    """
    try:
        seasonality = data_forecasting_model.get_seasonality()
    except Exception as e:
        st.error(e)
        return None
    else:
        return seasonality


async def handle_arima_sarima_training_and_predictions(train, test, column_mapping, product_name=None, seasonality=None,
                                                       is_log_transformed=None):
    """
    Trains Arima, Sarima
    Generates Predictions
    Plots Prediction
    Performance Metrics to evaluate model

    """
    try:
        features = column_mapping["date_column"]
        target = column_mapping["quantity_sold_column"]

        if seasonality is None:
            seasonality = SessionManager.get_state('selected_seasonality')
        if is_log_transformed is None:
            is_log_transformed = SessionManager.get_state("is_log_transformed")

        X_train, X_test, y_train, y_test = train[features], test[features], train[target], test[target]
        json_response = SessionManager.fast_api("fit_all_models_in_parallel_api", model_one="arima", model_two="sarima",
                                                y_train=y_train.to_dict(),
                                                seasonality=seasonality,
                                                product_name=product_name)
        if json_response.status_code == 200:

            arima_model_path = json_response.json()["arima"]["arima_model_path"]
            sarima_model_path = json_response.json()["sarima"]["sarima_model_path"]

            # Predict ARIMA
            json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                    model_path=arima_model_path, model_name="arima",
                                                    is_log_transformed=is_log_transformed)
            if json_response.status_code == 200:
                y_train_prediction_arima = pd.Series(json_response.json()["y_train_prediction"])
                y_test_prediction_arima = pd.Series(json_response.json()["y_test_prediction"])

                st.markdown("### ARIMA Model:")
                # st.write(joblib.load(arima_model_path).summary())
                # st.write(joblib.load(arima_model_path).params)

                data_forecasting_model.print_performance_metrics(y_train, y_train_prediction_arima, y_test,
                                                                 y_test_prediction_arima)
                data_forecasting_model.interpret_slope(X_test, y_test_prediction_arima)
                data_forecasting_model.plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arima,
                                                       column_mapping)
            else:
                st.error(json_response.text)

            # Predict SARIMA
            json_response = SessionManager.fast_api("predict_train_test_api", test_forecast_steps=len(X_test),
                                                    model_path=sarima_model_path, model_name="sarima",
                                                    is_log_transformed=is_log_transformed)

            if json_response.status_code == 200:
                y_train_prediction_sarima = pd.Series(json_response.json()["y_train_prediction"])
                y_test_prediction_sarima = pd.Series(json_response.json()["y_test_prediction"])

                st.markdown("### SARIMA Model:")
                # st.write(joblib.load(sarima_model_path).summary())
                # st.write(joblib.load(sarima_model_path).get_params())
                data_forecasting_model.interpret_slope(X_test, y_test_prediction_sarima)
                data_forecasting_model.print_performance_metrics(y_train, y_train_prediction_sarima, y_test,
                                                                 y_test_prediction_sarima)
                data_forecasting_model.plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarima,
                                                       column_mapping)
            else:
                st.error(json_response.text)

        else:
            st.error(json_response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")

async def handle_arimax_sarimax_training_and_predictions(train, test, column_mapping, product_name=None, is_log_transformed=None,
                                                         seasonality=None):
    try:
        features = train.columns.tolist()
        features.remove(column_mapping["quantity_sold_column"])
        features.remove(column_mapping["date_column"])
        exog_features = features

        date_column = column_mapping["date_column"]
        target = column_mapping["quantity_sold_column"]

        X_train_exog, X_test_exog, X_train, X_test, y_train, y_test = train[exog_features], test[exog_features], train[
            date_column], test[date_column], train[target], test[target]
        if seasonality is None:
            seasonality = SessionManager.get_state('selected_seasonality')
        if is_log_transformed is None:
            is_log_transformed = SessionManager.get_state("is_log_transformed")
        json_response = SessionManager.fast_api("fit_all_models_in_parallel_api",
                                                y_train=y_train.to_dict(),
                                                X_train=X_train_exog.to_dict(orient='records'),
                                                seasonality=seasonality,
                                                product_name=product_name,
                                                model_one="arimax", model_two="sarimax",
                                                column_mapping=column_mapping)

        if json_response.status_code == 200:

            arimax_model_path = json_response.json()["arimax"]["arimax_model_path"]
            sarimax_model_path = json_response.json()["sarimax"]["sarimax_model_path"]

            # Predict ARIMAX
            json_response = SessionManager.fast_api("predict_train_test_api",
                                                    model_path=arimax_model_path,
                                                    model_name="arimax",
                                                    test_forecast_steps=len(X_test_exog),
                                                    X_train=X_train_exog.to_dict(orient='records'),
                                                    X_test=X_test_exog.to_dict(orient='records'),
                                                    column_mapping=column_mapping,
                                                    is_log_transformed=is_log_transformed)

            if json_response.status_code == 200:
                y_train_prediction_arimax = pd.Series(json_response.json()["y_train_prediction"])
                y_test_prediction_arimax = pd.Series(json_response.json()["y_test_prediction"])

                st.markdown("### ARIMAX Model:")
                data_forecasting_model.interpret_slope(test[column_mapping["date_column"]], y_test_prediction_arimax)
                data_forecasting_model.print_performance_metrics(y_train, y_train_prediction_arimax, y_test,
                                                                 y_test_prediction_arimax)
                data_forecasting_model.plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_arimax,
                                                       column_mapping)
            else:
                st.error(json_response.text)

            # Predict SARIMAX
            json_response = SessionManager.fast_api("predict_train_test_api",
                                                    test_forecast_steps=len(X_test_exog),
                                                    model_path=sarimax_model_path,
                                                    model_name="sarimax",
                                                    X_train=X_train_exog.to_dict(orient='records'),
                                                    X_test=X_test_exog.to_dict(orient='records'),
                                                    column_mapping=column_mapping,
                                                    is_log_transformed=is_log_transformed)

            if json_response.status_code == 200:
                y_train_prediction_sarimax = pd.Series(json_response.json()["y_train_prediction"])
                y_test_prediction_sarimax = pd.Series(json_response.json()["y_test_prediction"])

                st.markdown("### SARIMAX Model:")
                data_forecasting_model.interpret_slope(test[column_mapping["date_column"]], y_test_prediction_sarimax)
                data_forecasting_model.print_performance_metrics(y_train, y_train_prediction_sarimax, y_test,
                                                                 y_test_prediction_sarimax)
                data_forecasting_model.plot_prediction(X_train, y_train, X_test, y_test, y_test_prediction_sarimax,
                                                       column_mapping)
            else:
                st.error(json_response.text)
        else:
            st.error(json_response.text)
    except Exception as e:
        st.error(f"An error occurred: {e}")
async def handle_fb_prophet_with_and_without_exog_training_and_predictions(train, test, train_with_exog, test_with_exog, column_mapping, product_name=None,
                                                                           is_log_transformed=None):
    # fit model parallel
    if is_log_transformed is None:
        is_log_transformed = SessionManager.get_state("is_log_transformed")
    json_response = SessionManager.fast_api("fit_all_models_in_parallel_api",
                                            model_one="fb_prophet_without_exog",
                                            model_two="fb_prophet_with_exog",
                                            train=train.to_dict(orient='records'),
                                            train_with_exog=train_with_exog.to_dict(orient='records'),
                                            column_mapping=column_mapping,
                                            product_name=product_name)
    if json_response.status_code == 200:
        fb_prophet_without_exog_path = json_response.json()["fb_prophet_without_exog"]["fb_prophet_model_path"]
        fb_prophet_with_exog_path = json_response.json()["fb_prophet_with_exog"]["fb_prophet_with_exog_model_path"]

        # predict without exog
        json_response = SessionManager.fast_api("predict_train_test_api",
                                                column_mapping=column_mapping,
                                                model_path=fb_prophet_without_exog_path,
                                                model_name="fb_prophet_without_exog",
                                                is_log_transformed=is_log_transformed,
                                                train=train.to_dict(orient='records'),
                                                test=test.to_dict(orient='records'))
        if json_response.status_code == 200:
            y_train_prediction_prophet_without_exog = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_prophet_without_exog = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### FB-Prophet Model Without Exogenous Features:")

            data_forecasting_model.interpret_slope(test[column_mapping["date_column"]],  y_test_prediction_prophet_without_exog)
            data_forecasting_model.print_performance_metrics(train[column_mapping['quantity_sold_column']],
                                                             y_train_prediction_prophet_without_exog,
                                                             test[column_mapping['quantity_sold_column']],
                                                             y_test_prediction_prophet_without_exog)
            data_forecasting_model.plot_prediction(pd.to_datetime(train[column_mapping["date_column"]]),
                                                   train[column_mapping['quantity_sold_column']],
                                                   pd.to_datetime(test[column_mapping["date_column"]]),
                                                   test[column_mapping['quantity_sold_column']],
                                                   y_test_prediction_prophet_without_exog,
                                                   column_mapping)


        else:
            st.error(json_response.text)

        json_response = SessionManager.fast_api("predict_train_test_api",
                                                column_mapping=column_mapping,
                                                model_path=fb_prophet_with_exog_path,
                                                model_name="fb_prophet_with_exog",
                                                is_log_transformed=is_log_transformed,
                                                train=train_with_exog.to_dict(orient='records'),
                                                test=test_with_exog.to_dict(orient='records'))
        if json_response.status_code == 200:
            y_train_prediction_prophet_with_exog = pd.Series(json_response.json()["y_train_prediction"])
            y_test_prediction_prophet_with_exog = pd.Series(json_response.json()["y_test_prediction"])

            st.markdown("### FB-Prophet Model With Exogenous Features:")

            data_forecasting_model.interpret_slope(test[column_mapping["date_column"]],
                                                   y_test_prediction_prophet_with_exog)
            data_forecasting_model.print_performance_metrics(train[column_mapping['quantity_sold_column']],
                                                             y_train_prediction_prophet_with_exog,
                                                             test[column_mapping['quantity_sold_column']],
                                                             y_test_prediction_prophet_with_exog)
            data_forecasting_model.plot_prediction(pd.to_datetime(train[column_mapping["date_column"]]),
                                                   train[column_mapping['quantity_sold_column']],
                                                   pd.to_datetime(test[column_mapping["date_column"]]),
                                                   test[column_mapping['quantity_sold_column']],
                                                   y_test_prediction_prophet_with_exog,
                                                   column_mapping)
        else:
            st.error(json_response.text)
    else:
        st.error(json_response.text)