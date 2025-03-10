import joblib
import json
from flask import Flask, request, jsonify
# from utils.data_preprocessing import format_dates, handle_missing_values, handle_outliers, encode_product_column
import pandas as pd
from data_preprocessing import format_dates, handle_missing_values, handle_outliers, encode_product_column
# from forecasting_sales import split_training_testing_data, fit_arima_model, print_performance_metrics, get_seasonality, fit_sarima_model, predict
from forecasting_sales import fit_arima_model, fit_sarima_model, predict
# "python App/utils/flask_app.py"
app = Flask(__name__)

## Pre Processing
@app.route("/format_dates_call", methods=["POST"])
def format_dates_call():
    try:
        data_received = request.get_json()
        data = pd.DataFrame(data_received["data"])
        column_mapping = data_received["column_mapping"]

        return jsonify(format_dates(data, column_mapping).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/handle_missing_values_call", methods=["POST"])
def handle_missing_values_call():
    try:
        data_received = request.get_json()
        data = pd.DataFrame(data_received["data"])
        column_mapping = data_received["column_mapping"]

        return jsonify(handle_missing_values(data, column_mapping).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/handle_outliers_call", methods=["POST"])
def handle_outliers_call():
    try:
        data_received = request.get_json()
        data = pd.DataFrame(data_received["data"])
        column_mapping = data_received["column_mapping"]

        return jsonify(handle_outliers(data, column_mapping).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/encode_product_column_call", methods=["POST"])
def encode_product_column_call():
    try:
        data_received = request.get_json()
        data = pd.DataFrame(data_received["data"])
        column_mapping = data_received["column_mapping"]

        return jsonify(encode_product_column(data, column_mapping).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Model Fitting
@app.route("/fit_and_store_arima_model_call", methods=["POST"])
def fit_and_store_arima_model_call():
    try:
        data_received = request.get_json()
        y_train = pd.DataFrame(data_received["y_train"].values())
        print(y_train)
        arima_model = fit_arima_model(y_train)

        joblib.dump(arima_model, 'models/arima.pkl')

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fit_and_store_sarima_model_call", methods=["POST"])
def fit_and_store_sarima_model_call():
    try:
        data_received = request.get_json()
        y_train = pd.DataFrame(data_received["y_train"].values())
        print(y_train)
        seasonality = data_received["seasonality"]

        sarima_model = fit_sarima_model(y_train, seasonality)
        joblib.dump(sarima_model, 'models/sarima.pkl')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Model Predictions
@app.route("/predict_train_test", methods=["POST"])
def predict_train_test():
    try:
        data_received = request.get_json()
        train_forecast_steps = data_received["train_forecast_steps"]
        test_forecast_steps = data_received["test_forecast_steps"]
        model_name = data_received["model_name"]

        y_train_prediction= predict(train_forecast_steps, model_name)
        y_test_prediction = predict(test_forecast_steps, model_name)

        return jsonify({f"y_train_prediction": y_train_prediction, f"y_test_prediction": y_test_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)