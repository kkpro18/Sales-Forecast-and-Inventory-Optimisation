from flask import Flask, request, jsonify
from App.utils.data_preprocessing import format_dates, handle_missing_values, handle_outliers, encode_product_column
import pandas as pd

# "python -m flask_app/flask_app"
app = Flask(__name__)

## Pre Processing
@app.route("/format_dates_call", methods=["POST"])
def format_dates_call():
    try:
        data = pd.DataFrame(request.get_json())
        column_mapping = request.get_json()
        data = format_dates(data, column_mapping)
        processed_data = data.to_dict(orient="records")

        return jsonify({"processed_data": processed_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/handle_missing_values_call", methods=["POST"])
def handle_missing_values_call():
    try:
        data = pd.DataFrame(request.get_json())
        column_mapping = request.get_json()
        data = handle_missing_values(data, column_mapping)
        processed_data = data.to_dict(orient="records")

        return jsonify({"processed_data": processed_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/handle_outliers_call", methods=["POST"])
def handle_outliers_call():
    try:
        data = pd.DataFrame(request.get_json())
        column_mapping = request.get_json()
        data = handle_outliers(data, column_mapping)
        processed_data = data.to_dict(orient="records")

        return jsonify({"processed_data": processed_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/encode_product_column_call", methods=["POST"])
def encode_product_column_call():
    try:
        data = pd.DataFrame(request.get_json())
        column_mapping = request.get_json()
        data = encode_product_column(data, column_mapping)
        processed_data = data.to_dict(orient="records")

        return jsonify({"processed_data": processed_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Model Fitting

# Model Predictions

if __name__ == "__main__":
    app.run(debug=True)