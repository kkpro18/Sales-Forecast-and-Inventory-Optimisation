import streamlit as st
from App.utils.session_manager import *

def split_training_testing_data(data, column_map):
    features = column_map.copy().pop("quantity_sold_column")
    target = column_map["quantity_sold_column"]
    # 70 : 30 split
    train_size = int(len(data) * 0.70)

    train = data[:train_size]
    test = data[train_size:]

    X_train, X_test = train[features.values()], test[features.values()]
    y_train, y_test = train[target], test[target]

    return X_train, X_test, y_train, y_test

def get_seasonality():
    pass

def fit_arima_model(y_train):
    pass

def fit_sarima_model(y_train, seasonality):
    pass

def print_performance_metrics(y_test_prediction, y_test):
    pass