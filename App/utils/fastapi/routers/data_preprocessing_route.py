from fastapi import APIRouter, HTTPException
from App.utils.fastapi.schemas.input_data_model import InputData

import pandas as pd
from App.Models import data_preprocessing_model

router = APIRouter()

## Pre Processing

@router.post("/transform_data_api")
def transform_data_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping
        data, is_log_transformed = data_preprocessing_model.transform_data(data, column_mapping)
        return {"data": data_preprocessing_model.convert_to_dict(data), "is_log_transformed": is_log_transformed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/handle_outliers_api")
def handle_outliers_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        return data_preprocessing_model.convert_to_dict(data_preprocessing_model.handle_outliers(data, column_mapping))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fix_dates_and_split_into_product_sales_and_daily_sales_api")
def fix_dates_and_split_into_product_sales_and_daily_sales_api(received_data: InputData):
    try:
        data = pd.DataFrame(received_data.data)
        column_mapping = received_data.column_mapping

        daily_store_sales, daily_product_sales = data_preprocessing_model.fix_dates_and_split_into_product_sales_and_daily_sales(data, column_mapping)
        # needs to be filtered i.e NaN to None as JSON does not allow this
        daily_store_sales, daily_product_sales = (data_preprocessing_model.convert_to_dict(daily_store_sales), data_preprocessing_model.convert_to_dict(daily_product_sales))
        return {
            "daily_store_sales": daily_store_sales,
            "daily_product_sales": daily_product_sales,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train_test_split_api")
def train_test_split_api(received_data: InputData):
    try:
        daily_store_sales = pd.DataFrame(received_data.daily_store_sales)
        daily_product_sales = pd.DataFrame(received_data.daily_product_sales)

        column_mapping = received_data.column_mapping

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_model.split_training_testing_data(daily_store_sales, column_mapping)
        train_daily_product_sales, test_daily_product_sales = data_preprocessing_model.split_training_testing_data(daily_product_sales, column_mapping)

        return {
            "train_daily_store_sales": data_preprocessing_model.convert_to_dict(train_daily_store_sales),
            "test_daily_store_sales": data_preprocessing_model.convert_to_dict(test_daily_store_sales),
            "train_daily_product_sales": data_preprocessing_model.convert_to_dict(train_daily_product_sales),
            "test_daily_product_sales": data_preprocessing_model.convert_to_dict(test_daily_product_sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/handle_missing_values_api")
def handle_missing_values_api(received_data: InputData):
    try:

        train_daily_store_sales = pd.DataFrame(received_data.train_daily_store_sales)
        test_daily_store_sales = pd.DataFrame(received_data.test_daily_store_sales)
        train_daily_product_sales = pd.DataFrame(received_data.train_daily_product_sales)
        test_daily_product_sales = pd.DataFrame(received_data.test_daily_product_sales)
        column_mapping = received_data.column_mapping

        train_daily_store_sales, test_daily_store_sales = data_preprocessing_model.handle_missing_values(train_daily_store_sales,
                                                                                test_daily_store_sales, column_mapping)
        train_daily_product_sales, test_daily_product_sales = data_preprocessing_model.handle_missing_values(train_daily_product_sales,
                                                                                    test_daily_product_sales,
                                                                                    column_mapping)

        return {
            "train_daily_store_sales": data_preprocessing_model.convert_to_dict(train_daily_store_sales),
            "test_daily_store_sales": data_preprocessing_model.convert_to_dict(test_daily_store_sales),
            "train_daily_product_sales": data_preprocessing_model.convert_to_dict(train_daily_product_sales),
            "test_daily_product_sales": data_preprocessing_model.convert_to_dict(test_daily_product_sales),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
