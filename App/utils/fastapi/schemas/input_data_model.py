from pydantic import BaseModel, conint
from typing import Optional, Dict, List, Any


class InputData(BaseModel):
    column_mapping: Optional[Dict[str, str]] = None
    data: Optional[List[Dict[str, Any]]] = None
    train: Optional[List[Dict[str, Any]]] = None
    test: Optional[List[Dict[str, Any]]] = None
    train_with_exog: Optional[List[Dict[str, Any]]] = None
    test_with_exog: Optional[List[Dict[str, Any]]] = None
    daily_store_sales: Optional[List[Dict[str, Any]]] = None
    daily_product_sales: Optional[List[Dict[str, Any]]] = None
    train_daily_store_sales: Optional[List[Dict[str, Any]]] = None
    test_daily_store_sales: Optional[List[Dict[str, Any]]] = None
    train_daily_product_sales: Optional[List[Dict[str, Any]]] = None
    test_daily_product_sales: Optional[List[Dict[str, Any]]] = None
    train_data: Optional[List[Dict[str, Any]]] = None
    test_data: Optional[List[Dict[str, Any]]] = None
    X_train: Optional[List[Dict[str, Any]]] = None
    X_test: Optional[List[Dict[str, Any]]] = None
    y_train: Optional[Dict[int, Any]] = None
    y_test: Optional[Dict[int, Any]] = None
    seasonality: Optional[conint(gt=0)] = None
    test_forecast_steps: Optional[conint(gt=0)] = None
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    product_name: Optional[str] = None
    model_one: Optional[str] = None
    model_two: Optional[str] = None
    is_log_transformed: Optional[bool] = None
