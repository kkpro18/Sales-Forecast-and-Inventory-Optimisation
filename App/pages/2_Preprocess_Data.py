import time
import streamlit as st
from App.utils.session_manager import SessionManager
import pandas as pd
import os

from App.utils.data_preprocessing import format_dates, fill_missing_date_range

st.set_page_config(
    page_title="Preprocess Data",
    page_icon="🧼",
    layout="wide",
)
st.markdown("# Preprocess Your Sales Data")
st.write(
    """Here you can Clean (pre-process) the data!""")


if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
else:
    data = SessionManager.get_state("data")
    column_mapping = SessionManager.get_state("column_mapping")
    data_as_dictionary = data.to_dict(orient='records')
    st.success(f"Successfully loaded data No. Rows: {len(data_as_dictionary)}")

    st.write("Applying Transformation to the Data")
    json_response = SessionManager.fast_api("transform_data_api", data=data_as_dictionary, column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Transformed Data No. Rows: {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Handling Outliers")
    json_response = SessionManager.fast_api("handle_outliers_api", data =json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Outliers No. Rows: {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Splitting into Train, Test")
    json_response = SessionManager.fast_api("train_test_split_api", data=json_response.json(),
                                            column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Split Dataset Train Size: {len(json_response.json()['train'])} Test Size: {len(json_response.json()['test'])}")
    else:
        st.error(json_response.text)

    st.write("Numerically Encoded Product ID")
    json_response = SessionManager.fast_api("encode_product_column_call", train=json_response.json()['train'], test=json_response.json()['test'], column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Encoded Product IDs Train Size: {len(json_response.json()['train'])} Test Size: {len(json_response.json()['test'])}")
    else:
        st.error(json_response.text)

    st.write("Handling Missing Values ")
    json_response = SessionManager.fast_api("handle_missing_values_api", train=json_response.json()['train'], test=json_response.json()['test'], column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Missing Values Train Size: {len(json_response.json()['train'])} Test Size: {len(json_response.json()['test'])}")
    else:
        st.error(json_response.text)

    st.write("Processing Dates in the Correct Format")
    train, test = pd.DataFrame(json_response.json()["train"]), pd.DataFrame(json_response.json()["test"])
    train, test = format_dates(train, test, column_mapping)
    st.success(f"Successfully Formatted Dates Train Size: {len(train)} Test Size: {len(test)}")

    train_daily_store_sales = train.groupby(column_mapping["date_column"], as_index=False).agg({column_mapping["quantity_sold_column"]: 'sum'})
    train_date_range = pd.date_range(start=train_daily_store_sales[column_mapping["date_column"]].min(), end=train_daily_store_sales[column_mapping["date_column"]].max(), freq='D')
    train_daily_dates_df = pd.DataFrame(train_date_range, columns=[column_mapping["date_column"]])
    train_daily_store_sales = pd.merge(train_daily_dates_df, train_daily_store_sales, on=column_mapping["date_column"], how='left')
    train_daily_store_sales[column_mapping["quantity_sold_column"]] = train_daily_store_sales[column_mapping["quantity_sold_column"]].fillna(0)
    train_daily_store_sales.reset_index(drop=True, inplace=True)

    test_daily_store_sales = test.groupby(column_mapping["date_column"], as_index=False).agg({column_mapping["quantity_sold_column"]: 'sum'})
    test_date_range = pd.date_range(start=test_daily_store_sales[column_mapping["date_column"]].min(), end=test_daily_store_sales[column_mapping["date_column"]].max(), freq='D')
    test_daily_dates_df = pd.DataFrame(test_date_range, columns=[column_mapping["date_column"]])
    test_daily_store_sales = pd.merge(test_daily_dates_df, test_daily_store_sales, on=column_mapping["date_column"], how='left')
    test_daily_store_sales[column_mapping["quantity_sold_column"]] = test_daily_store_sales[column_mapping["quantity_sold_column"]].fillna(0)
    test_daily_store_sales.reset_index(drop=True, inplace=True)

    train_product_sales = train.groupby([column_mapping["product_column"], column_mapping["date_column"]], as_index=False).agg(
        {
            column_mapping["price_column"]: 'mean',
            column_mapping["quantity_sold_column"]: 'sum'
        })
    train_product_sales = train_product_sales.groupby(column_mapping["product_column"]).apply(lambda group: fill_missing_date_range(group, column_mapping))
    train_product_sales.reset_index(drop=True, inplace=True)

    test_product_sales = test.groupby([column_mapping["product_column"], column_mapping["date_column"]], as_index=False).agg(
        {
            column_mapping["price_column"]: 'mean',
            column_mapping["quantity_sold_column"]: 'sum'
        })
    test_product_sales = test_product_sales.groupby(column_mapping["product_column"]).apply(lambda group: fill_missing_date_range(group, column_mapping))
    test_product_sales.reset_index(drop=True, inplace=True)

    SessionManager.set_state("preprocess_data_complete", True)

    st.markdown("## Preprocessed Data")
    SessionManager.set_state("train", train)
    SessionManager.set_state("test", test)

    SessionManager.set_state("train_daily_store_sales", train_daily_store_sales)
    SessionManager.set_state("train_daily_product_grouped_sales", train_product_sales)

    SessionManager.set_state("test_daily_store_sales", test_daily_store_sales)
    SessionManager.set_state("test_daily_product_grouped_sales", test_product_sales)

    st.markdown("### Train Data")
    st.subheader("train_daily_store_sales")
    st.dataframe(SessionManager.get_state("train_daily_store_sales"))

    st.subheader("train_daily_product_grouped_sales")
    st.dataframe(SessionManager.get_state("train_daily_product_grouped_sales"))

    st.markdown("### Test Data")
    st.subheader("test_daily_store_sales")
    st.dataframe(SessionManager.get_state("test_daily_store_sales"))

    st.subheader("test_daily_product_grouped_sales")
    st.dataframe(SessionManager.get_state("test_daily_product_grouped_sales"))

    st.balloons()

    # os.makedirs("store_sales", exist_ok=True)
    # daily_store_sales.to_csv("store_sales/daily_store_sales.csv", index=False)
    # st.success(f"daily_store_sales saved")

    # os.makedirs("product_sales", exist_ok=True)
    # for product, group in product_sales.groupby(column_mapping["product_column"]):
    #     file_name = f"product_sales/{product}_sales.csv"
    #     group.to_csv(file_name, index=False)
    #     st.success(f"Saved: {file_name}")


    # time.sleep(3)
    # st.switch_page("pages/3_Visualise_Data.py")

