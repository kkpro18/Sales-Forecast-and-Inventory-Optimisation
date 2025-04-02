import streamlit as st
from App.utils.session_manager import SessionManager
import pandas as pd

from App.utils.data_preprocessing import format_dates, convert_to_dict, concatenate_exogenous_data

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
    data_as_dictionary = convert_to_dict(data)
    st.success(f"Successfully loaded data No. Rows: {len(data_as_dictionary)}")
    # check if data as dictionary contains anything invalid for json
    # st.write(data_as_dictionary)


    st.write("Applying Transformation to the Data")
    json_response = SessionManager.fast_api("transform_data_api",
                                            data=data_as_dictionary,
                                            column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Transformed Data No. Rows: {len(json_response.json())}")
        SessionManager.set_state("is_log_transformed", True)
    else:
        st.error(json_response.text)

    st.write("Handling Outliers")
    json_response = SessionManager.fast_api("handle_outliers_api",
                                            data = json_response.json(),
                                            column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Outliers No. Rows: {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Fixing Dates and Splitting into Product Sales and Overall Store Sales")
    json_response = SessionManager.fast_api("fix_dates_and_split_into_product_sales_and_daily_sales_api",
                                            data=json_response.json(),
                                            column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Fixed Dates No. Rows: Daily Sales Size: {len(json_response.json()['daily_store_sales'])}, Product Sales Size: {len(json_response.json()['daily_product_sales'])}")
    else:
        st.error(json_response.text)

    st.write("Splitting into Train, Test")
    json_response = SessionManager.fast_api("train_test_split_api",
                                            daily_store_sales=json_response.json()['daily_store_sales'],
                                            daily_product_sales=json_response.json()['daily_product_sales'],
                                            column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Split Dataset  {len(json_response.json()['train_daily_store_sales'])}, Test Daily Sales Size: {len(json_response.json()['test_daily_store_sales'])}, Train Product Daily Sales Size: {len(json_response.json()['train_daily_product_sales'])}, Test Product Daily Sales Size: {len(json_response.json()['test_daily_product_sales'])}")
    else:
        st.error(json_response.text)

    st.write("Handling Missing Values ")
    json_response = SessionManager.fast_api("handle_missing_values_api",
                                            train_daily_store_sales=json_response.json()["train_daily_store_sales"],
                                            test_daily_store_sales=json_response.json()["test_daily_store_sales"],
                                            train_daily_product_sales=json_response.json()["train_daily_product_sales"],
                                            test_daily_product_sales=json_response.json()["test_daily_product_sales"],
                                            column_mapping=column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Missing Values Train Size:{len(json_response.json()['train_daily_store_sales'])}, Test Daily Sales Size: {len(json_response.json()['test_daily_store_sales'])}, Train Product Daily Sales Size: {len(json_response.json()['train_daily_product_sales'])}, Test Product Daily Sales Size: {len(json_response.json()['test_daily_product_sales'])}")
    else:
        st.error(json_response.text)

    st.write("Processing Dates in the Correct Format")
    train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales = pd.DataFrame(json_response.json()["train_daily_store_sales"]), pd.DataFrame(json_response.json()["test_daily_store_sales"]), pd.DataFrame(json_response.json()["train_daily_product_sales"]), pd.DataFrame(json_response.json()["test_daily_product_sales"])
    train_daily_store_sales, test_daily_store_sales = format_dates(train_daily_store_sales, test_daily_store_sales, column_mapping)
    train_daily_product_sales, test_daily_product_sales = format_dates(train_daily_product_sales, test_daily_product_sales, column_mapping)
    st.success(f"Successfully Formatted Dates Train Size: {len(train_daily_store_sales)}, Test Daily Sales Size: {len(test_daily_store_sales)}, Train Product Daily Sales Size: {len(train_daily_product_sales)}, Test Product Daily Sales Size: {len(test_daily_product_sales)}")

    st.write("Concatenating Exogenous Variables")
    selected_region = SessionManager.get_state("region")
    if SessionManager.get_state("region") != "N/A":
        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_product_sales_with_exog = concatenate_exogenous_data(selected_region, train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, column_mapping)
    st.success(f"Successfully concatenating Exogenous Features")

    SessionManager.set_state("train_daily_store_sales", train_daily_store_sales)
    SessionManager.set_state("test_daily_store_sales", test_daily_store_sales)
    SessionManager.set_state("train_daily_product_sales", train_daily_product_sales)
    SessionManager.set_state("test_daily_product_sales", test_daily_product_sales)

    # scale exog

    # add lag features

    st.write("train_daily_store_sales_with_exog", len(train_daily_store_sales_with_exog))

    st.write("test_daily_store_sales_with_exog", len(test_daily_store_sales_with_exog))

    st.write("train_daily_product_sales_with_exog", len(test_product_sales_with_exog))

    st.write("test_product_sales_with_exog", len(test_product_sales_with_exog))

    SessionManager.set_state("preprocess_data_complete",True)

    st.markdown("## Preprocessed Data")

    SessionManager.set_state("train_daily_sales", train_daily_store_sales)
    SessionManager.set_state("train_daily_store_sales_with_exog", train_daily_store_sales_with_exog)
    SessionManager.set_state("train_daily_product_sales", train_daily_product_sales)
    SessionManager.set_state("train_daily_product_sales_with_exog", train_daily_product_sales_with_exog)

    SessionManager.set_state("test_daily_sales", test_daily_store_sales)
    SessionManager.set_state("test_daily_store_sales_with_exog", test_daily_store_sales_with_exog)
    SessionManager.set_state("test_daily_product_sales", test_daily_product_sales)
    SessionManager.set_state("test_product_sales_with_exog", test_product_sales_with_exog)

    st.markdown("### Train Data")
    st.subheader("train_daily_sales")
    st.dataframe(SessionManager.get_state("train_daily_sales"))

    st.subheader("train_daily_sales_with_exog")
    st.dataframe(SessionManager.get_state("train_daily_sales_with_exog"))

    st.subheader("train_daily_product_sales")
    st.dataframe(SessionManager.get_state("train_daily_product_sales"))

    st.subheader("train_daily_product_sales_with_exog")
    st.dataframe(SessionManager.get_state("train_daily_product_sales_with_exog"))

    st.markdown("### Test Data")
    st.subheader("test_daily_sales")
    st.dataframe(SessionManager.get_state("test_daily_sales"))

    st.subheader("test_daily_sales_with_exog")
    st.dataframe(SessionManager.get_state("test_daily_sales_with_exog"))

    st.subheader("test_daily_product_sales")
    st.dataframe(SessionManager.get_state("test_daily_product_sales"))

    st.subheader("test_product_sales_with_exog")
    st.dataframe(SessionManager.get_state("test_product_sales_with_exog"))


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

