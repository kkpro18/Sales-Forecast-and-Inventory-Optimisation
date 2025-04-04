import streamlit as st
from App.utils.session_manager import SessionManager
import pandas as pd

from App.utils.data_preprocessing import format_dates, convert_to_dict, concatenate_exogenous_data, scale_exogenous_data, add_lag_features

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
    # st.write(pd.DataFrame(json_response.json()['daily_product_sales']))

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
        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog = concatenate_exogenous_data(selected_region, train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, column_mapping)
    st.success(f"Successfully concatenating Exogenous Features")

    st.write("Scaling Exogenous Variables")
    selected_region = SessionManager.get_state("region")
    if SessionManager.get_state("region") != "N/A":
        train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_daily_product_sales_with_exog_scaled, test_daily_product_sales_with_exog_scaled = scale_exogenous_data(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog, column_mapping)
    st.success(f"Successfully scaled Exogenous Features")

    st.write("Adding Lag Features")
    train_daily_store_sales_with_exog_lagged, test_daily_store_sales_with_exog_lagged, train_daily_product_sales_with_exog_lagged, test_daily_product_sales_with_exog_lagged = add_lag_features(train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_daily_product_sales_with_exog_scaled, test_daily_product_sales_with_exog_scaled, column_mapping)
    st.success(f"Successfully scaled Exogenous Features")

    st.write(pd.DataFrame(train_daily_product_sales_with_exog_lagged))

    # st.dataframe(train_daily_product_sales_with_exog_scaled)


    SessionManager.set_state("train_daily_store_sales", train_daily_store_sales)
    SessionManager.set_state("test_daily_store_sales", test_daily_store_sales)
    SessionManager.set_state("train_daily_product_sales", train_daily_product_sales)
    SessionManager.set_state("test_daily_product_sales", test_daily_product_sales)

    SessionManager.set_state("train_daily_store_sales_with_exog", train_daily_store_sales_with_exog_lagged)
    SessionManager.set_state("test_daily_store_sales_with_exog", test_daily_store_sales_with_exog_lagged)
    SessionManager.set_state("train_daily_product_sales_with_exog", train_daily_product_sales_with_exog_lagged)
    SessionManager.set_state("test_daily_product_sales_with_exog", test_daily_product_sales_with_exog_lagged)


    st.markdown("## Preprocessed Data")

    st.markdown("### Train Data")

    st.write("train_daily_store_sales_with_exog_scaled_lagged", len(train_daily_store_sales_with_exog_lagged))

    st.write("train_daily_product_sales_with_exog_scaled_lagged", len(train_daily_product_sales_with_exog_lagged))

    st.markdown("### Test Data")

    st.write("test_daily_store_sales_with_exog_scaled_lagged", len(test_daily_store_sales_with_exog_lagged))

    st.write("test_product_sales_with_exog_scaled_lagged", len(test_daily_product_sales_with_exog_lagged))

    SessionManager.set_state("preprocess_data_complete",True)


    st.balloons()


    # time.sleep(3)
    # st.switch_page("pages/3_Visualise_Data.py")

