import streamlit as st
from App.utils.session_manager import SessionManager
from App.Controllers import data_preprocessing_controller

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
    data_as_dictionary = data_preprocessing_controller.handle_dictionary_conversion(data)
    st.success(f"Successfully loaded data No. Rows: {len(data_as_dictionary)}")
    # check if data as dictionary contains anything invalid for json
    # st.write(data_as_dictionary)


    st.write("Applying Transformation to the Data")
    response = data_preprocessing_controller.handle_data_transformation(data_as_dictionary, column_mapping)
    if response.status_code == 200:
        st.success(f"Successfully Transformed Data No. Rows: {len(response.json()['data'])}")
        SessionManager.set_state("is_log_transformed", response.json()['is_log_transformed'])
    else:
        st.error(response.text)

    st.write("Handling Outliers")
    response = data_preprocessing_controller.handle_outliers(data=response.json()['data'], column_mapping=column_mapping)
    if response.status_code == 200:
        st.success(f"Successfully Handled Outliers No. Rows: {len(response.json())}")
    else:
        st.error(response.text)

    st.write("Fixing Dates and Splitting into Product Sales and Overall Store Sales")
    response = data_preprocessing_controller.handle_dates_and_split_product_and_overall_sales(data=response.json(), column_mapping=column_mapping)
    if response.status_code == 200:
        st.success(f"Successfully Fixed Dates No. Rows: Daily Sales Size: {len(response.json()['daily_store_sales'])}, Product Sales Size: {len(response.json()['daily_product_sales'])}")
    else:
        st.error(response.text)

    st.write("Splitting into Train, Test")
    response = data_preprocessing_controller.handle_train_test_split(daily_store_sales=response.json()['daily_store_sales'], daily_product_sales=response.json()['daily_product_sales'], column_mapping=column_mapping)
    if response.status_code == 200:
        st.success(f"Successfully Split Dataset  {len(response.json()['train_daily_store_sales'])}, Test Daily Sales Size: {len(response.json()['test_daily_store_sales'])}, Train Product Daily Sales Size: {len(response.json()['train_daily_product_sales'])}, Test Product Daily Sales Size: {len(response.json()['test_daily_product_sales'])}")
    else:
        st.error(response.text)

    st.write("Handling Missing Values ")
    response = data_preprocessing_controller.handle_missing_values(
                                       train_daily_store_sales=response.json()["train_daily_store_sales"],
                                       test_daily_store_sales=response.json()["test_daily_store_sales"],
                                       train_daily_product_sales=response.json()["train_daily_product_sales"],
                                       test_daily_product_sales=response.json()["test_daily_product_sales"],
                                       column_mapping=column_mapping)
    if response.status_code == 200:
        st.success(f"Successfully Handled Missing Values Train Size:{len(response.json()['train_daily_store_sales'])}, Test Daily Sales Size: {len(response.json()['test_daily_store_sales'])}, Train Product Daily Sales Size: {len(response.json()['train_daily_product_sales'])}, Test Product Daily Sales Size: {len(response.json()['test_daily_product_sales'])}")
    else:
        st.error(response.text)

    st.write("Processing Dates in the Correct Format")
    train_daily_store_sales, test_daily_store_sales = data_preprocessing_controller.handle_date_formatting(response.json()["train_daily_store_sales"], response.json()["test_daily_store_sales"], column_mapping)

    train_daily_product_sales, test_daily_product_sales = data_preprocessing_controller.handle_date_formatting(response.json()["train_daily_product_sales"], response.json()["test_daily_product_sales"], column_mapping)
    st.success(f"Successfully Formatted Dates Train Size: {len(train_daily_store_sales)}, Test Daily Sales Size: {len(test_daily_store_sales)}, Train Product Daily Sales Size: {len(train_daily_product_sales)}, Test Product Daily Sales Size: {len(test_daily_product_sales)}")

    if SessionManager.get_state("region") != "N/A":
        st.write("Concatenating Exogenous Variables")
        selected_region = SessionManager.get_state("region")
        train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog = data_preprocessing_controller.handle_inclusion_of_exogenous_variables(selected_region, train_daily_store_sales, test_daily_store_sales, train_daily_product_sales, test_daily_product_sales, column_mapping)
        st.success(f"Successfully concatenating Exogenous Features")


        st.write("Scaling Exogenous Variables")
        selected_region = SessionManager.get_state("region")
        if SessionManager.get_state("region") != "N/A":
            train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_daily_product_sales_with_exog_scaled, test_daily_product_sales_with_exog_scaled = data_preprocessing_controller.handle_exogenous_scaling(train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, train_daily_product_sales_with_exog, test_daily_product_sales_with_exog, column_mapping)
        st.success(f"Successfully scaled Exogenous Features")

    st.write("Adding Lag Features")
    train_daily_store_sales_with_exog_lagged, test_daily_store_sales_with_exog_lagged, train_daily_product_sales_with_exog_lagged, test_daily_product_sales_with_exog_lagged = data_preprocessing_controller.handle_lag_features(train_daily_store_sales_with_exog_scaled, test_daily_store_sales_with_exog_scaled, train_daily_product_sales_with_exog_scaled, test_daily_product_sales_with_exog_scaled, column_mapping)
    st.success(f"Successfully scaled Exogenous Features")

    # forst omotoa;ose

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

    st.write("train_daily_store_sales", train_daily_store_sales.shape)
    st.write("train_daily_product_sales", train_daily_product_sales.shape)

    st.write("train_daily_store_sales_with_exog_scaled_lagged", train_daily_store_sales_with_exog_lagged.shape)
    st.write("train_daily_product_sales_with_exog_scaled_lagged", train_daily_product_sales_with_exog_lagged.shape)

    st.markdown("### Test Data")

    st.write("test_daily_store_sales", test_daily_store_sales.shape)
    st.write("test_daily_product_sales", test_daily_product_sales.shape)

    st.write("test_daily_store_sales_with_exog_scaled_lagged", test_daily_store_sales_with_exog_lagged.shape)
    st.write("test_product_sales_with_exog_scaled_lagged", test_daily_product_sales_with_exog_lagged.shape)

    SessionManager.set_state("preprocess_data_complete",True)


    st.balloons()


    st.page_link("pages/3_Visualise_Data.py", label="👈 Next Visualise the Data", icon="🧼")


