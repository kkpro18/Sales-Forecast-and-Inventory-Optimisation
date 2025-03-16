import time
import streamlit as st
from App.utils.session_manager import SessionManager
import pandas as pd

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

    st.write("Processing Dates in the Correct Format")
    data_as_dictionary = data.to_dict(orient='records')


    json_response = SessionManager.fast_api("format_dates_api", data = data_as_dictionary, column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Formatted Dates: No. Rows = {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Handling Missing Values ")
    json_response = SessionManager.fast_api("handle_missing_values_api", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Missing Values: No. Rows = {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Handling Outliers")
    json_response = SessionManager.fast_api("handle_outliers_api", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Handled Outliers: No. Rows = {len(json_response.json())}")
    else:
        st.error(json_response.text)

    st.write("Numerically Encoding Product ID (Unique Identifier)")
    json_response = SessionManager.fast_api("encode_product_column_api", data = json_response.json(), column_mapping = column_mapping)
    if json_response.status_code == 200:
        st.success(f"Successfully Encoded Product IDs: No. Rows = {len(json_response.json())}")
    else:
        st.error(json_response.text)

    data = pd.DataFrame(json_response.json())
    data[column_mapping["date_column"]] = pd.to_datetime(data[column_mapping["date_column"]], errors="coerce")
    data[column_mapping["date_column"]] = data[column_mapping["date_column"]].dt.tz_localize(None)

    st.subheader("Preprocessed Data: ")
    daily_store_sales = data.groupby(column_mapping["date_column"], as_index=False).agg({column_mapping["quantity_sold_column"]: 'sum'})
    st.dataframe(daily_store_sales)

    # product_group_daily_sales = data.groupby([column_mapping["date_column"], column_mapping["product_column"]], as_index=False).agg({column_mapping['quantity_sold_column']: 'sum'})
    product_sales = data.groupby([column_mapping["product_column"], column_mapping["date_column"]], as_index=False).agg(
        {
            column_mapping["quantity_sold_column"]: 'sum'
        })

    # for product, group in product_sales.groupby(column_mapping["product_column"]):
    #     st.header(f"Product: {product}")
    #     st.dataframe(group)



    SessionManager.set_state("preprocess_data_complete", True)
    # SessionManager.set_state("data", data)
    SessionManager.set_state("daily_store_sales", daily_store_sales)
    SessionManager.set_state("daily_product_grouped_sales", product_sales)

    st.subheader("Preprocessed Data: ")
    st.dataframe(SessionManager.get_state("data"))
    st.balloons()

    daily_store_sales_csv = daily_store_sales.to_csv(index=False)
    st.download_button(
        label="Download the daily_store_sales Data as a CSV File",
        data=daily_store_sales_csv,
        file_name="preprocessed_data.csv",
    )

    daily_product_grouped_sales_csv = data.to_csv(index=False)
    st.download_button(
        label="Download the daily_product_grouped_sales Data as a CSV File",
        data=daily_product_grouped_sales_csv,
        file_name="preprocessed_data.csv",
    )
    # time.sleep(3)
    # st.switch_page("pages/3_Visualise_Data.py")

