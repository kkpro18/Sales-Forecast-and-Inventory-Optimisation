import asyncio

import streamlit as st
import os
from App.utils.forecasting_sales import get_seasonality, predict_sales_fb_prophet, predict_sales_arima_sarima, predict_sales_arimax_sarimax
from App.utils.session_manager import SessionManager

st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
    layout="wide",
)

st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

if not SessionManager.is_there("data") or not SessionManager.is_there("column_mapping"):
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")
    st.page_link("pages/1_Upload_Data.py", label="👈 Upload The Dataset", icon="📁")
elif not SessionManager.get_state("preprocess_data_complete"):
    st.page_link("pages/2_Preprocess_Data.py", label="👈 Pre-process The Dataset", icon="📁")
else:
    column_mapping = SessionManager.get_state("column_mapping")
    date_column = column_mapping["date_column"]

    train_daily_store_sales = SessionManager.get_state("train_daily_store_sales")
    train_daily_store_sales[date_column] = train_daily_store_sales[date_column].astype(str)

    test_daily_store_sales = SessionManager.get_state("test_daily_store_sales")
    test_daily_store_sales[date_column] = test_daily_store_sales[date_column].astype(str)

    train_daily_store_sales_with_exog = SessionManager.get_state("train_daily_store_sales_with_exog")
    train_daily_store_sales_with_exog[date_column] = train_daily_store_sales_with_exog[date_column].astype(str)

    test_daily_store_sales_with_exog = SessionManager.get_state("test_daily_store_sales_with_exog")
    test_daily_store_sales_with_exog[date_column] = test_daily_store_sales_with_exog[date_column].astype(str)

    get_seasonality()
    st.write(SessionManager.get_state("is_log_transformed"))
    if SessionManager.is_there("selected_seasonality"):
        st.markdown("### Store Wide Sales Forecasting")

        if not os.path.isdir("models"):
            os.mkdir("models")

        asyncio.run(
            predict_sales_arima_sarima(
                train_daily_store_sales,
                test_daily_store_sales,
                column_mapping,
            )
        )
        asyncio.run(
            predict_sales_arimax_sarimax(
                train_daily_store_sales_with_exog,
                test_daily_store_sales_with_exog,
                column_mapping,
            )
        )
        asyncio.run(
            predict_sales_fb_prophet(
                train_daily_store_sales, test_daily_store_sales, train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, column_mapping,
            )
        )



        # st.markdown("### Individual Product Sales Forecasting")
        #
        #
        # train_product_sales_raw = SessionManager.get_state("train_daily_product_sales")
        # train_product_sales_grouped = train_product_sales_raw.groupby(column_mapping["product_column"]) # groups by product column
        # train_product_names = list(train_product_sales_grouped.groups.keys()) # gets each product names
        #
        # test_product_sales_raw = SessionManager.get_state("test_daily_product_sales")
        # test_product_sales_grouped = test_product_sales_raw.groupby(column_mapping["product_column"])  # groups by product column
        # test_product_names = list(test_product_sales_grouped.groups.keys())  # gets each product names
        #
        # train_product_sales_raw_with_exog = SessionManager.get_state("train_daily_product_sales_with_exog")
        # train_product_sales_with_exog_grouped = train_product_sales_raw_with_exog.groupby(column_mapping["product_column"])  # groups by product column
        # train_product_with_exog_names = list(train_product_sales_with_exog_grouped.groups.keys())  # gets each product names
        #
        # test_product_sales_with_exog_raw = SessionManager.get_state("test_daily_product_sales_with_exog")
        # test_product_sales_with_exog_grouped = test_product_sales_with_exog_raw.groupby(column_mapping["product_column"])  # groups by product column
        # test_product_with_exog_names = list(test_product_sales_with_exog_grouped.groups.keys())  # gets each product names
        #
        # if not set(test_product_names).issubset(train_product_names):
        #     st.warning("Test set contains products not in train set, please check your data.")
        #
        # left_column, right_column = st.columns(2)
        # previous_button = left_column.button("Previous", disabled=SessionManager.get_state("product_index") == 0)
        # next_button = right_column.button("Next", disabled=SessionManager.get_state("product_index") >= len(test_product_names) - 1)
        #
        # if previous_button:
        #     if SessionManager.get_state("product_index") > 0:
        #         SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)
        #
        #     product_name = test_product_names[SessionManager.get_state('product_index')]
        #
        #     st.write(f"### {product_name} Train Samples: {len(train_product_sales_grouped.get_group(product_name))}, Test Samples: {len(test_product_sales_grouped.get_group(product_name))}")
        #
        #     train_product_data = train_product_sales_grouped.get_group(product_name)
        #     test_product_data = test_product_sales_grouped.get_group(product_name)
        #     train_product_data_with_exog = train_product_sales_grouped.get_group(product_name)
        #     test_product_data_with_exog = test_product_sales_grouped.get_group(product_name)
        #
        #
        #     if len(train_product_data) <= 20:
        #         st.warning("Not enough data for this product to train the model.")
        #     else:
        #         asyncio.run(
        #             predict_sales_arima_sarima(
        #                 train_product_data, test_product_data,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )
        #         asyncio.run(
        #             predict_sales_arimax_sarimax(
        #                 train_product_data_with_exog, test_product_data_with_exog,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )
        #         asyncio.run(
        #             predict_sales_fb_prophet(
        #                 train_product_data_with_exog, test_product_data_with_exog,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )
        #
        #
        # elif next_button:
        #     SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
        #     product_name = test_product_names[SessionManager.get_state('product_index')]
        #     st.write(
        #         f"### {product_name} Train Samples: {len(train_product_sales_grouped.get_group(product_name))}, Test Samples: {len(test_product_sales_grouped.get_group(product_name))}")
        #
        #     train_product_data = train_product_sales_grouped.get_group(product_name)
        #     test_product_data = test_product_sales_grouped.get_group(product_name)
        #     train_product_data_exog = train_product_sales_with_exog_grouped.get_group(product_name)
        #     test_product_data_exog = test_product_sales_with_exog_grouped.get_group(product_name)
        #
        #     if len(train_product_data) < 20:
        #         st.warning("Not enough data for this product to train the model.")
        #     else:
        #         asyncio.run(
        #             predict_sales_arima_sarima(
        #                 train_product_data, test_product_data,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )
        #         asyncio.run(
        #             predict_sales_arimax_sarimax(
        #                 test_product_data_exog, test_product_data_exog,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )
        #         asyncio.run( # exog data is passed and non exog data is extracted and trained
        #             predict_sales_fb_prophet(
        #                 test_product_data_exog, test_product_data_exog,
        #                 column_mapping,
        #                 product_name=product_name
        #             )
        #         )

        st.page_link("pages/5_Inventory_Policy_Simulator.py", label="👈 Next Stage: Simulate your inventory policy",
                     icon="⚙️")
