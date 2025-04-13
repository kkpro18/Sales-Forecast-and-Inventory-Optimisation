import asyncio

import streamlit as st
import os
from App.utils.session_manager import SessionManager
from App.Controllers import forecasting_controller

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

    train_daily_store_sales, test_daily_store_sales, train_daily_store_sales_with_exog, test_daily_store_sales_with_exog = forecasting_controller.handle_store_sales_data(column_mapping)

    forecasting_controller.handle_seasonality_input()

    if SessionManager.is_there("selected_seasonality"):
        if not os.path.isdir("models"):
            os.mkdir("models")

        st.markdown("### Store Wide Sales Forecasting")

        asyncio.run(
            forecasting_controller.handle_arima_sarima_training_and_predictions(
                train_daily_store_sales,
                test_daily_store_sales,
                column_mapping,
            )
        )
        asyncio.run(
            forecasting_controller.handle_arimax_sarimax_training_and_predictions(
                train_daily_store_sales_with_exog,
                test_daily_store_sales_with_exog,
                column_mapping,
            )
        )
        asyncio.run(
            forecasting_controller.handle_fb_prophet_with_and_without_exog_training_and_predictions(
                train_daily_store_sales, test_daily_store_sales, train_daily_store_sales_with_exog, test_daily_store_sales_with_exog, column_mapping,
            )
        )


        st.markdown("### Individual Product Sales Forecasting")
        # initialise product index
        if not SessionManager.is_there("product_index"):
            SessionManager.set_state("product_index", 0)

        train_product_sales_grouped, train_product_names, test_product_sales_grouped, test_product_names, train_product_sales_with_exog_grouped, train_product_with_exog_names, test_product_sales_with_exog_grouped, test_product_with_exog_names = forecasting_controller.handle_product_sales_data(column_mapping)

        if not set(test_product_names).issubset(train_product_names):
            st.warning("Test set contains products not in train set, please check your data.")

        left_column, right_column = st.columns(2)
        previous_button = left_column.button("Previous", disabled=SessionManager.get_state("product_index") == 0)
        next_button = right_column.button("Next", disabled=SessionManager.get_state("product_index") >= len(test_product_names) - 1)

        try:
            if previous_button:
                if SessionManager.get_state("product_index") > 0:
                    SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)

                product_name = test_product_names[SessionManager.get_state('product_index')]

                st.write(f"### {product_name} Train Samples: {len(train_product_sales_grouped.get_group(product_name))}, Test Samples: {len(test_product_sales_grouped.get_group(product_name))}")

                train_product_data = train_product_sales_grouped.get_group(product_name)
                train_product_data = train_product_data.drop(columns=column_mapping['product_column'])

                test_product_data = test_product_sales_grouped.get_group(product_name)
                test_product_data = test_product_data.drop(columns=column_mapping['product_column'])

                train_product_data_with_exog = train_product_sales_with_exog_grouped.get_group(product_name)
                train_product_data_with_exog = train_product_data_with_exog.drop(columns=column_mapping['product_column'])

                test_product_data_with_exog = test_product_sales_with_exog_grouped.get_group(product_name)
                test_product_data_with_exog = test_product_data_with_exog.drop(columns=column_mapping['product_column'])

                if len(train_product_data) <= 20:
                    st.warning("Not enough data for this product to train the model.")
                else:
                    asyncio.run(
                        forecasting_controller.handle_arima_sarima_training_and_predictions(
                            train_product_data, test_product_data,
                            column_mapping,
                            product_name=product_name
                        )
                    )
                    asyncio.run(
                        forecasting_controller.handle_arimax_sarimax_training_and_predictions(
                            train_product_data_with_exog, test_product_data_with_exog,
                            column_mapping,
                            product_name=product_name
                        )
                    )
                    asyncio.run(
                        forecasting_controller.handle_fb_prophet_with_and_without_exog_training_and_predictions(
                            train_product_data, test_product_data,
                            train_product_data_with_exog, test_product_data_with_exog,
                            column_mapping,
                            product_name=product_name
                        )
                    )
            elif next_button:
                SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
                product_name = test_product_names[SessionManager.get_state('product_index')]
                st.write(
                    f"### {product_name} Train Samples: {len(train_product_sales_grouped.get_group(product_name))}, Test Samples: {len(test_product_sales_grouped.get_group(product_name))}")

                train_product_data = train_product_sales_grouped.get_group(product_name)
                train_product_data = train_product_data.drop(columns=column_mapping["product_column"])

                test_product_data = test_product_sales_grouped.get_group(product_name)
                test_product_data = test_product_data.drop(columns=column_mapping["product_column"])

                train_product_data_with_exog = train_product_sales_with_exog_grouped.get_group(product_name)
                train_product_data_with_exog = train_product_data_with_exog.drop(columns=column_mapping["product_column"])

                test_product_data_with_exog = test_product_sales_with_exog_grouped.get_group(product_name)
                test_product_data_with_exog = test_product_data_with_exog.drop(columns=column_mapping["product_column"])

                if len(train_product_data) < 20:
                    st.warning("Not enough data for this product to train the model.")
                else:

                    asyncio.run(
                        forecasting_controller.handle_arima_sarima_training_and_predictions(
                            train_product_data, test_product_data,
                            column_mapping,
                            product_name=product_name
                        )
                    )
                    asyncio.run(
                        forecasting_controller.handle_arimax_sarimax_training_and_predictions(
                            train_product_data_with_exog.reset_index(drop=True), test_product_data_with_exog.reset_index(drop=True),
                            column_mapping,
                            product_name=product_name
                        )
                    )
                    asyncio.run(
                        forecasting_controller.handle_fb_prophet_with_and_without_exog_training_and_predictions(
                            train_product_data, test_product_data,
                            train_product_data_with_exog, test_product_data_with_exog,
                            column_mapping,
                            product_name=product_name
                        )
                    )
        except Exception as e:
            st.error(e)


        st.page_link("pages/5_Calculate_Optimal_Inventory_Policy.py", label="👈 Next Stage: Calculate Optimal inventory policy",
                     icon="⚙️")
