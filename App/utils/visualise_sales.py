import plotly.graph_objects as go
import streamlit as st
from App.utils.session_manager import *


def plot_individual_product(all_products_grouped, unique_products, date_column, quantity_sold_column):
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=all_products_grouped.get_group(unique_products[st.session_state.product_index])[date_column],
        y=all_products_grouped.get_group(unique_products[st.session_state.product_index])[quantity_sold_column])
    )
    figure.update_layout(
        title_text=f"Current Sales Data for Product {unique_products[st.session_state.product_index]}",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        xaxis_title=date_column,
        yaxis_title=quantity_sold_column,
    )
    st.plotly_chart(figure)

def visualise_storewide_sales(data, column_map):
    date_column = column_map["date_column"]
    quantity_sold_column = column_map["quantity_sold_column"]

    store_overall_sales_figure = go.Figure()
    store_overall_sales_figure.add_trace(
            go.Scatter(x=data[date_column], y=data[quantity_sold_column]))
    store_overall_sales_figure.update_layout(
            title_text=f"Current Sales Data for whole store",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column,
            yaxis_title=quantity_sold_column, )
    st.plotly_chart(store_overall_sales_figure)


def visualise_individual_product_sales(data, column_map):
    date_column = column_map["date_column"]
    product_column = column_map["product_column"]
    quantity_sold_column = column_map["quantity_sold_column"]
    SessionManager.set_state("product_index", 0)
    SessionManager.set_state("is_first", True)

    if "is_first" not in st.session_state or SessionManager.get_state("product_index") == 0:
        SessionManager.set_state("is_first", True)
    elif SessionManager.get_state("product_index") > 0:
        SessionManager.set_state("is_first", False)

    all_products_grouped = data.groupby(product_column)
    unique_products = [product_name for product_name, product_group in data.groupby(product_column)]

    if st.button("Visualise Individual Product Sales"):
        st.session_state.product_index = 0
        plot_individual_product(all_products_grouped, unique_products, date_column, quantity_sold_column)

    left_column, right_column = st.columns(2)
    previous_button = left_column.button("Previous", disabled=st.session_state.is_first)
    next_button = right_column.button("Next")

    if previous_button:
        SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)
        if SessionManager.get_state("product_index") != 0 < len(all_products_grouped) - 1:
            plot_individual_product(all_products_grouped, unique_products, date_column, quantity_sold_column)

    elif next_button:
        SessionManager.set_state("is_first", False)
        SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
        if 0 <= SessionManager.get_state("product_index") < len(all_products_grouped) - 1:
            plot_individual_product(all_products_grouped, unique_products, date_column, quantity_sold_column)
