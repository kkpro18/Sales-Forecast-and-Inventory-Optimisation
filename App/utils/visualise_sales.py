import uuid

import streamlit as st
import plotly.graph_objects as go
from App.utils.session_manager import SessionManager

def visualise_storewide_sales(data, column_mapping):
    date_column = column_mapping["date_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data[date_column], y=data[quantity_sold_column]))
    figure.update_layout(
            title_text=f"Current Sales Data for whole store",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column,
            yaxis_title=quantity_sold_column, )
    st.plotly_chart(figure)

def visualise_individual_product_sales(data, column_mapping):
    def plot_individual_product(products_groups, product_names, date_column_name, quantity_sold_column_name):
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=products_groups.get_group(product_names[st.session_state.product_index])[date_column_name],
            y=products_groups.get_group(product_names[st.session_state.product_index])[quantity_sold_column_name])
        )
        figure.update_layout(
            title_text=f"Current Sales for Product: {product_names[st.session_state.product_index]}",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column_name,
            yaxis_title=quantity_sold_column_name,
        )
        st.plotly_chart(figure, key=uuid.uuid4())

    date_column = column_mapping["date_column"]
    product_column = column_mapping["product_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]

    product_groups = data.groupby(product_column)
    products = list(product_groups.groups.keys()) # [product_name for product_name, product_group in data.groupby(product_column)]

    plot_individual_product(product_groups, products, date_column, quantity_sold_column)

    left_column, right_column = st.columns(2)
    previous_button = left_column.button("Previous", disabled=SessionManager.get_state("product_index") == 0)
    next_button = right_column.button("Next", disabled=SessionManager.get_state("product_index") >= len(products) - 1)

    if previous_button and SessionManager.get_state("product_index") != 0:
        if SessionManager.get_state("product_index") - 1 == 0:
            SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)
        plot_individual_product(product_groups, products, date_column, quantity_sold_column)


    elif next_button:
        SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
        plot_individual_product(product_groups, products, date_column, quantity_sold_column)
