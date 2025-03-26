import uuid

import streamlit as st
import plotly.graph_objects as go
from App.utils.session_manager import SessionManager

def visualise_storewide_sales(data, column_mapping, quantity_sold_column=None):
    date_column = column_mapping["date_column"]
    if quantity_sold_column is None:
        quantity_sold_column = column_mapping["quantity_sold_column"]
    else:
        quantity_sold_column = quantity_sold_column

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data[date_column], y=data[quantity_sold_column]))
    figure.update_layout(
            title_text=f"Current Sales Data for whole store",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column,
            yaxis_title=quantity_sold_column, )
    st.plotly_chart(figure)

def visualise_individual_product_sales(product_groups, column_mapping):
    date_column = column_mapping["date_column"]
    quantity_sold_column = column_mapping["quantity_sold_column"]
    product_groups = product_groups.groupby(column_mapping["product_column"])
    product_names = list(product_groups.groups.keys())

    def plot_individual_product():
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=product_groups.get_group(product_names[st.session_state.product_index])[date_column],
            y=product_groups.get_group(product_names[st.session_state.product_index])[quantity_sold_column])
        )
        figure.update_layout(
            title_text=f"Current Sales for Product: {product_names[st.session_state.product_index]}",
            xaxis=dict(rangeslider=dict(visible=True), type="date"),
            xaxis_title=date_column,
            yaxis_title=quantity_sold_column,
        )
        st.plotly_chart(figure, key=uuid.uuid4())

    if not SessionManager.is_there("product_index"):
        SessionManager.set_state("product_index", 0)
        plot_individual_product()

    left_column, right_column = st.columns(2)
    previous_button = left_column.button("Previous", disabled=SessionManager.get_state("product_index") == 0)
    next_button = right_column.button("Next", disabled=SessionManager.get_state("product_index") >= len(product_names) - 1)

    if previous_button:
        if SessionManager.get_state("product_index") > 0:
            SessionManager.set_state("product_index", SessionManager.get_state("product_index") - 1)
        plot_individual_product()


    elif next_button:
        SessionManager.set_state("product_index", SessionManager.get_state("product_index") + 1)
        plot_individual_product()