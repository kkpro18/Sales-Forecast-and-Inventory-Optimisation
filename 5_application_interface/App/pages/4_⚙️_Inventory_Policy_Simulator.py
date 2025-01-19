import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from matplotlib.pyplot import xlabel

# to run application type this into the terminal "streamlit run 5_application_interface/App/1_📁_Upload_Dataset.py"
st.set_page_config(
    page_title="Sales Forecasting and Inventory Optimisation Service",
    page_icon="📈",
)

# upload dataset
uploaded_dataset = st.file_uploader("Upload your sales data", type="csv")
if uploaded_dataset is not None:
    df = pd.read_csv(uploaded_dataset)
    st.write(df)
    st.write("Select Columns to Use For the Forecast")
    date_column = st.selectbox("Select the Column for Dates", df.columns)
    sales_column = st.selectbox("Select the Column for Sales", df.columns.drop(date_column))

    # visualise data
    visualise_button = st.button("Visualise Current Sales")
    if visualise_button:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=df[date_column],y=df[sales_column]))
        figure.update_layout(
                    title_text=f"Sales Forecasting",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    xaxis_title=date_column,
                    yaxis_title=sales_column,
                )
        st.plotly_chart(figure)

    st.multiselect("Select features that can be used to predict sales", df.columns.drop(date_column).drop(sales_column))

    start_button = st.button("Begin Forecasting Sales")

    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value

        # create acf and dcf plots to identify other parameters for ARIMA model (p,q)

        # fit model

        # forecast

        # calculate performance metrics
        pass

