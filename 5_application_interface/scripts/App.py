import streamlit as st
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from methods import *
# to run application type this into the terminal "streamlit run 5_application_interface/scripts/App.py"

st.title("Sales Forecasting and Inventory Optimisation Service")
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
        

        # figure = plt.figure()
        # plt.plot(df[date_column].head(60000), df[sales_column].head(60000))
        # plt.title('Sales Data')
        # plt.xlabel('Date')
        # plt.ylabel('Units Sold')
        # st.pyplot(figure,use_container_width=True)

    st.multiselect("Select additional features that may enhance sales prediction accuracy", df.columns.drop(date_column).drop(sales_column))
    start_button = st.button("Begin Forecasting Sales")
    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value

        # create acf and dcf plots to identify other parameters for ARIMA model (p,q)

        # fit model

        # forecast

        # calculate performance metrics
        pass

