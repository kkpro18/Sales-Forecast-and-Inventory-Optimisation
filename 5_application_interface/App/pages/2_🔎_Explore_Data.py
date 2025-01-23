import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Explore Data",
    page_icon="🔎",
)
st.markdown("# Explore and Clean Your Sales Dataset")
st.write(
    """Here you can view your dataset and also pre-process it prior to the Sales Forecasting.""")


if 'uploaded_dataset' in st.session_state:
    uploaded_dataset = st.session_state["uploaded_dataset"]
    date_column = st.session_state["date_column"]
    sales_column = st.session_state["sales_column"]
    # visualise data
    visualise_button = st.button("Visualise Current Sales")
    if visualise_button:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=uploaded_dataset[date_column],y=uploaded_dataset[sales_column]))
        figure.update_layout(
                    title_text="Current Sales Data",
                    xaxis=dict(rangeslider=dict(visible=True), type="date"),
                    xaxis_title=date_column,
                    yaxis_title=sales_column,
                )
        st.plotly_chart(figure)

        st.write("👈 Next Stage: Forecast Sales")
else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")