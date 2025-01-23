import streamlit as st

# to run application type this into the terminal "streamlit run 5_application_interface/App/0_Home.py"
st.set_page_config(
    page_title="Forecast Sales",
    page_icon="📈",
)
st.markdown("# Forecast Sales")
st.write("""Let's Begin Forecasting Sales!""")

if 'uploaded_dataset' in st.session_state:
    uploaded_dataset = st.session_state["uploaded_dataset"]
    date_column = st.session_state["date_column"]
    sales_column = st.session_state["sales_column"]
    st.multiselect("Select features that can be used to predict sales", uploaded_dataset.columns.drop(date_column).drop(sales_column))

    start_button = st.button("Begin Forecasting Sales")

    if start_button:
        # check if data is stationary, otherwise apply differencing until stationary - number of differencing steps is noted as d value

        # create acf and dcf plots to identify other parameters for ARIMA model (p,q)

        # fit model

        # forecast

        # calculate performance metrics
        pass

else:
    st.warning("Missing Your Dataset, 👈 Please Upload Dataset ")