from time import sleep

import pandas as pd
import streamlit as st

# to run application type this into the terminal " streamlit run experiments/5_application_interface/App/0_Home.py "
st.set_page_config(
    page_title="Upload Dataset",
    page_icon="📁",
    layout="wide",
)

st.markdown("# Upload Dataset")
st.write("""Here you can upload your dataset in CSV format. You can click to upload or drag a CSV file over to this page.""")

uploaded_dataset = st.file_uploader("Upload your sales data", type="csv")

if uploaded_dataset is not None:
    raw_df = pd.read_csv(uploaded_dataset, encoding="unicode_escape")
    st.session_state['uploaded_dataset'] = raw_df
    st.success("File uploaded successfully!")

if 'uploaded_dataset' in st.session_state:
    if 'confirmation_button_clicked' not in st.session_state:
        st.session_state.confirmation_button_clicked = False

    if not st.session_state.confirmation_button_clicked:
        st.write(st.session_state['uploaded_dataset'].head())

        st.write("Select Columns to Use For the Forecast")

        columns = list(st.session_state['uploaded_dataset'].columns)

        # invoice date // pre-process data e.g. some may have times
        if 'selected_date_column' not in st.session_state:
            st.session_state.selected_date_column = 0
        date_column = st.selectbox("Select the Column for Invoice Date", options=columns, index=st.session_state.selected_date_column, disabled=st.session_state.confirmation_button_clicked)
        st.session_state["date_column"] = date_column
        if date_column in columns:
            st.session_state.selected_date_column = columns.index(date_column)

        # productID/ProductName // separate into products and its own sales
        columns.remove(date_column)
        if 'selected_product_column' not in st.session_state:
            st.session_state.selected_product_column = 0
        product_column = st.selectbox("Select the Column for Product ID / Name", options=columns, index=st.session_state.selected_product_column,
                                      disabled=st.session_state.confirmation_button_clicked)
        st.session_state["product_column"] = product_column
        if product_column in columns:
            st.session_state.selected_product_column = columns.index(product_column)

        # price
        columns.remove(product_column)
        if 'selected_price_column' not in st.session_state:
            st.session_state.selected_price_column = 0
        unit_price_column = st.selectbox("Select the Column for Unit Price", options=columns, index=st.session_state.selected_price_column,
                                         disabled=st.session_state.confirmation_button_clicked)
        st.session_state["unit_price_column"] = unit_price_column
        if unit_price_column in columns:
            st.session_state.selected_price_column = columns.index(unit_price_column)

        # Quantity Sold
        columns.remove(unit_price_column)

        if 'selected_units_sold_column' not in st.session_state:
            st.session_state.selected_units_sold_column = 0
        units_sold_column = st.selectbox("Select the Column for Units Sold", options=columns, index=st.session_state.selected_units_sold_column,
                                         disabled=st.session_state.confirmation_button_clicked)
        st.session_state["units_sold_column"] = units_sold_column
        st.session_state.selected_units_sold_column = columns.index(units_sold_column)

        regions = ["UK", "USA"]
        if 'selected_region' not in st.session_state:
            st.session_state.selected_region = 0
        selected_region = st.session_state.selected_region
        region = st.radio(label="Enter the region where your store is based?", options=regions, index=st.session_state.selected_region, disabled=st.session_state.confirmation_button_clicked)
        st.session_state["region"] = region
        st.session_state.selected_region = regions.index(region)
        st.markdown(f"You Selected {st.session_state.region}.")

        confirmation_button_clicked = st.button("Confirm Selected Options", disabled=st.session_state.confirmation_button_clicked)
        if confirmation_button_clicked:
            st.session_state.uploaded_dataset = st.session_state.uploaded_dataset[[date_column, product_column, unit_price_column, units_sold_column]]
            st.session_state.confirmation_button_clicked = True
    else:
        st.success("Selection Complete!")
        st.write("Dataset with the selected columns: ")
        st.write("Invoice Date Column: ", st.session_state.date_column)
        st.write("Product Info Column: ", st.session_state.product_column)
        st.write("Unit Price: ", st.session_state.unit_price_column)
        st.write("Units Sold Column: ", st.session_state.units_sold_column)
        st.write("Region: ", st.session_state.region)

        # st.switch_page("pages/2_🔎_Explore_Data.py")

    st.page_link("pages/2_🔎_Explore_Data.py", label="👈 Next Stage: Visualise The Dataset", icon="🔎")