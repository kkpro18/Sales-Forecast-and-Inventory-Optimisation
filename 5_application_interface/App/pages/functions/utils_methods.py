import streamlit as st
import pandas as pd

global df, date_column, sales_column

def update_dataset(uploaded_df):
    df = uploaded_df

def get_df():
    return df

def update_date_column(column):
    date_column = column

def get_date_column():
    return date_column

def update_sales_column(column):
    sales_column = column

def get_sales_column():
    return sales_column
