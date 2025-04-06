import pandas as pd

# update filenames when a release is made for any dataset and rerun script

interest_rate = pd.read_csv("USA_interest-rates-2000-2024.csv", parse_dates=["Date"], dayfirst=True)
interest_rate.rename(columns={"LT Real Average (10> Yrs)" : "Interest Rate"}, inplace=True)
unemployment_rate = pd.read_csv("USA_Unemployment_Rate_1948_2024.csv")

def process_dates(data_frame, date_col, drop_columns=None, rename_column=None):
    if drop_columns:
        data_frame = data_frame.drop(columns=drop_columns)
    if rename_column:
        data_frame.rename(columns={data_frame.columns[1]: rename_column}, inplace=True)
    # uses regular expressions to only select rows with 4 digits(d) for years eg 1989 and 3 letters([A-Za-z]) for months eg Jan, if null skipped
    data_frame = data_frame[data_frame[date_col].str.match(r'\d{4} [A-Za-z]{3}', na=False)]

    # converts existing date column to datetime format with existing format Year and Abbreviated month
    data_frame["Date"] = pd.to_datetime(data_frame[date_col], format='%Y %b', errors='coerce')
    data_frame = data_frame.drop(columns=[date_col])  # keeps all files with same Date column name and drops original
    return data_frame


unemployment_rate = process_dates(unemployment_rate, "Label", drop_columns=["Series ID", "Year", "Period"], rename_column="Unemployment Rate")

# uses bank rate as the date range as that contains daily data
date_range = pd.DataFrame({
    "Date": pd.date_range(
        start=interest_rate["Date"].min(),  # earliest date in the dataset
        end=interest_rate["Date"].max(),  # latest date in the dataset
        freq='D'  # date range is daily
    )
})

# iteratively adds each dataset to the data frame by matching the date column
merged_data_frame = date_range.merge(interest_rate, on="Date", how="left")
for data_frame in [unemployment_rate]:
    merged_data_frame = merged_data_frame.merge(data_frame, on="Date", how="left")

# since the datasets tend to include monthly entries, it needs to be forward filled to fill in the missing values between each month.
merged_data_frame.ffill(inplace=True)

# if there is still missing values, it is due to the actual dataset not having the data, so it is filled with 0 so no issues are created
merged_data_frame.fillna(0, inplace=True)

merged_data_frame.to_csv("Processed/usa_macro_economical_data.csv", index=False)

print("Merging complete. Saved as usa_macro_economical_data.csv")
