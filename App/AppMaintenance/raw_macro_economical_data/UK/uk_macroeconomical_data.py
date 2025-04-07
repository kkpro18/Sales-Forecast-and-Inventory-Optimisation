import pandas as pd

# update filenames when a release is made for any dataset and rerun script

bank_rate = pd.read_csv("UK_BoE_Bank_Rate.csv", parse_dates=["Date"], dayfirst=True)
cphih = pd.read_csv("UK_CPIH_250125.csv")
internet_sales = pd.read_csv("UK_Internet_Sales_Ratio.csv")
ukrpi = pd.read_csv("UK_RPI_250125.csv")
unemployment = pd.read_csv("UK_Unemployment_Rate.csv")

def process_monthly_data(data_frame, date_col):
    # uses regular expressions to only select rows with 4 digits(d) for years eg 1989 and 3 letters([A-Z]) for months eg JAN, if null skipped
    data_frame = data_frame[data_frame[date_col].str.match(r'\d{4} [A-Z]{3}', na=False)]

    # converts existing date column to datetime format with existing format Year and Abbreviated month
    data_frame["Date"] = pd.to_datetime(data_frame[date_col], format='%Y %b', errors='coerce')
    data_frame = data_frame.drop(columns=[date_col]) # keeps all files with same Date column name and drops original
    return data_frame

cphih = process_monthly_data(cphih, "Title")
internet_sales = process_monthly_data(internet_sales, "Title")
ukrpi = process_monthly_data(ukrpi, "Title")
unemployment = process_monthly_data(unemployment, "Title")

# uses bank rate as the date range as that contains daily data
date_range = pd.DataFrame({
    "Date": pd.date_range(
        start=bank_rate["Date"].min(), # earliest date in the dataset
        end=bank_rate["Date"].max(),  # latest date in the dataset
        freq='D' # date range is daily
    )
})

# iteratively adds each dataset to the data frame by matching the date column
merged_data_frame = date_range.merge(bank_rate, on="Date", how="left")
for data_frame in [cphih, internet_sales, ukrpi, unemployment]:
    merged_data_frame = merged_data_frame.merge(data_frame, on="Date", how="left")

# since the datasets tend to include monthly entries, it needs to be forward filled to fill in the missing values between each month.
merged_data_frame.ffill(inplace=True)

# if there is still missing values, it is due to the actual dataset not having the data, so it is filled with 0 so no issues are created
merged_data_frame.fillna(0, inplace=True)

merged_data_frame.to_csv("Processed/uk_macro_economical_data.csv", index=False)

print("Merging complete. Saved as uk_macro_economical_data.csv")