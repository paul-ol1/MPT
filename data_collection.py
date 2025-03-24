import pandas as pd
import yfinance as yf
from scipy.stats.mstats import winsorize
import numpy as np
import os
import datetime
import time

# This code snippet is used to collect data for the assets in the Russell 2000 index but can be used for any other ticker sets or diversified portfolio of assets
# The data is collected from Yahoo Finance using the yfinance library
# The data is then winsorized to remove outliers and saved as CSV files
# execute this by running python data_collection.py

# Function to winsorize data between lower and upper percentiles
def winzorize(data, lower, upper):
    lower_bound = np.percentile(data, lower)
    upper_bound = np.percentile(data, upper)
    data = np.clip(data, lower_bound, upper_bound)
    return data

save_dir = os.path.join(os.path.dirname(__file__), '..', 'MPT', 'data')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'MPT', 'russell2000.csv')) #replace this by any ticker available with an already diversified portfolio of assets, in this case, the Russell 2000 or S&P 500
tickers = df['Symbol'].values.tolist()

end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(days=10 * 365)  # 10 years in days

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)[['Close']]

    if data.isnull().values.any():
        print(f'{ticker} has missing values.')
        continue

    # Check if the data has at least 10 years of daily data
    ten_years_days = 10 * 365 # approximate 10 years in days
    if len(data) < 2514:
        print(f'{ticker} has insufficient data for 10 years ({len(data)} days)')
        continue

    data = data.dropna() #removes all nan values

    sleep_time = 0.1
    time.sleep(sleep_time)

    data['Close'] = winzorize(data['Close'], 1, 95)
    data.to_csv(f'{save_dir}/{ticker}.csv', index=False)