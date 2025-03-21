import pandas as pd
import yfinance as yf
from scipy.stats.mstats import winsorize
import numpy as np
import os
import datetime
import time



def winzorize(data,lower,upper):
    lower_bound= np.percentile(data, lower)
    upper_bound = np.percentile(data, upper)
    data = np.clip(data, lower_bound, upper_bound)
    return data

save_dir = os.path.join(os.path.dirname(__file__), '..', 'MPT', 'data')
# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'MPT', 'sp500.csv'))
tickers = df['Symbol'].values.tolist()
end_date = datetime.datetime.today().date()
start_date = end_date - datetime.timedelta(weeks=512)


for ticker in tickers:

    # Correcting start and end order
    data = yf.download(ticker, start=start_date, end=end_date, interval='1wk')[['Close']]

    # Ensure exactly 512 weeks of data
    if data.isnull().values.any() or len(data) < 512:

        print(f'{ticker} has missing values or insufficient data ({len(data)} weeks)')
        continue
    sleep_time = 0.1
    time.sleep(sleep_time)

    #winsorize the data
    data['Close'] = winzorize(data['Close'], 1, 95)
    # Save cleaned dataset
    data.to_csv(f'{save_dir}/{ticker}.csv', index=False)
