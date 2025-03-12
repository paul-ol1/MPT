import pandas as pd
import yfinance as yf
import os



save_dir = os.path.join(os.path.dirname(__file__), '..', 'MPT', 'data')
# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'MPT', 'sp500_companies.csv'))
tickers = df['Symbol'].values.tolist()

for ticker in tickers:
    data = yf.download(ticker, period='10y', interval='1wk')[['Close']]

    if(data.isnull().values.any() or len(data) < 523):
        print(f'{ticker} has missing values')
        continue

    data.reset_index(inplace=True)
    data.to_csv(f'{save_dir}/{ticker}.csv', index=False)
