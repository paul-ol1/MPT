import os
import pandas as pd
import numpy as np

# This code is used to compute the Global Minimum Variance Portfolio (GMVP) for a given set of assets
# The GMVP is the portfolio with the lowest risk (volatility) and is calculated using the assets' historical returns and covariance matrix
# The GMVP weights are calculated using the formula: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
# The expected return and volatility of the GMVP are then calculated
# The code reads the CSV files saved in the data folder and computes the GMVP for different asset sizes
# to execute this code, run python compute.py
# please run data_collection.py before running this code to collect the data

# Function to read and concatenate CSV files side by side to form a DataFrame
def read_and_concat_csvs_side_by_side(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, skiprows=1)
        dfs.append(df)

    df_combined = pd.concat(dfs, axis=1)
    df_combined = df_combined[sorted(df_combined.columns)]
    return df_combined

# Folder path to data
folder_path = '/Users/pauloladele/MPT/datarussell' #please change this to your own path
df_full = read_and_concat_csvs_side_by_side(folder_path)

# List of asset sizes to test
execsizes = [10, 100, 200, 500, 955]

for execsize in execsizes:
    df = df_full.iloc[:, :execsize]  # Use only the first 'execsize' assets

    # Calculate log returns
    df_log_returns = np.log(df / df.shift(1)).dropna()

    # Calculate the covariance matrix and mean returns
    cov_matrix = df_log_returns.cov()
    mean_returns = df_log_returns.mean()

    # Invert the covariance matrix
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # Number of assets
    n = len(mean_returns)

    # Vector of ones
    ones = np.ones(n)

    # Function to compute GMVP weights
    def calculate_portfolio_weights(inv_cov_matrix, ones):
        numerator = np.dot(inv_cov_matrix, ones)
        denominator = np.dot(ones.T, np.dot(inv_cov_matrix, ones))
        weights = numerator / denominator
        return weights

    # Calculate weights
    optimal_weights = calculate_portfolio_weights(inv_cov_matrix, ones)

    # Calculate portfolio return and volatility
    portfolio_return = np.dot(optimal_weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

    print(f"\nOptimal Portfolio Weights for asset size: {execsize}")
    print(np.round(optimal_weights, 4))

    print(f"Expected Return: {portfolio_return:.4f}")
    print(f"Portfolio Volatility (Risk): {portfolio_volatility:.4f}")
