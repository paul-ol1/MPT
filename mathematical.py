import os
import pandas as pd
import numpy as np

#Markowitz Portfolio Theory
#Optimization Problem

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


folder_path = '/Users/pauloladele/MPT/data'
df = read_and_concat_csvs_side_by_side(folder_path)


# Calculate log returns: log(P_t) - log(P_{t-1})
df_log_returns = np.log(df/ df.shift(1))

# Remove the first row (which contains NaN values after shifting)
df_log_returns = df_log_returns.iloc[1:]

# Calculate the covariance matrix
cov_matrix = df_log_returns.cov()

#means of df_log_returns
mean_returns = df_log_returns.mean()
print(cov_matrix)
#invert the covariance matrix
inv_cov_matrix = np.linalg.inv(cov_matrix)

# Number of assets in the portfolio
n = len(mean_returns)

#vector of ones
ones = np.ones(n)
#vector of mean returns

# Objective: Minimize portfolio variance for a given expected return
def calculate_portfolio_weights(inv_cov_matrix, ones):
    numerator = np.dot(inv_cov_matrix, ones)
    denominator = np.dot(np.dot(ones.T, inv_cov_matrix), ones)
    weights = numerator / denominator
    return weights

# Calculate optimal portfolio weights
optimal_weights = calculate_portfolio_weights(inv_cov_matrix, ones)

# Calculate portfolio return and volatility
portfolio_return = np.dot(optimal_weights, mean_returns)
portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

print("Optimal Portfolio Weights:")
print(optimal_weights)

print(f"Expected Portfolio Return: {portfolio_return:.4f}")
print(f"Portfolio Volatility (Risk): {portfolio_volatility:.4f}")

