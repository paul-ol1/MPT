import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Markowitz Portfolio Theory - Monte Carlo Simulation

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
df_log_returns = np.log(df / df.shift(1))

# Remove the first row (which contains NaN values after shifting)
df_log_returns = df_log_returns.iloc[1:]

# Calculate the covariance matrix
cov_matrix = df_log_returns.cov()

# Means of df_log_returns
mean_returns = df_log_returns.mean()

# Number of assets in the portfolio
n = len(mean_returns)

# Monte Carlo simulation function
def monte_carlo_simulation(mean_returns, cov_matrix, num_simulations=10000):
    # Initialize an array to store the results
    results = np.zeros((3, num_simulations))  # [0] - return, [1] - volatility, [2] - Sharpe ratio

    for i in range(num_simulations):
        # Generate random portfolio weights that sum to 1
        weights = np.random.random(n)
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        # Calculate portfolio return
        portfolio_return = np.dot(weights, mean_returns)

        # Calculate portfolio volatility (standard deviation)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # Calculate Sharpe ratio (assuming a risk-free rate of 0)
        sharpe_ratio = portfolio_return / portfolio_volatility

        # Store the results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    return results

# Run Monte Carlo simulation
num_simulations = 10000
results = monte_carlo_simulation(mean_returns, cov_matrix, num_simulations)

# Extract results
portfolio_returns = results[0, :]
portfolio_volatilities = results[1, :]
portfolio_sharpe_ratios = results[2, :]

# Plot the Monte Carlo simulations (Return vs. Volatility)
plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.5)
plt.title('Monte Carlo Simulation: Portfolio Optimization')
plt.xlabel('Portfolio Volatility (Risk)')
plt.ylabel('Portfolio Return')
plt.colorbar(label='Sharpe Ratio')
plt.show()

# Print some statistics for the optimal portfolio
max_sharpe_idx = np.argmax(portfolio_sharpe_ratios)
print(f"Maximum Sharpe Ratio Portfolio:\nReturn: {portfolio_returns[max_sharpe_idx]:.4f}, Volatility: {portfolio_volatilities[max_sharpe_idx]:.4f}, Sharpe Ratio: {portfolio_sharpe_ratios[max_sharpe_idx]:.4f}")
