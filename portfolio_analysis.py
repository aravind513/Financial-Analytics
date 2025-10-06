import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from itertools import product

# Load stock data
stock1 = pd.read_csv('data.csv', usecols=['Date', 'Price'])
stock2 = pd.read_csv('data2.csv', usecols=['Date', 'Price'])
stock1.columns = ['date', 'price1']
stock2.columns = ['date', 'price2']
stock1['date'] = pd.to_datetime(stock1['date'], format='%d-%m-%Y', errors='coerce')
stock2['date'] = pd.to_datetime(stock2['date'], format='%d-%m-%Y', errors='coerce')
stock1['price1'] = pd.to_numeric(stock1['price1'], errors='coerce')
stock2['price2'] = pd.to_numeric(stock2['price2'], errors='coerce')

# Merge on date
df = pd.merge(stock1, stock2, on='date', how='inner')
df = df.sort_values('date')

# Calculate daily returns
returns1 = df['price1'].pct_change().dropna()
returns2 = df['price2'].pct_change().dropna()
returns = pd.DataFrame({'returns1': returns1, 'returns2': returns2})

# Markowitz mean-variance optimization
allocations = np.arange(0, 1.05, 0.05)
results = []
for w1 in allocations:
    w2 = 1 - w1
    port_return = (returns['returns1'] * w1 + returns['returns2'] * w2).mean() * 252
    port_vol = (returns['returns1'] * w1 + returns['returns2'] * w2).std() * np.sqrt(252)
    results.append({'w1': w1, 'w2': w2, 'annual_return': port_return, 'annual_volatility': port_vol})
results_df = pd.DataFrame(results)

# Cluster analysis (categorise by volatility and return)
kmeans = KMeans(n_clusters=2, random_state=42)
results_df['cluster'] = kmeans.fit_predict(results_df[['annual_return', 'annual_volatility']])

# Highlight best combination (highest Sharpe ratio)
risk_free_rate = 0.06  # Example: 6% annual risk-free rate
results_df['sharpe_ratio'] = (results_df['annual_return'] - risk_free_rate) / results_df['annual_volatility']
best_idx = results_df['sharpe_ratio'].idxmax()
results_df['best'] = False
results_df.loc[best_idx, 'best'] = True

# Calculate risk-adjusted ratios for best portfolio
best = results_df.loc[best_idx]
portfolio_returns = returns['returns1'] * best['w1'] + returns['returns2'] * best['w2']
portfolio_vol = portfolio_returns.std() * np.sqrt(252)
portfolio_mean = portfolio_returns.mean() * 252
market_return = (returns['returns1'].mean() + returns['returns2'].mean()) / 2 * 252
market_vol = (returns['returns1'].std() + returns['returns2'].std()) / 2 * np.sqrt(252)
beta = np.cov(portfolio_returns, returns['returns1'])[0, 1] / np.var(returns['returns1'])

sharpe = (portfolio_mean - risk_free_rate) / portfolio_vol
treynor = (portfolio_mean - risk_free_rate) / beta
jensen = portfolio_mean - (risk_free_rate + beta * (market_return - risk_free_rate))

ratios_df = pd.DataFrame({
    'Sharpe Ratio': [sharpe],
    'Treynor Ratio': [treynor],
    'Jensen Alpha': [jensen]
})

# Write to Excel
with pd.ExcelWriter('portfolio_analysis.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Allocations', index=False)
    ratios_df.to_excel(writer, sheet_name='RiskAdjusted', index=False)
