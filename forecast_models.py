# Forecasting Models for Time Series Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# For ARCH/GARCH
from arch import arch_model

# Load data

# Read only Date and Price columns
file_path = 'data.csv'
df = pd.read_csv(file_path, usecols=['Date', 'Price'])
df.columns = ['date', 'close']
df['date'] = pd.to_datetime(df['date'])
close_prices = df['close'].astype(float).values.reshape(-1, 1)

# 1. Linear Regression Forecast
X = np.arange(len(close_prices)).reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X, close_prices)
lin_pred = lin_reg.predict(X)

# 2. Logistic Regression (predicting up/down movement)



# Logistic regression removed as per user request

# 3. LSTM Forecast
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(close_prices)
look_back = 10
X_lstm, y_lstm = [], []
for i in range(len(prices_scaled) - look_back):
    X_lstm.append(prices_scaled[i:i+look_back, 0])
    y_lstm.append(prices_scaled[i+look_back, 0])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
lstm_model = Sequential([
    LSTM(50, input_shape=(look_back, 1)),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0)
lstm_pred_scaled = lstm_model.predict(X_lstm)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)

# 4. SARIMA Forecast
sarima_order = (1, 1, 1)
sarima_seasonal_order = (1, 1, 1, 12)
sarima_model = SARIMAX(close_prices, order=sarima_order, seasonal_order=sarima_seasonal_order)
sarima_result = sarima_model.fit(disp=False)
sarima_pred = sarima_result.predict(start=look_back, end=len(close_prices)-1)

# ARMA Forecast
from statsmodels.tsa.arima.model import ARIMA
arma_model = ARIMA(close_prices, order=(2,0,2))
arma_result = arma_model.fit()
arma_pred = arma_result.predict(start=look_back, end=len(close_prices)-1)

# ARIMA Forecast
arima_model = ARIMA(close_prices, order=(2,1,2))
arima_result = arima_model.fit()
arima_pred = arima_result.predict(start=look_back, end=len(close_prices)-1)

# Plotting

# Prepare results for Excel
# Calculate daily returns for volatility modeling
returns = pd.Series(close_prices.flatten()).pct_change().dropna()

# Fit ARCH model
arch_mod = arch_model(returns, vol='ARCH', p=1)
arch_res = arch_mod.fit(disp='off')
arch_forecast = arch_res.forecast(horizon=30)
arch_vol_forecast = arch_forecast.variance.values[-1] ** 0.5  # std dev for next 30 days

# Fit GARCH model
garch_mod = arch_model(returns, vol='Garch', p=1, q=1)
garch_res = garch_mod.fit(disp='off')
garch_forecast = garch_res.forecast(horizon=30)
garch_vol_forecast = garch_forecast.variance.values[-1] ** 0.5

results = pd.DataFrame({'date': df['date']})
results['actual_close'] = close_prices.flatten()
results['linear_regression'] = lin_pred.flatten()
lstm_full = np.full(len(close_prices), np.nan)
lstm_full[look_back:] = lstm_pred.flatten()
results['lstm'] = lstm_full
sarima_full = np.full(len(close_prices), np.nan)
sarima_full[look_back:] = sarima_pred.flatten()
results['sarima'] = sarima_full
arma_full = np.full(len(close_prices), np.nan)
arma_full[look_back:] = arma_pred.flatten()
results['arma'] = arma_full
arima_full = np.full(len(close_prices), np.nan)
arima_full[look_back:] = arima_pred.flatten()
results['arima'] = arima_full

# Save to Excel


# Forecast next 30 days starting from Oct 6, 2025
future_days = 30
start_date = pd.Timestamp('2025-10-06')
future_dates = pd.date_range(start_date, periods=future_days)

# Linear Regression future forecast
X_future = np.arange(len(close_prices), len(close_prices) + future_days).reshape(-1, 1)
lin_pred_future = lin_reg.predict(X_future).flatten()

# SARIMA future forecast
sarima_pred_future = sarima_result.predict(start=len(close_prices), end=len(close_prices) + future_days - 1)

# ARMA future forecast
arma_pred_future = arma_result.predict(start=len(close_prices), end=len(close_prices) + future_days - 1)

# ARIMA future forecast
arima_pred_future = arima_result.predict(start=len(close_prices), end=len(close_prices) + future_days - 1)

# LSTM future forecast
lstm_input = prices_scaled[-look_back:].reshape(1, look_back, 1)
lstm_pred_future = []
for _ in range(future_days):
    next_pred_scaled = lstm_model.predict(lstm_input, verbose=0)
    next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
    lstm_pred_future.append(next_pred)
    # update input for next prediction
    lstm_input = np.roll(lstm_input, -1, axis=1)
    lstm_input[0, -1, 0] = next_pred_scaled[0, 0]

# Prepare future results

future_results = pd.DataFrame({'date': future_dates})
future_results['actual_close'] = [np.nan] * future_days
future_results['linear_regression'] = lin_pred_future
future_results['lstm'] = lstm_pred_future
future_results['sarima'] = sarima_pred_future
future_results['arma'] = arma_pred_future
future_results['arima'] = arima_pred_future
future_results['arch_volatility'] = arch_vol_forecast
future_results['garch_volatility'] = garch_vol_forecast

# Calculate price ranges after DataFrame creation
last_price = close_prices.flatten()[-1]
future_results['arch_upper'] = last_price * (1 + future_results['arch_volatility'])
future_results['arch_lower'] = last_price * (1 - future_results['arch_volatility'])
future_results['garch_upper'] = last_price * (1 + future_results['garch_volatility'])
future_results['garch_lower'] = last_price * (1 - future_results['garch_volatility'])

# Combine and save
final_results = pd.concat([results, future_results], ignore_index=True)
final_results.to_excel('forecast_results.xlsx', index=False)
print('Forecast results saved to forecast_results.xlsx')
