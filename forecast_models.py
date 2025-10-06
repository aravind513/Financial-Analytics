# Forecasting Models for Time Series Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
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

# Number of future days to forecast (used by RF iterative loop)
future_days = 30

# 5. Random Forest Forecast (lag features)
lags = 10
df_rf = pd.DataFrame({'close': close_prices.flatten()})
for lag in range(1, lags+1):
    df_rf[f'lag_{lag}'] = df_rf['close'].shift(lag)
df_rf = df_rf.dropna().reset_index(drop=True)
X_rf = df_rf[[f'lag_{lag}' for lag in range(1, lags+1)]].values
y_rf = df_rf['close'].values
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)

# XGBoost model (same lag features)
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_rf, y_rf)

# LightGBM model
lgb = LGBMRegressor(n_estimators=100, random_state=42)
lgb.fit(X_rf, y_rf)

# Prepare iterative RF forecast
rf_input = df_rf[[f'lag_{lag}' for lag in range(1, lags+1)]].values[-1].tolist()
rf_pred_future = []
for _ in range(future_days):
    pred = rf.predict([rf_input])[0]
    rf_pred_future.append(pred)
    rf_input.pop(0)
    rf_input.append(pred)

# XGBoost iterative forecast
xgb_input = df_rf[[f'lag_{lag}' for lag in range(1, lags+1)]].values[-1].tolist()
xgb_pred_future = []
for _ in range(future_days):
    pred = xgb.predict([xgb_input])[0]
    xgb_pred_future.append(pred)
    xgb_input.pop(0)
    xgb_input.append(pred)

# LightGBM iterative forecast
lgb_input = df_rf[[f'lag_{lag}' for lag in range(1, lags+1)]].values[-1].tolist()
lgb_pred_future = []
for _ in range(future_days):
    pred = lgb.predict([lgb_input])[0]
    lgb_pred_future.append(pred)
    lgb_input.pop(0)
    lgb_input.append(pred)


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

# GRU model (same architecture idea as LSTM)
gru_model = Sequential([
    GRU(50, input_shape=(look_back, 1)),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
gru_model.fit(X_lstm, y_lstm, epochs=20, batch_size=16, verbose=0)
gru_pred_scaled = gru_model.predict(X_lstm)
gru_pred = scaler.inverse_transform(gru_pred_scaled)

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
gru_full = np.full(len(close_prices), np.nan)
gru_full[look_back:] = gru_pred.flatten()
results['gru'] = gru_full
sarima_full = np.full(len(close_prices), np.nan)
sarima_full[look_back:] = sarima_pred.flatten()
results['sarima'] = sarima_full
arma_full = np.full(len(close_prices), np.nan)
arma_full[look_back:] = arma_pred.flatten()
results['arma'] = arma_full
arima_full = np.full(len(close_prices), np.nan)
arima_full[look_back:] = arima_pred.flatten()
results['arima'] = arima_full
# Add historical RF predictions where possible (align with lags)
rf_hist = np.full(len(close_prices), np.nan)
rf_hist[len(close_prices) - len(X_rf):] = rf.predict(X_rf)
results['random_forest'] = rf_hist
results['xgboost'] = np.full(len(close_prices), np.nan)
results['xgboost'][len(close_prices) - len(X_rf):] = xgb.predict(X_rf)
results['lightgbm'] = np.full(len(close_prices), np.nan)
results['lightgbm'][len(close_prices) - len(X_rf):] = lgb.predict(X_rf)

# Save to Excel


# Forecast next 30 days starting from Oct 6, 2025
future_days = 30
start_date = pd.Timestamp('2025-10-06')
future_dates = pd.date_range(start_date, periods=future_days)

# Linear Regression future forecast
X_future = np.arange(len(close_prices), len(close_prices) + future_days).reshape(-1, 1)
lin_pred_future = lin_reg.predict(X_future).flatten()

# SARIMA future forecast
# SARIMAX future forecast with confidence intervals
sarimax_fore = sarima_result.get_forecast(steps=future_days)
sarimax_df = sarimax_fore.summary_frame(alpha=0.05)
sarima_pred_future = sarimax_df['mean'].values
sarima_pred_lower = sarimax_df['mean_ci_lower'].values
sarima_pred_upper = sarimax_df['mean_ci_upper'].values

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
future_results['random_forest'] = rf_pred_future
future_results['xgboost'] = xgb_pred_future
future_results['lightgbm'] = lgb_pred_future
# Add SARIMAX confidence intervals
future_results['sarimax_lower'] = sarima_pred_lower
future_results['sarimax_upper'] = sarima_pred_upper
# GRU future forecast
gru_input = prices_scaled[-look_back:].reshape(1, look_back, 1)
gru_pred_future = []
for _ in range(future_days):
    next_pred_scaled = gru_model.predict(gru_input, verbose=0)
    next_pred = scaler.inverse_transform(next_pred_scaled)[0, 0]
    gru_pred_future.append(next_pred)
    gru_input = np.roll(gru_input, -1, axis=1)
    gru_input[0, -1, 0] = next_pred_scaled[0, 0]
future_results['gru'] = gru_pred_future

# Calculate price ranges after DataFrame creation
last_price = close_prices.flatten()[-1]
future_results['arch_upper'] = last_price * (1 + future_results['arch_volatility'])
future_results['arch_lower'] = last_price * (1 - future_results['arch_volatility'])
future_results['garch_upper'] = last_price * (1 + future_results['garch_volatility'])
future_results['garch_lower'] = last_price * (1 - future_results['garch_volatility'])

# Combine and save
final_results = pd.concat([results, future_results], ignore_index=True)
try:
    final_results.to_excel('forecast_results.xlsx', index=False)
    print('Forecast results saved to forecast_results.xlsx')
except PermissionError:
    # If the file is open in Excel (locked), save to a timestamped file instead
    fallback_name = f"forecast_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    final_results.to_excel(fallback_name, index=False)
    print(f"Could not write to 'forecast_results.xlsx' (file may be open). Saved to {fallback_name} instead.")
