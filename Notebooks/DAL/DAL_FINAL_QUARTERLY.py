#!/usr/bin/env python
# coding: utf-8

# In[100]:


# Enhanced feature engineering for improved test accuracy

# Reload base data
df = pd.read_csv("/Users/saketh/Downloads/DAL_Quarterly.csv")
df = df.dropna(subset=['High', 'Low', 'Adj Close'])

# Feature engineering
df['Volatility'] = (df['High'] - df['Low']) / df['Low']
df['Adj_Close_Lag1'] = df['Adj Close'].shift(1)
df['Return'] = df['Adj Close'].pct_change()
df['Return_Lag1'] = df['Return'].shift(1)
df['Adj_Close_MA2'] = df['Adj Close'].rolling(window=2).mean()
df['Adj_Close_STD2'] = df['Adj Close'].rolling(window=2).std()

# Drop NA from rolling/lags
df_model = df.dropna(subset=[
    'Volatility', 'Adj_Close_Lag1', 'Return', 'Return_Lag1', 
    'Adj_Close_MA2', 'Adj_Close_STD2', 'Adj Close'
])

# Define features and target
features = df_model[[
    'Volatility', 'Adj_Close_Lag1', 'Return', 
    'Return_Lag1', 'Adj_Close_MA2', 'Adj_Close_STD2'
]]
target = df_model['Adj Close']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
y = target.values

# Split
split_idx = int(len(X_scaled) * 0.85)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train ElasticNet
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
metrics = {
    "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
    "Train_MAE": round(mean_absolute_error(y_train, y_train_pred), 4),
    "Train_R2": round(r2_score(y_train, y_train_pred), 4),
    "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
    "Test_MAE": round(mean_absolute_error(y_test, y_test_pred), 4),
    "Test_R2": round(r2_score(y_test, y_test_pred), 4),
}

# Save predictions for plotting
df_model = df_model.copy()
df_model['Predicted'] = np.nan
df_model.iloc[split_idx:, df_model.columns.get_loc('Predicted')] = y_test_pred

# Forecast
model.fit(X_scaled, y)
X_future = X_scaled[-4:]
future_pred = model.predict(X_future)

# Future quarters
last_quarter = df_model['Timeframe_Quarter'].iloc[-1]
last_year, last_q = map(int, last_quarter.split('-Q'))
future_quarters = [f"{last_year + (last_q + i - 1) // 4}-Q{(last_q + i - 1) % 4 + 1}" for i in range(1, 5)]

# Plot
all_quarters = list(df_model['Timeframe_Quarter']) + future_quarters
x_indices = np.arange(len(all_quarters))
actual_prices = df_model['Adj Close'].tolist()
predicted_prices = df_model['Predicted'].tolist()

plt.figure(figsize=(16, 6))
plt.plot(x_indices[:len(actual_prices)], actual_prices, color='blue', marker='o', label='Actual Prices (Train + Test)')
for i, price in enumerate(actual_prices):
    plt.text(x_indices[i], price + 1, f"${price:.2f}", fontsize=7, color='blue', ha='center')

plt.plot(x_indices[split_idx:len(actual_prices)], predicted_prices[split_idx:], 'g--', marker='x', label='Test Predicted')
plt.plot(x_indices[len(actual_prices):], future_pred, 'ro-', label='Forecasted Prices')
plt.fill_between(
    x_indices[len(actual_prices):],
    np.array(future_pred) - 2,
    np.array(future_pred) + 2,
    color='red',
    alpha=0.2,
    label='95% CI'
)
plt.xticks(ticks=x_indices, labels=all_quarters, rotation=45)
plt.title("DAL Stock Price: ElasticNet Forecast with Engineered Features")
plt.xlabel("Quarter")
plt.ylabel("Adjusted Close Price ($)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

metrics


# In[104]:


import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("/Users/saketh/Downloads/DAL_Quarterly.csv")
df = df.dropna(subset=['High', 'Low', 'Adj Close'])

# Feature engineering
df['Volatility'] = (df['High'] - df['Low']) / df['Low']
df['Adj_Close_Lag1'] = df['Adj Close'].shift(1)
df['Return'] = df['Adj Close'].pct_change()
df['Return_Lag1'] = df['Return'].shift(1)
df['Adj_Close_MA2'] = df['Adj Close'].rolling(window=2).mean()
df['Adj_Close_STD2'] = df['Adj Close'].rolling(window=2).std()

# Drop NA introduced by lags/rolling
df_model = df.dropna(subset=[
    'Volatility', 'Adj_Close_Lag1', 'Return', 'Return_Lag1',
    'Adj_Close_MA2', 'Adj_Close_STD2', 'Adj Close'
]).copy()

# Define features and target
features = df_model[[
    'Volatility', 'Adj_Close_Lag1', 'Return',
    'Return_Lag1', 'Adj_Close_MA2', 'Adj_Close_STD2'
]]
target = df_model['Adj Close']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
y = target.values

# Train/test split (85% train, 15% test)
split_idx = int(len(X_scaled) * 0.85)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Train ElasticNet model
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
metrics = {
    "Train_R2": round(r2_score(y_train, y_train_pred), 4),
    "Test_R2": round(r2_score(y_test, y_test_pred), 4),
    "Train_RMSE": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
    "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
    "Train_MAE": round(mean_absolute_error(y_train, y_train_pred), 4),
    "Test_MAE": round(mean_absolute_error(y_test, y_test_pred), 4),
}
print("ElasticNet Metrics:", metrics)

# Add test predictions to DataFrame
df_model['Predicted'] = np.nan
df_model.iloc[split_idx:, df_model.columns.get_loc('Predicted')] = y_test_pred

# Refit model on all data for forecasting
model.fit(X_scaled, y)
X_future = X_scaled[-5:]  # use last 5 for forecast context
future_pred = model.predict(X_future)

# Generate next 5 quarters (including 2025-Q4)
last_quarter = df_model['Timeframe_Quarter'].iloc[-1]
last_year, last_q = map(int, last_quarter.split('-Q'))
future_quarters = [f"{last_year + (last_q + i - 1) // 4}-Q{(last_q + i - 1) % 4 + 1}" for i in range(1, 6)]

# Prepare output DataFrame
output_df = df_model[['Timeframe_Quarter', 'Adj Close', 'Predicted']].rename(
    columns={'Timeframe_Quarter': 'Timeframe', 'Adj Close': 'Close'}
)

forecast_df = pd.DataFrame({
    'Timeframe': future_quarters,
    'Close': np.nan,
    'Predicted': np.nan,
    'Forecast': future_pred
})

final_df = pd.concat([output_df, forecast_df], ignore_index=True)

# Save or return
final_df.to_csv("/Users/saketh/Downloads/DAL_ElasticNet_Forecast.csv", index=False)
print("✅ Forecast saved as: DAL_ElasticNet_Forecast_2025Q4.csv")


# In[130]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_csv("/Users/saketh/DAL_Quarterly.csv")
df = df.dropna(subset=['High', 'Low', 'Adj Close'])
df['Volatility'] = (df['High'] - df['Low']) / df['Low']
df['Adj_Close_Lag1'] = df['Adj Close'].shift(1)
df['Return'] = df['Adj Close'].pct_change()
df['Return_Lag1'] = df['Return'].shift(1)
df['Adj_Close_MA2'] = df['Adj Close'].rolling(2).mean()
df['Adj_Close_STD2'] = df['Adj Close'].rolling(2).std()
df_model = df.dropna().copy()

features = df_model[['Volatility', 'Adj_Close_Lag1', 'Return',
                     'Return_Lag1', 'Adj_Close_MA2', 'Adj_Close_STD2']]
target = df_model['Adj Close']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
y = target.values

# Split and model
split_idx = int(len(X_scaled) * 0.85)
model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
model.fit(X_scaled[:split_idx], y[:split_idx])
df_model['Predicted'] = np.nan
df_model.iloc[split_idx:, df_model.columns.get_loc('Predicted')] = model.predict(X_scaled[split_idx:])

# Forecast future
model.fit(X_scaled, y)
future_pred = model.predict(X_scaled[-4:])
last_q = df_model['Timeframe_Quarter'].iloc[-1]
year, q = map(int, last_q.split('-Q'))
future_quarters = [f"{year + (q + i - 1)//4}-Q{(q + i - 1) % 4 + 1}" for i in range(1, 5)]

# Plot
all_quarters = list(df_model['Timeframe_Quarter']) + future_quarters
x = np.arange(len(all_quarters))
zoom_start = all_quarters.index('2022-Q1')

plt.figure(figsize=(14, 6))

# Actual
actual_x = x[zoom_start:len(df_model)]
actual_y = df_model['Adj Close'].iloc[zoom_start:]
plt.plot(actual_x, actual_y, 'bo-', label='Actual')
# for i, val in zip(actual_x, actual_y):
#     plt.text(i, val + 1, f"${val:.2f}", fontsize=8, color='blue', ha='center')

# Predicted
predicted_x = x[split_idx:len(df_model)]
predicted_y = df_model['Predicted'].iloc[split_idx:]
plt.plot(predicted_x, predicted_y, 'gx--', label='Predicted (Test)')
# for i, val in zip(predicted_x, predicted_y):
#     plt.text(i, val + 1, f"${val:.2f}", fontsize=8, color='green', ha='center')

# Forecasted
forecast_x = x[len(df_model):]
plt.plot(forecast_x, future_pred, 'ro-', label='Forecasted')
for i, val in zip(forecast_x, future_pred):
    plt.text(i, val + 1, f"${val:.2f}", fontsize=8, color='red', ha='center')

# Confidence interval
plt.fill_between(
    forecast_x,
    np.array(future_pred) - 2,
    np.array(future_pred) + 2,
    color='red', alpha=0.2, label='95% CI'
)

# Formatting
plt.xticks(ticks=x[zoom_start:], labels=all_quarters[zoom_start:], rotation=45)
plt.xlabel("Year-Quarter")
plt.ylabel("Close")
plt.title("(Delta Air (DAL) Stock Price - Historical Data and Forecast)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:




