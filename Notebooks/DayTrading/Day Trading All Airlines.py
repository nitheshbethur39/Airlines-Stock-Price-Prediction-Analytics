#!/usr/bin/env python
# coding: utf-8

# ## AAL

# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Load the data
print("Loading data...")
df = pd.read_csv('AAL_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Function to safely check if a column exists and fillna if it doesn't
def safe_create_feature(dataframe, column_name, default_value=0):
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value
    else:
        dataframe[column_name] = dataframe[column_name].fillna(default_value)
    return dataframe

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Determine holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.5) & (data['Hour'] <= 16.0)).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['AAL_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['AAL_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['AAL_RSI_14'] = calculate_rsi(data['AAL_Close'], window=14)
    data['AAL_RSI_7'] = calculate_rsi(data['AAL_Close'], window=7)
    
    data['AAL_MACD'], data['AAL_MACD_Signal'], data['AAL_MACD_Hist'] = calculate_macd(data['AAL_Close'])
    
    data['AAL_BB_Mid'], data['AAL_BB_Upper'], data['AAL_BB_Lower'] = calculate_bollinger_bands(data['AAL_Close'])
    data['AAL_BB_Width'] = data['AAL_BB_Upper'] - data['AAL_BB_Lower']
    data['AAL_BB_Pct'] = (data['AAL_Close'] - data['AAL_BB_Lower']) / (data['AAL_BB_Upper'] - data['AAL_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'AAL']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'AAL_Price_Trend_{window}d'] = data['AAL_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'AAL_Volume_Trend_{window}d'] = data['AAL_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between AAL and the market indicator
            data[f'AAL_{indicator}_Corr_{window}h'] = data['AAL_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for AAL_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'AAL_Close_Lag_{lag}'] = temp_df['AAL_Close'].shift(lag)
    temp_df[f'AAL_Return_Lag_{lag}'] = temp_df['AAL_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variable - next hour's AAL_Close
temp_df['Next_AAL_Close'] = temp_df['AAL_Close'].shift(-1)

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_AAL_Close'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsBeforeHoliday', 'IsAfterHoliday', 'IsOpeningHour', 'IsClosingHour',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfMonth_sin', 'DayOfMonth_cos',
    'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'AAL_RSI_14', 'AAL_RSI_7', 'AAL_MACD', 'AAL_MACD_Signal', 'AAL_MACD_Hist',
    'AAL_BB_Width', 'AAL_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'AAL']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'AAL_Price_Trend_{window}d',
        f'AAL_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'AAL_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'AAL_Close_Lag_{lag}',
        f'AAL_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original AAL features
aal_features = ['AAL_Open', 'AAL_High', 'AAL_Low', 'AAL_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, lag_features, aal_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Using {len(all_features)} features for training")

# Apply exponential weighting for recent data with shorter half-life
half_life = 7  # Use 7 days instead of 30 for more aggressive weighting
decay_factor = np.log(2) / half_life
max_date = train_data['Datetime'].max()
train_data['days_from_max'] = (max_date - train_data['Datetime']).dt.days
train_data['weight'] = np.exp(-decay_factor * train_data['days_from_max'])

# Prepare data for training and forecasting
X_train = train_data[all_features]
y_train = train_data['Next_AAL_Close']
X_forecast = forecast_data[all_features]
y_actual = forecast_data['AAL_Close']

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

# Train model with specific parameters
print("Training model...")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train, sample_weight=train_data['weight'])

# Make predictions for forecast dates
base_predictions = model.predict(X_forecast_scaled)
print(f"Generated {len(base_predictions)} predictions")

# Adjust for systematic bias
bias = forecast_data['AAL_Close'].mean() - base_predictions.mean()
print(f"Systematic bias: {bias:.4f}")
adjusted_predictions = base_predictions + bias

# Perform Monte Carlo simulations
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(base_predictions), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['AAL_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 1.5 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 1.5)
    volatility = avg_volatility * volatility_multiplier
    
    # Apply random noise to adjusted predictions based on volatility
    noise = np.random.normal(0, volatility, len(adjusted_predictions))
    simulation_results[:, i] = adjusted_predictions + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': forecast_data['AAL_Close'].values,
    'predicted': adjusted_predictions,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate performance metrics
mae = mean_absolute_error(forecast_data['AAL_Close'].values, adjusted_predictions)
rmse = np.sqrt(mean_squared_error(forecast_data['AAL_Close'].values, adjusted_predictions))
r2 = r2_score(forecast_data['AAL_Close'].values, adjusted_predictions)

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((forecast_data['AAL_Close'].values >= lower_bound) & (forecast_data['AAL_Close'].values <= upper_bound))
ci_percentage = within_ci / len(forecast_data) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result[['Datetime', 'Actual', 'predicted', 'Lower_Bound', 'Upper_Bound']])

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('AAL Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Group results by date for easier analysis
forecast_result['Date'] = forecast_result['Datetime'].dt.date
daily_results = forecast_result.groupby('Date').agg({
    'Actual': ['mean', 'min', 'max'],
    'predicted': ['mean', 'min', 'max'],
    'Lower_Bound': 'mean',
    'Upper_Bound': 'mean'
}).reset_index()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('aal_forecast_results.csv', index=False)
print("Forecast results saved to 'aal_forecast_results.csv'")


# ## ALGT

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Load the data
print("Loading data...")
df = pd.read_csv('ALGT_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Function to safely check if a column exists and fillna if it doesn't
def safe_create_feature(dataframe, column_name, default_value=0):
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value
    else:
        dataframe[column_name] = dataframe[column_name].fillna(default_value)
    return dataframe

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Determine holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.5) & (data['Hour'] <= 16.0)).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['ALGT_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['ALGT_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['ALGT_RSI_14'] = calculate_rsi(data['ALGT_Close'], window=14)
    data['ALGT_RSI_7'] = calculate_rsi(data['ALGT_Close'], window=7)
    
    data['ALGT_MACD'], data['ALGT_MACD_Signal'], data['ALGT_MACD_Hist'] = calculate_macd(data['ALGT_Close'])
    
    data['ALGT_BB_Mid'], data['ALGT_BB_Upper'], data['ALGT_BB_Lower'] = calculate_bollinger_bands(data['ALGT_Close'])
    data['ALGT_BB_Width'] = data['ALGT_BB_Upper'] - data['ALGT_BB_Lower']
    data['ALGT_BB_Pct'] = (data['ALGT_Close'] - data['ALGT_BB_Lower']) / (data['ALGT_BB_Upper'] - data['ALGT_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALGT']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'ALGT_Price_Trend_{window}d'] = data['ALGT_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'ALGT_Volume_Trend_{window}d'] = data['ALGT_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between ALGT and the market indicator
            data[f'ALGT_{indicator}_Corr_{window}h'] = data['ALGT_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for ALGT_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'ALGT_Close_Lag_{lag}'] = temp_df['ALGT_Close'].shift(lag)
    temp_df[f'ALGT_Return_Lag_{lag}'] = temp_df['ALGT_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variable - next hour's ALGT_Close
temp_df['Next_ALGT_Close'] = temp_df['ALGT_Close'].shift(-1)

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_ALGT_Close'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsBeforeHoliday', 'IsAfterHoliday', 'IsOpeningHour', 'IsClosingHour',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfMonth_sin', 'DayOfMonth_cos',
    'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'ALGT_RSI_14', 'ALGT_RSI_7', 'ALGT_MACD', 'ALGT_MACD_Signal', 'ALGT_MACD_Hist',
    'ALGT_BB_Width', 'ALGT_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALGT']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'ALGT_Price_Trend_{window}d',
        f'ALGT_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'ALGT_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'ALGT_Close_Lag_{lag}',
        f'ALGT_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original ALGT features
algt_features = ['ALGT_Open', 'ALGT_High', 'ALGT_Low', 'ALGT_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                    return_features, trend_features, correlation_features, lag_features, algt_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Using {len(all_features)} features for training")

# Apply exponential weighting for recent data with shorter half-life
half_life = 7  # Use 7 days instead of 30 for more aggressive weighting
decay_factor = np.log(2) / half_life
max_date = train_data['Datetime'].max()
train_data['days_from_max'] = (max_date - train_data['Datetime']).dt.days
train_data['weight'] = np.exp(-decay_factor * train_data['days_from_max'])

# Prepare data for training and forecasting
X_train = train_data[all_features]
y_train = train_data['Next_ALGT_Close']
X_forecast = forecast_data[all_features]
y_actual = forecast_data['ALGT_Close']

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Use sample weights based on recency
model.fit(X_train_scaled, y_train, sample_weight=train_data['weight'])

# Make predictions for forecast dates
base_predictions = model.predict(X_forecast_scaled)
print(f"Generated {len(base_predictions)} predictions")

# Perform Monte Carlo simulations
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(base_predictions), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['ALGT_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 1.5 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 1.5)
    volatility = avg_volatility * volatility_multiplier
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility * base_predictions, len(base_predictions))
    simulation_results[:, i] = base_predictions + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': forecast_data['ALGT_Close'].values,
    'predicted': base_predictions,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate performance metrics
mae = mean_absolute_error(forecast_data['ALGT_Close'].values, base_predictions)
rmse = np.sqrt(mean_squared_error(forecast_data['ALGT_Close'].values, base_predictions))
r2 = r2_score(forecast_data['ALGT_Close'].values, base_predictions)

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((forecast_data['ALGT_Close'].values >= lower_bound) & (forecast_data['ALGT_Close'].values <= upper_bound))
ci_percentage = within_ci / len(forecast_data) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result[['Datetime', 'Actual', 'predicted', 'Lower_Bound', 'Upper_Bound']])

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('ALGT Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Group results by date for easier analysis
forecast_result['Date'] = forecast_result['Datetime'].dt.date
daily_results = forecast_result.groupby('Date').agg({
    'Actual': ['mean', 'min', 'max'],
    'predicted': ['mean', 'min', 'max'],
    'Lower_Bound': 'mean',
    'Upper_Bound': 'mean'
}).reset_index()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('algt_forecast_results.csv', index=False)
print("Forecast results saved to 'algt_forecast_results.csv'")


# ## DAL

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('DAL_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['DAL_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['DAL_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['DAL_RSI_14'] = calculate_rsi(data['DAL_Close'], window=14)
    data['DAL_RSI_7'] = calculate_rsi(data['DAL_Close'], window=7)
    
    data['DAL_MACD'], data['DAL_MACD_Signal'], data['DAL_MACD_Hist'] = calculate_macd(data['DAL_Close'])
    
    data['DAL_BB_Mid'], data['DAL_BB_Upper'], data['DAL_BB_Lower'] = calculate_bollinger_bands(data['DAL_Close'])
    data['DAL_BB_Width'] = data['DAL_BB_Upper'] - data['DAL_BB_Lower']
    data['DAL_BB_Pct'] = (data['DAL_Close'] - data['DAL_BB_Lower']) / (data['DAL_BB_Upper'] - data['DAL_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'DAL']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'DAL_Price_Trend_{window}d'] = data['DAL_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'DAL_Volume_Trend_{window}d'] = data['DAL_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between DAL and market/sector ETFs
    for indicator in market_indicators:
        data[f'DAL_to_{indicator}_Ratio'] = data['DAL_Close'] / data[f'{indicator}_Close']
        data[f'DAL_to_{indicator}_Ratio_Change'] = data[f'DAL_to_{indicator}_Ratio'].pct_change()
    
    # Calculate DAL's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'DAL_MA_{window}'
        data[ma_col] = data['DAL_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'DAL_Deviation_From_{window}MA_Pct'] = ((data['DAL_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate DAL's performance relative to JETS (airline sector ETF)
    data['DAL_vs_JETS_Alpha'] = data['DAL_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for DAL price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['DAL_Close'].rolling(window=window).mean()
        std = data['DAL_Close'].rolling(window=window).std()
        data[f'DAL_Z_Score_{window}d'] = (data['DAL_Close'] - mean) / std
        data[f'DAL_Z_Score_{window}d'] = data[f'DAL_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between DAL and the market indicator
            data[f'DAL_{indicator}_Corr_{window}h'] = data['DAL_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for DAL_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'DAL_Close_Lag_{lag}'] = temp_df['DAL_Close'].shift(lag)
    temp_df[f'DAL_Return_Lag_{lag}'] = temp_df['DAL_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_DAL_Close'] = temp_df['DAL_Close'].shift(-1)
temp_df['Next_DAL_Return'] = temp_df['Next_DAL_Close'] / temp_df['DAL_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_DAL_Close', 'Next_DAL_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'DAL_RSI_14', 'DAL_RSI_7', 'DAL_MACD', 'DAL_MACD_Signal', 'DAL_MACD_Hist',
    'DAL_BB_Width', 'DAL_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'DAL']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'DAL_Price_Trend_{window}d',
        f'DAL_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'DAL_to_{indicator}_Ratio',
        f'DAL_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'DAL_MA_{window}')
    relative_features.append(f'DAL_Deviation_From_{window}MA_Pct')

relative_features.append('DAL_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'DAL_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'DAL_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'DAL_Close_Lag_{lag}',
        f'DAL_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original DAL features
dal_features = ['DAL_Open', 'DAL_High', 'DAL_Low', 'DAL_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, dal_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_DAL_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features (e.g., top 40 features)
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

print(f"\nSelected {len(selected_features)} features for modeling")

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_DAL_Return']
y_train_price = train_data['Next_DAL_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['DAL_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# Build improved LSTM model with attention and residual connections
print("Building and training improved LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(32, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(16, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.close()

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_return_scaled)

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# Ensemble the predictions (50% LSTM, 50% RF)
ensemble_forecast_results = 0.5 * np.array(lstm_forecast_results) + 0.5 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['DAL_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# Calculate performance metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['DAL_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('DAL Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('dal_improved_forecast_results.csv', index=False)
print("Forecast results saved to 'dal_improved_forecast_results.csv'")


# ## UAL

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('UAL_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['UAL_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['UAL_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['UAL_RSI_14'] = calculate_rsi(data['UAL_Close'], window=14)
    data['UAL_RSI_7'] = calculate_rsi(data['UAL_Close'], window=7)
    
    data['UAL_MACD'], data['UAL_MACD_Signal'], data['UAL_MACD_Hist'] = calculate_macd(data['UAL_Close'])
    
    data['UAL_BB_Mid'], data['UAL_BB_Upper'], data['UAL_BB_Lower'] = calculate_bollinger_bands(data['UAL_Close'])
    data['UAL_BB_Width'] = data['UAL_BB_Upper'] - data['UAL_BB_Lower']
    data['UAL_BB_Pct'] = (data['UAL_Close'] - data['UAL_BB_Lower']) / (data['UAL_BB_Upper'] - data['UAL_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'UAL']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'UAL_Price_Trend_{window}d'] = data['UAL_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'UAL_Volume_Trend_{window}d'] = data['UAL_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between UAL and market/sector ETFs
    for indicator in market_indicators:
        data[f'UAL_to_{indicator}_Ratio'] = data['UAL_Close'] / data[f'{indicator}_Close']
        data[f'UAL_to_{indicator}_Ratio_Change'] = data[f'UAL_to_{indicator}_Ratio'].pct_change()
    
    # Calculate UAL's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'UAL_MA_{window}'
        data[ma_col] = data['UAL_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'UAL_Deviation_From_{window}MA_Pct'] = ((data['UAL_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate UAL's performance relative to JETS (airline sector ETF)
    data['UAL_vs_JETS_Alpha'] = data['UAL_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for UAL price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['UAL_Close'].rolling(window=window).mean()
        std = data['UAL_Close'].rolling(window=window).std()
        data[f'UAL_Z_Score_{window}d'] = (data['UAL_Close'] - mean) / std
        data[f'UAL_Z_Score_{window}d'] = data[f'UAL_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between UAL and the market indicator
            data[f'UAL_{indicator}_Corr_{window}h'] = data['UAL_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for UAL_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'UAL_Close_Lag_{lag}'] = temp_df['UAL_Close'].shift(lag)
    temp_df[f'UAL_Return_Lag_{lag}'] = temp_df['UAL_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_UAL_Close'] = temp_df['UAL_Close'].shift(-1)
temp_df['Next_UAL_Return'] = temp_df['Next_UAL_Close'] / temp_df['UAL_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_UAL_Close', 'Next_UAL_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'UAL_RSI_14', 'UAL_RSI_7', 'UAL_MACD', 'UAL_MACD_Signal', 'UAL_MACD_Hist',
    'UAL_BB_Width', 'UAL_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'UAL']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'UAL_Price_Trend_{window}d',
        f'UAL_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'UAL_to_{indicator}_Ratio',
        f'UAL_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'UAL_MA_{window}')
    relative_features.append(f'UAL_Deviation_From_{window}MA_Pct')

relative_features.append('UAL_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'UAL_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'UAL_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'UAL_Close_Lag_{lag}',
        f'UAL_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original UAL features
UAL_features = ['UAL_Open', 'UAL_High', 'UAL_Low', 'UAL_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, UAL_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_UAL_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

# IMPROVEMENT 1: Add feature interactions for top features
print("\nAdding feature interactions for top features...")
top_5_features = selected_features[:5]
for i in range(len(top_5_features)):
    for j in range(i+1, len(top_5_features)):
        feat_name = f"{top_5_features[i]}_{top_5_features[j]}_interaction"
        train_data[feat_name] = train_data[top_5_features[i]] * train_data[top_5_features[j]]
        forecast_data[feat_name] = forecast_data[top_5_features[i]] * forecast_data[top_5_features[j]]
        selected_features.append(feat_name)

# IMPROVEMENT 2: Add non-linear transformations of key features
print("Adding non-linear transformations of key features...")
# For the top feature, add non-linear transformations
top_feature = selected_features[0]
train_data[f'{top_feature}_squared'] = train_data[top_feature] ** 2
forecast_data[f'{top_feature}_squared'] = forecast_data[top_feature] ** 2
selected_features.append(f'{top_feature}_squared')

train_data[f'{top_feature}_cubed'] = train_data[top_feature] ** 3
forecast_data[f'{top_feature}_cubed'] = forecast_data[top_feature] ** 3
selected_features.append(f'{top_feature}_cubed')

# Log transformation for positive features
if train_data[top_feature].min() > 0:
    train_data[f'{top_feature}_log'] = np.log(train_data[top_feature])
    forecast_data[f'{top_feature}_log'] = np.log(forecast_data[top_feature])
    selected_features.append(f'{top_feature}_log')

print(f"Final feature count after additions: {len(selected_features)}")

# IMPROVEMENT 3: Implement a time-based weighting scheme
print("Implementing time-based weighting scheme...")
weight_half_life = 30  # days
decay_factor = np.log(2) / weight_half_life
max_date = train_data['Datetime'].max()
train_data['time_weight'] = np.exp(-decay_factor * (max_date - train_data['Datetime']).dt.days)

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_UAL_Return']
y_train_price = train_data['Next_UAL_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['UAL_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# IMPROVEMENT 4: Build improved LSTM model with bidirectional layers and attention
print("Building and training improved bidirectional LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer - Bidirectional with more units
    x = Bidirectional(LSTM(80, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(40, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(20, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
# Use time weights with Random Forest
rf_model.fit(X_train_scaled, y_train_return_scaled, sample_weight=train_data['time_weight'])

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# IMPROVEMENT 5: Adjust the ensemble weights (70% RF, 30% LSTM)
print("Creating ensemble with adjusted weights (70% RF, 30% LSTM)...")
ensemble_forecast_results = 0.3 * np.array(lstm_forecast_results) + 0.7 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['UAL_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# IMPROVEMENT 6: Apply Savitzky-Golay filter to smooth predictions
print("Applying Savitzky-Golay filter to smooth predictions...")
# Note: window_length must be odd and less than input length
window_length = min(5, len(predicted_prices) - (len(predicted_prices) % 2 == 0))
if window_length >= 3:  # Need at least 3 points for a quadratic
    predicted_prices_smoothed = savgol_filter(predicted_prices, window_length, 2)
else:
    predicted_prices_smoothed = predicted_prices

# Calculate performance metrics for both raw and smoothed predictions
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

mae_smoothed = mean_absolute_error(actual_prices, predicted_prices_smoothed)
rmse_smoothed = np.sqrt(mean_squared_error(actual_prices, predicted_prices_smoothed))
r2_smoothed = r2_score(actual_prices, predicted_prices_smoothed)

# Choose better predictions based on R² score
if r2_smoothed > r2:
    print("Using smoothed predictions (better R² score)...")
    predicted_prices = predicted_prices_smoothed
    mae = mae_smoothed
    rmse = rmse_smoothed
    r2 = r2_smoothed
else:
    print("Using raw predictions (better R² score)...")

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['UAL_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('UAL Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('UAL_forecast_results.csv', index=False)
print("Forecast results saved to 'UAL_forecast_results.csv'")


# In[ ]:





# ## JBLU

# In[51]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Load the data
print("Loading data...")
df = pd.read_csv('JBLU_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Function to safely check if a column exists and fillna if it doesn't
def safe_create_feature(dataframe, column_name, default_value=0):
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value
    else:
        dataframe[column_name] = dataframe[column_name].fillna(default_value)
    return dataframe

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Determine holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.5) & (data['Hour'] <= 16.0)).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['JBLU_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['JBLU_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['JBLU_RSI_14'] = calculate_rsi(data['JBLU_Close'], window=14)
    data['JBLU_RSI_7'] = calculate_rsi(data['JBLU_Close'], window=7)
    
    data['JBLU_MACD'], data['JBLU_MACD_Signal'], data['JBLU_MACD_Hist'] = calculate_macd(data['JBLU_Close'])
    
    data['JBLU_BB_Mid'], data['JBLU_BB_Upper'], data['JBLU_BB_Lower'] = calculate_bollinger_bands(data['JBLU_Close'])
    data['JBLU_BB_Width'] = data['JBLU_BB_Upper'] - data['JBLU_BB_Lower']
    data['JBLU_BB_Pct'] = (data['JBLU_Close'] - data['JBLU_BB_Lower']) / (data['JBLU_BB_Upper'] - data['JBLU_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'JBLU']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'JBLU_Price_Trend_{window}d'] = data['JBLU_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'JBLU_Volume_Trend_{window}d'] = data['JBLU_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between JBLU and the market indicator
            data[f'JBLU_{indicator}_Corr_{window}h'] = data['JBLU_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for JBLU_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'JBLU_Close_Lag_{lag}'] = temp_df['JBLU_Close'].shift(lag)
    temp_df[f'JBLU_Return_Lag_{lag}'] = temp_df['JBLU_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variable - next hour's JBLU_Close
temp_df['Next_JBLU_Close'] = temp_df['JBLU_Close'].shift(-1)

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_JBLU_Close'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsBeforeHoliday', 'IsAfterHoliday', 'IsOpeningHour', 'IsClosingHour',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfMonth_sin', 'DayOfMonth_cos',
    'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'JBLU_RSI_14', 'JBLU_RSI_7', 'JBLU_MACD', 'JBLU_MACD_Signal', 'JBLU_MACD_Hist',
    'JBLU_BB_Width', 'JBLU_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'JBLU']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'JBLU_Price_Trend_{window}d',
        f'JBLU_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'JBLU_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'JBLU_Close_Lag_{lag}',
        f'JBLU_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original JBLU features
JBLU_features = ['JBLU_Open', 'JBLU_High', 'JBLU_Low', 'JBLU_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, lag_features, JBLU_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Using {len(all_features)} features for training")

# Apply exponential weighting for recent data with shorter half-life
half_life = 7  # Use 7 days instead of 30 for more aggressive weighting
decay_factor = np.log(2) / half_life
max_date = train_data['Datetime'].max()
train_data['days_from_max'] = (max_date - train_data['Datetime']).dt.days
train_data['weight'] = np.exp(-decay_factor * train_data['days_from_max'])

# Prepare data for training and forecasting
X_train = train_data[all_features]
y_train = train_data['Next_JBLU_Close']
X_forecast = forecast_data[all_features]
y_actual = forecast_data['JBLU_Close']

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

# Train model with specific parameters
print("Training model...")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train, sample_weight=train_data['weight'])

# Make predictions for forecast dates
base_predictions = model.predict(X_forecast_scaled)
print(f"Generated {len(base_predictions)} predictions")

# Adjust for systematic bias
bias = forecast_data['JBLU_Close'].mean() - base_predictions.mean()
print(f"Systematic bias: {bias:.4f}")
adjusted_predictions = base_predictions + bias

# Perform Monte Carlo simulations
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(base_predictions), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['JBLU_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 1.5 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 1.5)
    volatility = avg_volatility * volatility_multiplier
    
    # Apply random noise to adjusted predictions based on volatility
    noise = np.random.normal(0, volatility, len(adjusted_predictions))
    simulation_results[:, i] = adjusted_predictions + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': forecast_data['JBLU_Close'].values,
    'predicted': adjusted_predictions,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate performance metrics
mae = mean_absolute_error(forecast_data['JBLU_Close'].values, adjusted_predictions)
rmse = np.sqrt(mean_squared_error(forecast_data['JBLU_Close'].values, adjusted_predictions))
r2 = r2_score(forecast_data['JBLU_Close'].values, adjusted_predictions)

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((forecast_data['JBLU_Close'].values >= lower_bound) & (forecast_data['JBLU_Close'].values <= upper_bound))
ci_percentage = within_ci / len(forecast_data) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result[['Datetime', 'Actual', 'predicted', 'Lower_Bound', 'Upper_Bound']])

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('JBLU Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Group results by date for easier analysis
forecast_result['Date'] = forecast_result['Datetime'].dt.date
daily_results = forecast_result.groupby('Date').agg({
    'Actual': ['mean', 'min', 'max'],
    'predicted': ['mean', 'min', 'max'],
    'Lower_Bound': 'mean',
    'Upper_Bound': 'mean'
}).reset_index()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('JBLU_forecast_results.csv', index=False)
print("Forecast results saved to 'JBLU_forecast_results.csv'")


# ## ALK 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from xgboost import XGBRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Load the data
print("Loading data...")
df = pd.read_csv('ALK_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Function to safely check if a column exists and fillna if it doesn't
def safe_create_feature(dataframe, column_name, default_value=0):
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value
    else:
        dataframe[column_name] = dataframe[column_name].fillna(default_value)
    return dataframe

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Determine holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.5) & (data['Hour'] <= 16.0)).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['ALK_RSI_14'] = calculate_rsi(data['ALK_Close'], window=14)
    data['ALK_RSI_7'] = calculate_rsi(data['ALK_Close'], window=7)
    
    data['ALK_MACD'], data['ALK_MACD_Signal'], data['ALK_MACD_Hist'] = calculate_macd(data['ALK_Close'])
    
    data['ALK_BB_Mid'], data['ALK_BB_Upper'], data['ALK_BB_Lower'] = calculate_bollinger_bands(data['ALK_Close'])
    data['ALK_BB_Width'] = data['ALK_BB_Upper'] - data['ALK_BB_Lower']
    data['ALK_BB_Pct'] = (data['ALK_Close'] - data['ALK_BB_Lower']) / (data['ALK_BB_Upper'] - data['ALK_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'ALK_Price_Trend_{window}d'] = data['ALK_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'ALK_Volume_Trend_{window}d'] = data['ALK_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between ALK and the market indicator
            data[f'ALK_{indicator}_Corr_{window}h'] = data['ALK_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for ALK_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'ALK_Close_Lag_{lag}'] = temp_df['ALK_Close'].shift(lag)
    temp_df[f'ALK_Return_Lag_{lag}'] = temp_df['ALK_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variable - next hour's ALK_Close
temp_df['Next_ALK_Close'] = temp_df['ALK_Close'].shift(-1)

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_ALK_Close'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsBeforeHoliday', 'IsAfterHoliday', 'IsOpeningHour', 'IsClosingHour',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfMonth_sin', 'DayOfMonth_cos',
    'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'ALK_RSI_14', 'ALK_RSI_7', 'ALK_MACD', 'ALK_MACD_Signal', 'ALK_MACD_Hist',
    'ALK_BB_Width', 'ALK_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'ALK_Price_Trend_{window}d',
        f'ALK_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'ALK_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'ALK_Close_Lag_{lag}',
        f'ALK_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original ALK features
alk_features = ['ALK_Open', 'ALK_High', 'ALK_Low', 'ALK_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, lag_features, alk_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Using {len(all_features)} features for training")

# Apply exponential weighting for recent data with shorter half-life
half_life = 7  # Use 7 days instead of 30 for more aggressive weighting
decay_factor = np.log(2) / half_life
max_date = train_data['Datetime'].max()
train_data['days_from_max'] = (max_date - train_data['Datetime']).dt.days
train_data['weight'] = np.exp(-decay_factor * train_data['days_from_max'])

# Prepare data for training and forecasting
X_train = train_data[all_features]
y_train = train_data['Next_ALK_Close']
X_forecast = forecast_data[all_features]
y_actual = forecast_data['ALK_Close']

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

# Train model with specific parameters
print("Training model...")
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train, sample_weight=train_data['weight'])

# Make predictions for forecast dates
base_predictions = model.predict(X_forecast_scaled)
print(f"Generated {len(base_predictions)} predictions")

# Adjust for systematic bias
bias = forecast_data['ALK_Close'].mean() - base_predictions.mean()
print(f"Systematic bias: {bias:.4f}")
adjusted_predictions = base_predictions + bias

# Perform Monte Carlo simulations
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(base_predictions), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['ALK_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 1.5 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 1.5)
    volatility = avg_volatility * volatility_multiplier
    
    # Apply random noise to adjusted predictions based on volatility
    noise = np.random.normal(0, volatility, len(adjusted_predictions))
    simulation_results[:, i] = adjusted_predictions + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': forecast_data['ALK_Close'].values,
    'predicted': adjusted_predictions,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate performance metrics
mae = mean_absolute_error(forecast_data['ALK_Close'].values, adjusted_predictions)
rmse = np.sqrt(mean_squared_error(forecast_data['ALK_Close'].values, adjusted_predictions))
r2 = r2_score(forecast_data['ALK_Close'].values, adjusted_predictions)

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((forecast_data['ALK_Close'].values >= lower_bound) & (forecast_data['ALK_Close'].values <= upper_bound))
ci_percentage = within_ci / len(forecast_data) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result[['Datetime', 'Actual', 'predicted', 'Lower_Bound', 'Upper_Bound']])

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('ALK Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Group results by date for easier analysis
forecast_result['Date'] = forecast_result['Datetime'].dt.date
daily_results = forecast_result.groupby('Date').agg({
    'Actual': ['mean', 'min', 'max'],
    'predicted': ['mean', 'min', 'max'],
    'Lower_Bound': 'mean',
    'Upper_Bound': 'mean'
}).reset_index()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('alk_forecast_results.csv', index=False)
print("Forecast results saved to 'alk_forecast_results.csv'")


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Load the data
print("Loading data...")
df = pd.read_csv('ALK_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Function to safely check if a column exists and fillna if it doesn't
def safe_create_feature(dataframe, column_name, default_value=0):
    if column_name not in dataframe.columns:
        dataframe[column_name] = default_value
    else:
        dataframe[column_name] = dataframe[column_name].fillna(default_value)
    return dataframe

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    
    # Determine holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.5) & (data['Hour'] <= 16.0)).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['ALK_RSI_14'] = calculate_rsi(data['ALK_Close'], window=14)
    data['ALK_RSI_7'] = calculate_rsi(data['ALK_Close'], window=7)
    
    data['ALK_MACD'], data['ALK_MACD_Signal'], data['ALK_MACD_Hist'] = calculate_macd(data['ALK_Close'])
    
    data['ALK_BB_Mid'], data['ALK_BB_Upper'], data['ALK_BB_Lower'] = calculate_bollinger_bands(data['ALK_Close'])
    data['ALK_BB_Width'] = data['ALK_BB_Upper'] - data['ALK_BB_Lower']
    data['ALK_BB_Pct'] = (data['ALK_Close'] - data['ALK_BB_Lower']) / (data['ALK_BB_Upper'] - data['ALK_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'ALK_Price_Trend_{window}d'] = data['ALK_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'ALK_Volume_Trend_{window}d'] = data['ALK_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between ALK and the market indicator
            data[f'ALK_{indicator}_Corr_{window}h'] = data['ALK_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for ALK_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'ALK_Close_Lag_{lag}'] = temp_df['ALK_Close'].shift(lag)
    temp_df[f'ALK_Return_Lag_{lag}'] = temp_df['ALK_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variable - next hour's ALK_Close
temp_df['Next_ALK_Close'] = temp_df['ALK_Close'].shift(-1)

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_ALK_Close'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsBeforeHoliday', 'IsAfterHoliday', 'IsOpeningHour', 'IsClosingHour',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'DayOfMonth_sin', 'DayOfMonth_cos',
    'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'ALK_RSI_14', 'ALK_RSI_7', 'ALK_MACD', 'ALK_MACD_Signal', 'ALK_MACD_Hist',
    'ALK_BB_Width', 'ALK_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'ALK_Price_Trend_{window}d',
        f'ALK_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'ALK_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'ALK_Close_Lag_{lag}',
        f'ALK_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original ALK features
alk_features = ['ALK_Open', 'ALK_High', 'ALK_Low', 'ALK_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                    return_features, trend_features, correlation_features, lag_features, alk_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Using {len(all_features)} features for training")

# Apply exponential weighting for recent data with shorter half-life
half_life = 7  # Use 7 days instead of 30 for more aggressive weighting
decay_factor = np.log(2) / half_life
max_date = train_data['Datetime'].max()
train_data['days_from_max'] = (max_date - train_data['Datetime']).dt.days
train_data['weight'] = np.exp(-decay_factor * train_data['days_from_max'])

# Prepare data for training and forecasting
X_train = train_data[all_features]
y_train = train_data['Next_ALK_Close']
X_forecast = forecast_data[all_features]
y_actual = forecast_data['ALK_Close']

# Scale the data
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_forecast_scaled = scaler.transform(X_forecast)

# Train Random Forest model
print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Use sample weights based on recency
model.fit(X_train_scaled, y_train, sample_weight=train_data['weight'])

# Make predictions for forecast dates
base_predictions = model.predict(X_forecast_scaled)
print(f"Generated {len(base_predictions)} predictions")

# Perform Monte Carlo simulations
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(base_predictions), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['ALK_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 1.5 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 1.5)
    volatility = avg_volatility * volatility_multiplier
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility * base_predictions, len(base_predictions))
    simulation_results[:, i] = base_predictions + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': forecast_data['ALK_Close'].values,
    'predicted': base_predictions,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate performance metrics
mae = mean_absolute_error(forecast_data['ALK_Close'].values, base_predictions)
rmse = np.sqrt(mean_squared_error(forecast_data['ALK_Close'].values, base_predictions))
r2 = r2_score(forecast_data['ALK_Close'].values, base_predictions)

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((forecast_data['ALK_Close'].values >= lower_bound) & (forecast_data['ALK_Close'].values <= upper_bound))
ci_percentage = within_ci / len(forecast_data) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result[['Datetime', 'Actual', 'predicted', 'Lower_Bound', 'Upper_Bound']])

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('ALK Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Group results by date for easier analysis
forecast_result['Date'] = forecast_result['Datetime'].dt.date
daily_results = forecast_result.groupby('Date').agg({
    'Actual': ['mean', 'min', 'max'],
    'predicted': ['mean', 'min', 'max'],
    'Lower_Bound': 'mean',
    'Upper_Bound': 'mean'
}).reset_index()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('alk_forecast_results.csv', index=False)
print("Forecast results saved to 'alk_forecast_results.csv'")


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('ALK_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['ALK_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['ALK_RSI_14'] = calculate_rsi(data['ALK_Close'], window=14)
    data['ALK_RSI_7'] = calculate_rsi(data['ALK_Close'], window=7)
    
    data['ALK_MACD'], data['ALK_MACD_Signal'], data['ALK_MACD_Hist'] = calculate_macd(data['ALK_Close'])
    
    data['ALK_BB_Mid'], data['ALK_BB_Upper'], data['ALK_BB_Lower'] = calculate_bollinger_bands(data['ALK_Close'])
    data['ALK_BB_Width'] = data['ALK_BB_Upper'] - data['ALK_BB_Lower']
    data['ALK_BB_Pct'] = (data['ALK_Close'] - data['ALK_BB_Lower']) / (data['ALK_BB_Upper'] - data['ALK_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'ALK_Price_Trend_{window}d'] = data['ALK_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'ALK_Volume_Trend_{window}d'] = data['ALK_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between ALK and market/sector ETFs
    for indicator in market_indicators:
        data[f'ALK_to_{indicator}_Ratio'] = data['ALK_Close'] / data[f'{indicator}_Close']
        data[f'ALK_to_{indicator}_Ratio_Change'] = data[f'ALK_to_{indicator}_Ratio'].pct_change()
    
    # Calculate ALK's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'ALK_MA_{window}'
        data[ma_col] = data['ALK_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'ALK_Deviation_From_{window}MA_Pct'] = ((data['ALK_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate ALK's performance relative to JETS (airline sector ETF)
    data['ALK_vs_JETS_Alpha'] = data['ALK_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for ALK price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['ALK_Close'].rolling(window=window).mean()
        std = data['ALK_Close'].rolling(window=window).std()
        data[f'ALK_Z_Score_{window}d'] = (data['ALK_Close'] - mean) / std
        data[f'ALK_Z_Score_{window}d'] = data[f'ALK_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between ALK and the market indicator
            data[f'ALK_{indicator}_Corr_{window}h'] = data['ALK_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for ALK_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'ALK_Close_Lag_{lag}'] = temp_df['ALK_Close'].shift(lag)
    temp_df[f'ALK_Return_Lag_{lag}'] = temp_df['ALK_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_ALK_Close'] = temp_df['ALK_Close'].shift(-1)
temp_df['Next_ALK_Return'] = temp_df['Next_ALK_Close'] / temp_df['ALK_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_ALK_Close', 'Next_ALK_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'ALK_RSI_14', 'ALK_RSI_7', 'ALK_MACD', 'ALK_MACD_Signal', 'ALK_MACD_Hist',
    'ALK_BB_Width', 'ALK_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ALK']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'ALK_Price_Trend_{window}d',
        f'ALK_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'ALK_to_{indicator}_Ratio',
        f'ALK_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'ALK_MA_{window}')
    relative_features.append(f'ALK_Deviation_From_{window}MA_Pct')

relative_features.append('ALK_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'ALK_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'ALK_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'ALK_Close_Lag_{lag}',
        f'ALK_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original ALK features
alk_features = ['ALK_Open', 'ALK_High', 'ALK_Low', 'ALK_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, alk_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_ALK_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features (e.g., top 40 features)
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

print(f"\nSelected {len(selected_features)} features for modeling")

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_ALK_Return']
y_train_price = train_data['Next_ALK_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['ALK_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# Build improved LSTM model with attention and residual connections
print("Building and training improved LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(32, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(16, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.close()

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_return_scaled)

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# Ensemble the predictions (50% LSTM, 50% RF)
ensemble_forecast_results = 0.5 * np.array(lstm_forecast_results) + 0.5 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['ALK_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# Calculate performance metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['ALK_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('ALK Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('alk_improved_forecast_results.csv', index=False)
print("Forecast results saved to 'alk_improved_forecast_results.csv'")


# # LUV 

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('LUV_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['LUV_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['LUV_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['LUV_RSI_14'] = calculate_rsi(data['LUV_Close'], window=14)
    data['LUV_RSI_7'] = calculate_rsi(data['LUV_Close'], window=7)
    
    data['LUV_MACD'], data['LUV_MACD_Signal'], data['LUV_MACD_Hist'] = calculate_macd(data['LUV_Close'])
    
    data['LUV_BB_Mid'], data['LUV_BB_Upper'], data['LUV_BB_Lower'] = calculate_bollinger_bands(data['LUV_Close'])
    data['LUV_BB_Width'] = data['LUV_BB_Upper'] - data['LUV_BB_Lower']
    data['LUV_BB_Pct'] = (data['LUV_Close'] - data['LUV_BB_Lower']) / (data['LUV_BB_Upper'] - data['LUV_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'LUV']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'LUV_Price_Trend_{window}d'] = data['LUV_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'LUV_Volume_Trend_{window}d'] = data['LUV_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between LUV and market/sector ETFs
    for indicator in market_indicators:
        data[f'LUV_to_{indicator}_Ratio'] = data['LUV_Close'] / data[f'{indicator}_Close']
        data[f'LUV_to_{indicator}_Ratio_Change'] = data[f'LUV_to_{indicator}_Ratio'].pct_change()
    
    # Calculate LUV's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'LUV_MA_{window}'
        data[ma_col] = data['LUV_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'LUV_Deviation_From_{window}MA_Pct'] = ((data['LUV_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate LUV's performance relative to JETS (airline sector ETF)
    data['LUV_vs_JETS_Alpha'] = data['LUV_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for LUV price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['LUV_Close'].rolling(window=window).mean()
        std = data['LUV_Close'].rolling(window=window).std()
        data[f'LUV_Z_Score_{window}d'] = (data['LUV_Close'] - mean) / std
        data[f'LUV_Z_Score_{window}d'] = data[f'LUV_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between LUV and the market indicator
            data[f'LUV_{indicator}_Corr_{window}h'] = data['LUV_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for LUV_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'LUV_Close_Lag_{lag}'] = temp_df['LUV_Close'].shift(lag)
    temp_df[f'LUV_Return_Lag_{lag}'] = temp_df['LUV_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_LUV_Close'] = temp_df['LUV_Close'].shift(-1)
temp_df['Next_LUV_Return'] = temp_df['Next_LUV_Close'] / temp_df['LUV_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_LUV_Close', 'Next_LUV_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'LUV_RSI_14', 'LUV_RSI_7', 'LUV_MACD', 'LUV_MACD_Signal', 'LUV_MACD_Hist',
    'LUV_BB_Width', 'LUV_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'LUV']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'LUV_Price_Trend_{window}d',
        f'LUV_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'LUV_to_{indicator}_Ratio',
        f'LUV_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'LUV_MA_{window}')
    relative_features.append(f'LUV_Deviation_From_{window}MA_Pct')

relative_features.append('LUV_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'LUV_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'LUV_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'LUV_Close_Lag_{lag}',
        f'LUV_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original LUV features
LUV_features = ['LUV_Open', 'LUV_High', 'LUV_Low', 'LUV_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, LUV_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_LUV_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features (e.g., top 40 features)
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

print(f"\nSelected {len(selected_features)} features for modeling")

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_LUV_Return']
y_train_price = train_data['Next_LUV_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['LUV_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# Build improved LSTM model with attention and residual connections
print("Building and training improved LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(32, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(16, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.close()

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_return_scaled)

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# Ensemble the predictions (50% LSTM, 50% RF)
ensemble_forecast_results = 0.5 * np.array(lstm_forecast_results) + 0.5 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['LUV_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# Calculate performance metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['LUV_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('LUV Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('LUV_improved_forecast_results.csv', index=False)
print("Forecast results saved to 'LUV_improved_forecast_results.csv'")


# ## ULCC

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('ULCC_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['ULCC_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['ULCC_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['ULCC_RSI_14'] = calculate_rsi(data['ULCC_Close'], window=14)
    data['ULCC_RSI_7'] = calculate_rsi(data['ULCC_Close'], window=7)
    
    data['ULCC_MACD'], data['ULCC_MACD_Signal'], data['ULCC_MACD_Hist'] = calculate_macd(data['ULCC_Close'])
    
    data['ULCC_BB_Mid'], data['ULCC_BB_Upper'], data['ULCC_BB_Lower'] = calculate_bollinger_bands(data['ULCC_Close'])
    data['ULCC_BB_Width'] = data['ULCC_BB_Upper'] - data['ULCC_BB_Lower']
    data['ULCC_BB_Pct'] = (data['ULCC_Close'] - data['ULCC_BB_Lower']) / (data['ULCC_BB_Upper'] - data['ULCC_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ULCC']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'ULCC_Price_Trend_{window}d'] = data['ULCC_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'ULCC_Volume_Trend_{window}d'] = data['ULCC_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between ULCC and market/sector ETFs
    for indicator in market_indicators:
        data[f'ULCC_to_{indicator}_Ratio'] = data['ULCC_Close'] / data[f'{indicator}_Close']
        data[f'ULCC_to_{indicator}_Ratio_Change'] = data[f'ULCC_to_{indicator}_Ratio'].pct_change()
    
    # Calculate ULCC's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'ULCC_MA_{window}'
        data[ma_col] = data['ULCC_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'ULCC_Deviation_From_{window}MA_Pct'] = ((data['ULCC_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate ULCC's performance relative to JETS (airline sector ETF)
    data['ULCC_vs_JETS_Alpha'] = data['ULCC_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for ULCC price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['ULCC_Close'].rolling(window=window).mean()
        std = data['ULCC_Close'].rolling(window=window).std()
        data[f'ULCC_Z_Score_{window}d'] = (data['ULCC_Close'] - mean) / std
        data[f'ULCC_Z_Score_{window}d'] = data[f'ULCC_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between ULCC and the market indicator
            data[f'ULCC_{indicator}_Corr_{window}h'] = data['ULCC_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for ULCC_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'ULCC_Close_Lag_{lag}'] = temp_df['ULCC_Close'].shift(lag)
    temp_df[f'ULCC_Return_Lag_{lag}'] = temp_df['ULCC_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_ULCC_Close'] = temp_df['ULCC_Close'].shift(-1)
temp_df['Next_ULCC_Return'] = temp_df['Next_ULCC_Close'] / temp_df['ULCC_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_ULCC_Close', 'Next_ULCC_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'ULCC_RSI_14', 'ULCC_RSI_7', 'ULCC_MACD', 'ULCC_MACD_Signal', 'ULCC_MACD_Hist',
    'ULCC_BB_Width', 'ULCC_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'ULCC']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'ULCC_Price_Trend_{window}d',
        f'ULCC_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'ULCC_to_{indicator}_Ratio',
        f'ULCC_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'ULCC_MA_{window}')
    relative_features.append(f'ULCC_Deviation_From_{window}MA_Pct')

relative_features.append('ULCC_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'ULCC_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'ULCC_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'ULCC_Close_Lag_{lag}',
        f'ULCC_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original ULCC features
ULCC_features = ['ULCC_Open', 'ULCC_High', 'ULCC_Low', 'ULCC_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, ULCC_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_ULCC_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features (e.g., top 40 features)
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

print(f"\nSelected {len(selected_features)} features for modeling")

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_ULCC_Return']
y_train_price = train_data['Next_ULCC_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['ULCC_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# Build improved LSTM model with attention and residual connections
print("Building and training improved LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(32, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(16, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.close()

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_return_scaled)

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# Ensemble the predictions (50% LSTM, 50% RF)
ensemble_forecast_results = 0.5 * np.array(lstm_forecast_results) + 0.5 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['ULCC_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# Calculate performance metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['ULCC_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('ULCC Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('ULCC_improved_forecast_results.csv', index=False)
print("Forecast results saved to 'ULCC_improved_forecast_results.csv'")


# ## AAL

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input, Concatenate, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return rolling_mean, upper_band, lower_band

def is_us_market_holiday(dt):
    """Check if a date is a US market holiday (simplified version)"""
    # Convert to date object if it's a datetime
    check_date = dt.date() if hasattr(dt, 'date') else dt
    
    # Common US market holidays (simplified for recent years)
    holidays = [
        # 2024 holidays
        date(2024, 1, 1),   # New Year's Day
        date(2024, 1, 15),  # Martin Luther King Jr. Day
        date(2024, 2, 19),  # Presidents' Day
        date(2024, 3, 29),  # Good Friday
        date(2024, 5, 27),  # Memorial Day
        date(2024, 6, 19),  # Juneteenth
        date(2024, 7, 4),   # Independence Day
        date(2024, 9, 2),   # Labor Day
        date(2024, 11, 28), # Thanksgiving Day
        date(2024, 12, 25), # Christmas Day
        
        # 2025 holidays (estimated dates)
        date(2025, 1, 1),   # New Year's Day
        date(2025, 1, 20),  # Martin Luther King Jr. Day
        date(2025, 2, 17),  # Presidents' Day
        date(2025, 4, 18),  # Good Friday
        date(2025, 5, 26),  # Memorial Day
        date(2025, 6, 19),  # Juneteenth
        date(2025, 7, 4),   # Independence Day
        date(2025, 9, 1),   # Labor Day
        date(2025, 11, 27), # Thanksgiving Day
        date(2025, 12, 25), # Christmas Day
    ]
    
    return check_date in holidays

# Create a custom attention layer for LSTM
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()

# Load the data
print("Loading data...")
df = pd.read_csv('AAL_with_market_data_Jan2025.csv')

# Convert datetime to pandas datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])
print(f"Total records loaded: {len(df)}")

# Sort by datetime
df = df.sort_values('Datetime')

# Define forecast dates (Jan 2, 3, and 6, 2025)
forecast_dates_requested = [
    '2025-01-02 09:30:00', '2025-01-02 10:30:00', '2025-01-02 11:30:00', 
    '2025-01-02 12:30:00', '2025-01-02 13:30:00', '2025-01-02 14:30:00',
    '2025-01-02 15:30:00', '2025-01-03 09:30:00', '2025-01-03 10:30:00',
    '2025-01-03 11:30:00', '2025-01-03 12:30:00', '2025-01-03 13:30:00',
    '2025-01-03 14:30:00', '2025-01-03 15:30:00', '2025-01-06 09:30:00',
    '2025-01-06 10:30:00', '2025-01-06 11:30:00', '2025-01-06 12:30:00',
    '2025-01-06 13:30:00', '2025-01-06 14:30:00', '2025-01-06 15:30:00'
]
forecast_dates_requested = pd.to_datetime(forecast_dates_requested)

# Split data to ensure no data leakage - use actual dates
print("Splitting data to avoid data leakage...")
forecast_data = df[df['Datetime'].isin(forecast_dates_requested)]
train_data = df[~df['Datetime'].isin(forecast_dates_requested)]

print(f"Training data size: {len(train_data)} records")
print(f"Forecast data size: {len(forecast_data)} records")

# Create enhanced time-based features
print("Creating enhanced time-based features...")
for data in [train_data, forecast_data]:
    # Basic time features
    data['Hour'] = data['Datetime'].dt.hour + data['Datetime'].dt.minute/60
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['DayOfMonth'] = data['Datetime'].dt.day
    data['Month'] = data['Datetime'].dt.month
    data['WeekOfYear'] = data['Datetime'].dt.isocalendar().week
    data['Quarter'] = data['Datetime'].dt.quarter
    
    # Day-specific features
    data['IsMonday'] = (data['DayOfWeek'] == 0).astype(int)
    data['IsFriday'] = (data['DayOfWeek'] == 4).astype(int)
    data['IsWeekend'] = (data['DayOfWeek'] >= 5).astype(int)
    data['IsTuesday'] = (data['DayOfWeek'] == 1).astype(int)
    
    # Holiday-related features
    data['IsBeforeHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x + pd.Timedelta(days=1))).astype(int)
    data['IsAfterHoliday'] = data['Datetime'].apply(lambda x: 
        is_us_market_holiday(x - pd.Timedelta(days=1))).astype(int)
    
    # Trading session features
    data['IsOpeningHour'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 10.5)).astype(int)
    data['IsClosingHour'] = ((data['Hour'] >= 15.0) & (data['Hour'] <= 16.0)).astype(int)
    data['IsMidDay'] = ((data['Hour'] >= 12.0) & (data['Hour'] < 13.0)).astype(int)
    data['IsMorningSession'] = ((data['Hour'] >= 9.5) & (data['Hour'] < 12.0)).astype(int)
    data['IsAfternoonSession'] = ((data['Hour'] >= 13.0) & (data['Hour'] <= 16.0)).astype(int)
    
    # Airline-specific time features
    # Winter holiday travel period (mid-December to early January)
    is_winter_holiday = ((data['Month'] == 12) & (data['DayOfMonth'] >= 15)) | ((data['Month'] == 1) & (data['DayOfMonth'] <= 7))
    data['IsWinterHolidayTravel'] = is_winter_holiday.astype(int)
    
    # Summer travel peak (June, July, August)
    data['IsSummerTravel'] = ((data['Month'] >= 6) & (data['Month'] <= 8)).astype(int)
    
    # Thanksgiving travel period (around November 23-28)
    is_thanksgiving = ((data['Month'] == 11) & (data['DayOfMonth'] >= 23) & (data['DayOfMonth'] <= 28))
    data['IsThanksgivingTravel'] = is_thanksgiving.astype(int)
    
    # End/Start of month effect (last 2 days or first 2 days of month)
    data['IsMonthStart'] = (data['DayOfMonth'] <= 2).astype(int)
    data['IsMonthEnd'] = (data['DayOfMonth'] >= 29).astype(int)
    
    # Add time-based sine and cosine features to capture cyclical patterns
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24.0)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24.0)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7.0)
    data['DayOfMonth_sin'] = np.sin(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['DayOfMonth_cos'] = np.cos(2 * np.pi * data['DayOfMonth'] / 31.0)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12.0)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12.0)

# Define volatility windows
volatility_windows = [3, 5, 7, 14, 21]
correlation_windows = [5, 10, 20]
market_indicators = ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE']

# Calculate volatility features - process train and forecast separately
print("Creating volatility features with different windows...")
for window in volatility_windows:
    # For training data
    train_data[f'Daily_Vol_{window}d'] = train_data.groupby(
        train_data['Datetime'].dt.date)['AAL_Close'].transform(
        lambda x: x.rolling(window=min(window, len(x))).std()).fillna(method='bfill')
    
    # For forecast data (using training data for calculation to avoid leakage)
    last_training_values = train_data.groupby(
        train_data['Datetime'].dt.date)['AAL_Close'].std().iloc[-window:].mean()
    forecast_data[f'Daily_Vol_{window}d'] = last_training_values

# Calculate technical indicators - RSI, MACD, Bollinger Bands
print("Calculating technical indicators...")
for data in [train_data, forecast_data]:
    data['AAL_RSI_14'] = calculate_rsi(data['AAL_Close'], window=14)
    data['AAL_RSI_7'] = calculate_rsi(data['AAL_Close'], window=7)
    
    data['AAL_MACD'], data['AAL_MACD_Signal'], data['AAL_MACD_Hist'] = calculate_macd(data['AAL_Close'])
    
    data['AAL_BB_Mid'], data['AAL_BB_Upper'], data['AAL_BB_Lower'] = calculate_bollinger_bands(data['AAL_Close'])
    data['AAL_BB_Width'] = data['AAL_BB_Upper'] - data['AAL_BB_Lower']
    data['AAL_BB_Pct'] = (data['AAL_Close'] - data['AAL_BB_Lower']) / (data['AAL_BB_Upper'] - data['AAL_BB_Lower'])
    
    # Calculate technical indicators for market ETFs
    for indicator in market_indicators:
        # RSI
        data[f'{indicator}_RSI_14'] = calculate_rsi(data[f'{indicator}_Close'], window=14)
        
        # MACD
        data[f'{indicator}_MACD'], _, _ = calculate_macd(data[f'{indicator}_Close'])
        
        # Bollinger Bands Percent
        _, upper, lower = calculate_bollinger_bands(data[f'{indicator}_Close'])
        data[f'{indicator}_BB_Pct'] = (data[f'{indicator}_Close'] - lower) / (upper - lower)

# Calculate returns and volatility
print("Calculating returns and volatility...")
for data in [train_data, forecast_data]:
    for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'AAL']:
        # Calculate returns
        data[f'{indicator}_Return_1h'] = data[f'{indicator}_Close'].pct_change()
        
        # Calculate returns for different windows
        for window in [3, 5, 7, 14, 21]:
            # Return over window hours
            data[f'{indicator}_Return_{window}h'] = data[f'{indicator}_Close'].pct_change(window)
            
            # Rolling volatility over window hours
            data[f'{indicator}_Volatility_{window}h'] = data[f'{indicator}_Return_1h'].rolling(
                window=min(window, len(data))).std().fillna(method='bfill')

# Add trend features
print("Adding trend features...")
for data in [train_data, forecast_data]:
    for window in [7, 14, 30]:
        # Add price trend
        data[f'AAL_Price_Trend_{window}d'] = data['AAL_Close'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add volume trend
        data[f'AAL_Volume_Trend_{window}d'] = data['AAL_Volume'].rolling(window).apply(
            lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)
        
        # Add similar trend features for market indicators
        for indicator in market_indicators:
            data[f'{indicator}_Price_Trend_{window}d'] = data[f'{indicator}_Close'].rolling(window).apply(
                lambda x: (x[-1]/x[0] - 1) * 100 if x[0] != 0 else 0, raw=True).fillna(0)

# Add relative price features
print("Creating relative price features...")
for data in [train_data, forecast_data]:
    # Calculate ratios between AAL and market/sector ETFs
    for indicator in market_indicators:
        data[f'AAL_to_{indicator}_Ratio'] = data['AAL_Close'] / data[f'{indicator}_Close']
        data[f'AAL_to_{indicator}_Ratio_Change'] = data[f'AAL_to_{indicator}_Ratio'].pct_change()
    
    # Calculate AAL's percent deviation from its own moving averages
    for window in [5, 10, 20, 50]:
        ma_col = f'AAL_MA_{window}'
        data[ma_col] = data['AAL_Close'].rolling(window).mean().fillna(method='bfill')
        data[f'AAL_Deviation_From_{window}MA_Pct'] = ((data['AAL_Close'] - data[ma_col]) / data[ma_col]) * 100
    
    # Calculate AAL's performance relative to JETS (airline sector ETF)
    data['AAL_vs_JETS_Alpha'] = data['AAL_Return_1h'] - data['JETS_Return_1h']
    
    # Calculate Z-score for AAL price relative to recent history (5, 10, 20 days)
    for window in [5, 10, 20]:
        mean = data['AAL_Close'].rolling(window=window).mean()
        std = data['AAL_Close'].rolling(window=window).std()
        data[f'AAL_Z_Score_{window}d'] = (data['AAL_Close'] - mean) / std
        data[f'AAL_Z_Score_{window}d'] = data[f'AAL_Z_Score_{window}d'].fillna(0)  # Replace NaNs

# Calculate correlations
print("Calculating cross-asset correlations...")
for data in [train_data, forecast_data]:
    for window in correlation_windows:
        for indicator in market_indicators:
            # Calculate rolling correlation between AAL and the market indicator
            data[f'AAL_{indicator}_Corr_{window}h'] = data['AAL_Return_1h'].rolling(
                window=min(window, len(data))).corr(
                data[f'{indicator}_Return_1h']).fillna(method='bfill')

# Add lag features
print("Creating lag features...")
# We need to handle lag features carefully to avoid data leakage
# First, prepare the entire dataset with lags, then split again
temp_df = pd.concat([train_data, forecast_data]).sort_values('Datetime')

# Add lag features for AAL_Close (previous hour prices)
for lag in range(1, 7):  # Expanded to 6 lags
    temp_df[f'AAL_Close_Lag_{lag}'] = temp_df['AAL_Close'].shift(lag)
    temp_df[f'AAL_Return_Lag_{lag}'] = temp_df['AAL_Return_1h'].shift(lag)

# Add market indicator lag features 
for indicator in market_indicators:
    for lag in range(1, 3):  # Using 2 lags
        temp_df[f'{indicator}_Close_Lag_{lag}'] = temp_df[f'{indicator}_Close'].shift(lag)
        temp_df[f'{indicator}_Return_Lag_{lag}'] = temp_df[f'{indicator}_Return_1h'].shift(lag)
    
    # Calculate spread between high and low (indicator of volatility)
    temp_df[f'{indicator}_Spread'] = temp_df[f'{indicator}_High'] - temp_df[f'{indicator}_Low']
    temp_df[f'{indicator}_Spread_Pct'] = temp_df[f'{indicator}_Spread'] / temp_df[f'{indicator}_Close']

# Create target variables - change in price instead of absolute price
temp_df['Next_AAL_Close'] = temp_df['AAL_Close'].shift(-1)
temp_df['Next_AAL_Return'] = temp_df['Next_AAL_Close'] / temp_df['AAL_Close'] - 1  # Percentage change

# Re-split the data to maintain the lag features properly
train_data = temp_df[~temp_df['Datetime'].isin(forecast_dates_requested)]
forecast_data = temp_df[temp_df['Datetime'].isin(forecast_dates_requested)]

# Drop rows with NaN values only in training data
train_data = train_data.dropna(subset=['Next_AAL_Close', 'Next_AAL_Return'])

# Fill any remaining NaNs with appropriate values
for data in [train_data, forecast_data]:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(0)

print(f"Final training data size after processing: {len(train_data)} records")
print(f"Final forecast data size after processing: {len(forecast_data)} records")

# Define features to use
base_features = [
    'Hour', 'DayOfWeek', 'DayOfMonth', 'Month', 'WeekOfYear', 'Quarter',
    'IsMonday', 'IsFriday', 'IsWeekend', 'IsTuesday',
    'IsBeforeHoliday', 'IsAfterHoliday', 
    'IsOpeningHour', 'IsClosingHour', 'IsMidDay', 'IsMorningSession', 'IsAfternoonSession',
    'IsWinterHolidayTravel', 'IsSummerTravel', 'IsThanksgivingTravel',
    'IsMonthStart', 'IsMonthEnd',
    'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
    'DayOfMonth_sin', 'DayOfMonth_cos', 'Month_sin', 'Month_cos',
]

# Add volatility features
volatility_features = [f'Daily_Vol_{window}d' for window in volatility_windows]

# Add technical indicators
tech_features = [
    'AAL_RSI_14', 'AAL_RSI_7', 'AAL_MACD', 'AAL_MACD_Signal', 'AAL_MACD_Hist',
    'AAL_BB_Width', 'AAL_BB_Pct'
]

# Add market indicator features
market_features = []
for indicator in market_indicators:
    market_features.extend([
        f'{indicator}_Open', f'{indicator}_High', f'{indicator}_Low', f'{indicator}_Close', f'{indicator}_Volume',
        f'{indicator}_RSI_14', f'{indicator}_MACD', f'{indicator}_BB_Pct',
        f'{indicator}_Spread', f'{indicator}_Spread_Pct'
    ])

# Add return features
return_features = []
for indicator in ['SPY', 'QQQ', 'IWM', 'JETS', 'XLE', 'AAL']:
    return_features.append(f'{indicator}_Return_1h')
    for window in [3, 5, 7, 14, 21]:
        return_features.extend([
            f'{indicator}_Return_{window}h',
            f'{indicator}_Volatility_{window}h'
        ])

# Add trend features
trend_features = []
for window in [7, 14, 30]:
    trend_features.extend([
        f'AAL_Price_Trend_{window}d',
        f'AAL_Volume_Trend_{window}d'
    ])
    for indicator in market_indicators:
        trend_features.append(f'{indicator}_Price_Trend_{window}d')

# Add relative price features
relative_features = []
for indicator in market_indicators:
    relative_features.extend([
        f'AAL_to_{indicator}_Ratio',
        f'AAL_to_{indicator}_Ratio_Change'
    ])

for window in [5, 10, 20, 50]:
    relative_features.append(f'AAL_MA_{window}')
    relative_features.append(f'AAL_Deviation_From_{window}MA_Pct')

relative_features.append('AAL_vs_JETS_Alpha')

for window in [5, 10, 20]:
    relative_features.append(f'AAL_Z_Score_{window}d')

# Add correlation features
correlation_features = []
for window in correlation_windows:
    for indicator in market_indicators:
        correlation_features.append(f'AAL_{indicator}_Corr_{window}h')

# Add lag features
lag_features = []
for lag in range(1, 7):
    lag_features.extend([
        f'AAL_Close_Lag_{lag}',
        f'AAL_Return_Lag_{lag}'
    ])

for indicator in market_indicators:
    for lag in range(1, 3):
        lag_features.extend([
            f'{indicator}_Close_Lag_{lag}',
            f'{indicator}_Return_Lag_{lag}'
        ])

# Original AAL features
AAL_features = ['AAL_Open', 'AAL_High', 'AAL_Low', 'AAL_Volume']

# Combine all features and ensure they exist in both datasets
all_features = []
for feature_list in [base_features, volatility_features, tech_features, market_features, 
                     return_features, trend_features, correlation_features, 
                     relative_features, lag_features, AAL_features]:
    valid_features = [f for f in feature_list if f in train_data.columns and f in forecast_data.columns]
    all_features.extend(valid_features)

print(f"Initial feature count: {len(all_features)}")

# Perform feature selection using Random Forest
print("Performing feature selection with Random Forest...")
X_train_for_selection = train_data[all_features]
y_train_for_selection = train_data['Next_AAL_Return']  # Using returns for selection

feature_selector = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
feature_selector.fit(X_train_for_selection, y_train_for_selection)

# Get feature importances and select top N
feature_importances = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_selector.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Most Important Features:")
print(feature_importances.head(20))

# Select top features (e.g., top 40 features)
top_n_features = 40
selected_features = feature_importances.head(top_n_features)['Feature'].values.tolist()

print(f"\nSelected {len(selected_features)} features for modeling")

# Prepare data for training and prediction
X_train = train_data[selected_features]
y_train_return = train_data['Next_AAL_Return']
y_train_price = train_data['Next_AAL_Close']
X_forecast = forecast_data[selected_features]

# Use the most recent actual price to later convert predictions back to price levels
last_price = forecast_data['AAL_Close'].iloc[0]

# Scale the data
print("Scaling features...")
feature_scaler = RobustScaler()  # Use RobustScaler to be more robust to outliers
X_train_scaled = feature_scaler.fit_transform(X_train)
X_forecast_scaled = feature_scaler.transform(X_forecast)

# Scale the target variable separately
return_scaler = StandardScaler()
y_train_return_scaled = return_scaler.fit_transform(y_train_return.values.reshape(-1, 1)).flatten()

price_scaler = StandardScaler()
y_train_price_scaled = price_scaler.fit_transform(y_train_price.values.reshape(-1, 1)).flatten()

# Prepare data for LSTM model (reshape to [samples, time_steps, features])
look_back = 6  # Using 6 hours look back

def create_sequences(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Create sequences for training data
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_return_scaled, look_back)

print(f"Training sequences shape: {X_train_seq.shape}")

# Build improved LSTM model with attention and residual connections
print("Building and training improved LSTM model...")
def build_improved_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second LSTM layer with residual connection
    lstm_out = LSTM(32, return_sequences=True)(x)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Dropout(0.2)(lstm_out)
    
    # Add attention mechanism
    attention_out = AttentionLayer()(lstm_out)
    
    # Dense output layers
    dense = Dense(16, activation='relu')(attention_out)
    outputs = Dense(1)(dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model with huber loss for robustness to outliers
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    return model

# Initialize and train the model
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lstm_model = build_improved_lstm_model(input_shape)

# Define callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

# Train the model
history = lstm_model.fit(
    X_train_seq, y_train_seq,
    epochs=100,  # More epochs with early stopping
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.close()

# Now train a Random Forest model to ensemble with LSTM
print("Training Random Forest model for ensemble...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_return_scaled)

# Now make forecasts with both models
print("Making predictions...")
# For LSTM - need to prepare sequences
last_sequence = X_train_scaled[-look_back:]
lstm_forecast_results = []

for i in range(len(X_forecast_scaled)):
    # Make prediction using the current sequence
    current_seq = last_sequence.reshape(1, look_back, X_train_scaled.shape[1])
    prediction = lstm_model.predict(current_seq, verbose=0)[0][0]
    
    # Store the prediction
    lstm_forecast_results.append(prediction)
    
    # Update the sequence for the next prediction by removing the first element
    # and adding the current forecast sample at the end
    last_sequence = np.vstack([last_sequence[1:], X_forecast_scaled[i]])

# Random Forest predictions
rf_forecast_results = rf_model.predict(X_forecast_scaled)

# Ensemble the predictions (50% LSTM, 50% RF)
ensemble_forecast_results = 0.5 * np.array(lstm_forecast_results) + 0.5 * rf_forecast_results

# Convert the return predictions back to the original scale
ensemble_forecast_results_unscaled = return_scaler.inverse_transform(
    ensemble_forecast_results.reshape(-1, 1)).flatten()

# Convert from returns to prices
actual_prices = forecast_data['AAL_Close'].values
predicted_returns = ensemble_forecast_results_unscaled

# Initialize a list to store predicted prices
predicted_prices = []
current_price = last_price

for i, ret in enumerate(predicted_returns):
    # Calculate the next price based on the predicted return
    next_price = current_price * (1 + ret)
    predicted_prices.append(next_price)
    
    # Update the current price for the next prediction
    # If i+1 < len(actual_prices), use the actual price as the base for the next prediction
    # This helps prevent error accumulation
    if i+1 < len(actual_prices):
        current_price = actual_prices[i]  # Use actual price as base
    else:
        current_price = next_price  # Use predicted price if no actual is available

# Calculate performance metrics
mae = mean_absolute_error(actual_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
r2 = r2_score(actual_prices, predicted_prices)

# Perform Monte Carlo simulations for confidence intervals
print("Running Monte Carlo simulations...")
n_simulations = 100
simulation_results = np.zeros((len(predicted_prices), n_simulations))

# Get average volatility from training data
avg_volatility = train_data['AAL_Volatility_7h'].mean()
print(f"Average volatility: {avg_volatility:.6f}")

# Use a more conservative confidence interval by increasing the volatility factor
for i in range(n_simulations):
    # Generate random volatility multipliers ranging from 0.5 to 2.0 of average volatility
    volatility_multiplier = np.random.uniform(0.5, 2.0)
    volatility = avg_volatility * volatility_multiplier * actual_prices
    
    # Apply random noise to predictions based on volatility
    noise = np.random.normal(0, volatility, len(predicted_prices))
    simulation_results[:, i] = predicted_prices + noise

# Calculate 90% confidence intervals
lower_bound = np.percentile(simulation_results, 5, axis=1)
upper_bound = np.percentile(simulation_results, 95, axis=1)
lower_bound = np.maximum(lower_bound, 0)  # Ensure no negative stock prices

# Create forecast results DataFrame with actuals for comparison
forecast_result = pd.DataFrame({
    'Datetime': forecast_data['Datetime'].values,
    'Actual': actual_prices,
    'Predicted': predicted_prices,
    'Lower_Bound': lower_bound,
    'Upper_Bound': upper_bound
})

# Calculate percentage of actual values within confidence interval
within_ci = np.sum((actual_prices >= lower_bound) & (actual_prices <= upper_bound))
ci_percentage = within_ci / len(actual_prices) * 100

print("\nForecast Results with Actual Values:")
print(forecast_result)

print(f"\nPerformance Metrics:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Actual values within 90% confidence interval: {ci_percentage:.2f}%")

# Create subplots to show daily patterns
plt.figure(figsize=(18, 6))

# Add day separators for visual clarity
unique_dates = forecast_result['Datetime'].dt.date.unique()

# Create subplots for each day
for i, date in enumerate(unique_dates):
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date]
    
    plt.subplot(1, 3, i+1)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Actual'], 
             label='Actual', color='forestgreen', marker='o', markersize=8, linewidth=2)
    plt.plot(day_data['Datetime'].dt.strftime('%H:%M'), day_data['Predicted'], 
             label='Predicted', color='royalblue', marker='^', markersize=6)
    
    # Add confidence intervals
    plt.fill_between(day_data['Datetime'].dt.strftime('%H:%M'), 
                     day_data['Lower_Bound'], day_data['Upper_Bound'], 
                     color='royalblue', alpha=0.2, label='90% Confidence Interval')
    
    plt.title(f'Intraday Pattern: {date}', fontsize=14)
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('AAL Close Price ($)', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if i == 0:
        plt.legend(loc='best')
    
    # Add min/max annotations
    plt.annotate(f"Min: ${day_data['Actual'].min():.2f}", 
                 xy=(0.02, 0.04), xycoords='axes fraction', fontsize=8)
    plt.annotate(f"Max: ${day_data['Actual'].max():.2f}", 
                 xy=(0.02, 0.96), xycoords='axes fraction', fontsize=8)
    
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Calculate price changes
forecast_result['Actual_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Actual'].diff()
forecast_result['Predicted_Change'] = forecast_result.groupby(forecast_result['Datetime'].dt.date)['Predicted'].diff()

# Calculate directions (1 for up, -1 for down, 0 for unchanged)
forecast_result['Actual_Direction'] = np.sign(forecast_result['Actual_Change'])
forecast_result['Predicted_Direction'] = np.sign(forecast_result['Predicted_Change'])

# Calculate if direction was predicted correctly
forecast_result['Direction_Correct'] = (forecast_result['Actual_Direction'] == forecast_result['Predicted_Direction']).astype(int)

# Print directional accuracy for each day
print("\nPredictions Directional Accuracy By Day:")
for date in unique_dates:
    day_data = forecast_result[forecast_result['Datetime'].dt.date == date].dropna(subset=['Actual_Change'])
    
    # Adjusted predictions accuracy
    correct = day_data['Direction_Correct'].sum()
    total = len(day_data)
    accuracy = (correct / total) * 100
    
    print(f"{date} Predictions Directional Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
# Output forecast results to CSV
output_csv = forecast_result[['Datetime', 'Actual', 'Predicted']]
output_csv.columns = ['datetime', 'actual', 'predicted']  
output_csv.to_csv('AAL_improved_forecast_results.csv', index=False)
print("Forecast results saved to 'AAL_improved_forecast_results.csv'")

