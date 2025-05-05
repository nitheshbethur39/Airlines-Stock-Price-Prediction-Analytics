#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess the data
df = pd.read_csv("/Users/saketh/Downloads/ALGT_Quarterly_data.csv")

# Ensure Timeframe_Quarter stays in original format
df['Timeframe_Quarter'] = df['Timeframe_Quarter'].astype(str)

# Sort by time
df = df.sort_values('Timeframe_Quarter')

# Drop columns as specified
columns_to_drop = ['Quarter', 'YEAR', 'Open', 'High', 'Low', 'Volume', 'Stock_End_Price']
df = df.drop(columns=columns_to_drop)

# Define target variable
y = df['Adj Close']

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Time range: {df['Timeframe_Quarter'].min()} to {df['Timeframe_Quarter'].max()}")
print(f"Target variable (Adj Close) range: ${y.min():.2f} to ${y.max():.2f}")

# Prepare data for modeling
# Create lag features only for Adj Close
for col in ['Adj Close']:
    if col in df.columns:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag2'] = df[col].shift(2)

# Drop rows with NaN values
df_model = df.dropna()

# Add quarter column
df_model['quarter'] = df_model['Timeframe_Quarter'].str[-1].astype(int)

# Define features and target
numeric_cols = df_model.select_dtypes(include=[np.number]).columns
X = df_model[numeric_cols].drop(columns=['Adj Close'])
y = df_model['Adj Close']

# Create dummy variables for quarter
X = pd.get_dummies(X, columns=['quarter'], prefix='quarter', drop_first=True)

# Convert all column names to strings
X.columns = X.columns.astype(str)

# Time-based split
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
train_dates = df_model['Timeframe_Quarter'].iloc[:train_size]
test_dates = df_model['Timeframe_Quarter'].iloc[train_size:]

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train Ridge model
model = Ridge(alpha=1.0)
print("\nTraining Ridge Regression...")
model.fit(X_train_scaled, y_train)

# Predict on testing data only
y_test_pred = model.predict(X_test_scaled)

# Evaluate model on test data
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"  Test MSE: {mse:.2f}")
print(f"  Test RMSE: {rmse:.2f}")
print(f"  Test MAE: {mae:.2f}")
print(f"  Test R2: {r2:.4f}")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)

print("\nTime Series Cross-Validation Results:")
print(f"Mean RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

# Train final model on all data
final_model = Ridge(alpha=1.0)
final_model.fit(scaler.fit_transform(X), y)

# Calculate trend slopes for key features using linear regression on recent data (last 8 quarters)
recent_df = df.tail(8)
time_idx = np.arange(len(recent_df))
trend_model = LinearRegression()

trend_slopes = {}
for col in ['Adj Close', 'RASM', 'CASM', 'RPM', 'ASM']:
    if col in recent_df.columns:
        trend_model.fit(time_idx.reshape(-1, 1), recent_df[col])
        trend_slopes[col] = trend_model.coef_[0]

# Forecasting for next 5 quarters
last_quarter = df_model.iloc[-1].copy()
forecasts = []
feature_columns = X.columns

for i in range(1, 6):
    new_point = last_quarter.copy()
    
    # Generate next quarter label
    last_year, last_q = last_quarter['Timeframe_Quarter'].split('-Q')
    last_year, last_q = int(last_year), int(last_q)
    new_q = last_q + 1 if last_q < 4 else 1
    new_year = last_year if last_q < 4 else last_year + 1
    new_quarter = f"{new_year}-Q{new_q}"
    new_point['Timeframe_Quarter'] = new_quarter
    new_point['quarter'] = new_q
    
    # Update lag features for Adj Close with trend adjustment
    if i == 1:
        new_point['Adj Close_lag1'] = last_quarter['Adj Close']
        new_point['Adj Close_lag2'] = df_model.iloc[-2]['Adj Close']
    elif i == 2:
        new_point['Adj Close_lag1'] = forecasts[0]['predicted']
        new_point['Adj Close_lag2'] = last_quarter['Adj Close']
    else:
        new_point['Adj Close_lag1'] = forecasts[i-2]['predicted'] + trend_slopes['Adj Close']
        new_point['Adj Close_lag2'] = forecasts[i-3]['predicted'] + trend_slopes['Adj Close']
    
    # Create forecast_features with all columns from X
    forecast_features = pd.DataFrame(columns=feature_columns, index=[0])
    for col in feature_columns:
        if col in ['Adj Close_lag1', 'Adj Close_lag2']:
            forecast_features[col] = new_point[col]
        elif col.startswith('quarter_'):
            continue
        else:
            if col == 'RASM':
                forecast_features[col] = last_quarter['RASM'] + i * trend_slopes['RASM']
            elif col == 'CASM':
                forecast_features[col] = last_quarter['CASM'] + i * trend_slopes['CASM']
            elif col == 'RPM':
                forecast_features[col] = last_quarter['RPM'] + i * trend_slopes['RPM']
            elif col == 'ASM':
                forecast_features[col] = last_quarter['ASM'] + i * trend_slopes['ASM']
            else:
                forecast_features[col] = last_quarter[col]
    
    # Create quarter dummies
    quarter_dummies = pd.get_dummies(pd.Series(new_point['quarter']), prefix='quarter', drop_first=True)
    quarter_dummies.columns = quarter_dummies.columns.astype(str)
    for col in feature_columns:
        if col.startswith('quarter_'):
            forecast_features[col] = quarter_dummies[col] if col in quarter_dummies.columns else 0
    
    # Ensure all column names are strings
    forecast_features.columns = forecast_features.columns.astype(str)
    
    # Make prediction
    forecast_scaled = scaler.transform(forecast_features)
    prediction = final_model.predict(forecast_scaled)[0]
    
    ci_width = 1.96 * cv_rmse.mean()
    ci_lower = prediction - ci_width
    ci_upper = prediction + ci_width
    
    forecasts.append({
        'quarter': new_quarter,
        'predicted': prediction,
        'lower_bound': max(0, ci_lower),
        'upper_bound': ci_upper,
        'RASM': forecast_features['RASM'].values[0],
        'CASM': forecast_features['CASM'].values[0],
        'RPM': forecast_features['RPM'].values[0],
        'ASM': forecast_features['ASM'].values[0]
    })
    
    last_quarter = new_point

# Display forecasts
print("\nForecasts for the next 5 quarters:")
for forecast in forecasts:
    print(f"{forecast['quarter']}: ${forecast['predicted']:.2f} (95% CI: ${forecast['lower_bound']:.2f} - ${forecast['upper_bound']:.2f})")
    print(f"  RASM: {forecast['RASM']:.4f}, CASM: {forecast['CASM']:.4f}, RPM: {forecast['RPM']:.2f}, ASM: {forecast['ASM']:.2f}")

# Combine training and testing data for a single plot
all_dates = list(train_dates) + list(test_dates)
all_actuals = list(y_train) + list(y_test)
forecast_dates = [f['quarter'] for f in forecasts]
forecast_values = [f['predicted'] for f in forecasts]
forecast_lower = [f['lower_bound'] for f in forecasts]
forecast_upper = [f['upper_bound'] for f in forecasts]

# Single Plot with Training, Testing, and Forecasts
plt.figure(figsize=(16, 8))
plt.plot(all_dates, all_actuals, 'b-', marker='o', label='Actual Prices (Train + Test)')
plt.plot(test_dates, y_test_pred, 'g--', marker='x', label='Test Predicted')
plt.plot(forecast_dates, forecast_values, 'r--', marker='o', label='Forecasted Prices')
plt.fill_between(forecast_dates, forecast_lower, forecast_upper, color='r', alpha=0.2, label='95% CI')

# Annotate values on the plot
for i, (date, value) in enumerate(zip(all_dates, all_actuals)):
    plt.text(date, value, f'${value:.2f}', fontsize=8, ha='center', va='bottom' if i % 2 == 0 else 'top')
for date, value in zip(test_dates, y_test_pred):
    plt.text(date, value, f'${value:.2f}', fontsize=8, ha='center', va='top', color='green')
for date, value in zip(forecast_dates, forecast_values):
    plt.text(date, value, f'${value:.2f}', fontsize=8, ha='center', va='bottom', color='red')

plt.title('ALGT Stock Price: Historical and Forecast (Ridge Regression)')
plt.xlabel('Quarter')
plt.ylabel('Adjusted Close Price ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Summary of model performance
print("\nModel Performance Summary:")
print(f"Ridge Regression (Test Set):")
print(f"  RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
print(f"Cross-Validated RMSE: {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")


# In[3]:


# Get top 10 features based on Ridge coefficients
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': final_model.coef_
})
feature_importance['Absolute_Coefficient'] = feature_importance['Coefficient'].abs()
top_10_features = feature_importance.sort_values('Absolute_Coefficient', ascending=False).head(10)

# Display the top 10 features
print("\nTop 10 Features by Absolute Coefficient (Ridge Regression):")
print(top_10_features[['Feature', 'Coefficient']].to_string(index=False))


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming test_dates, y_test, y_test_pred, forecast_dates, forecast_values, 
# forecast_lower, and forecast_upper are defined from your previous code

plt.figure(figsize=(14, 7))
plt.plot(test_dates, y_test, 'b-', marker='o', label='Testing Actual')
plt.plot(test_dates, y_test_pred, 'g--', marker='x', label='Testing Predicted')
plt.plot(forecast_dates, forecast_values, 'r--', marker='o', label='Forecasted Prices')
plt.fill_between(forecast_dates, forecast_lower, forecast_upper, color='r', alpha=0.2, label='95% CI')

# Calculate offset based on the range of values for spacing
all_values = list(y_test) + list(y_test_pred) + forecast_values
offset = max(all_values) * 0.02  # 2% of max value for offset

# Annotate Testing Actual values (blue)
for i, (date, value) in enumerate(zip(test_dates, y_test)):
    va = 'bottom' if i % 2 == 0 else 'top'  # Alternate to avoid overlap
    plt.text(date, value + (offset if va == 'bottom' else -offset), f'${value:.2f}', 
             fontsize=7, ha='center', va=va, color='blue')

# Annotate Testing Predicted values (green)
for date, value in zip(test_dates, y_test_pred):
    plt.text(date, value + offset, f'${value:.2f}', fontsize=7, ha='center', va='bottom', color='green')

# Annotate Forecasted values (red)
for date, value in zip(forecast_dates, forecast_values):
    plt.text(date, value + offset, f'${value:.2f}', fontsize=7, ha='center', va='bottom', color='red')

plt.title('ALGT Stock Price: Testing Data and Forecast (Ridge Regression)')
plt.xlabel('Quarter')
plt.ylabel('Adjusted Close Price ($)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Calculate errors for test data
errors = np.array(y_test) - np.array(y_test_pred)

# Create a DataFrame for test period results
test_results = pd.DataFrame({
    'Quarter': test_dates,
    'Actual Value': y_test,
    'Predicted Value': y_test_pred,
    'Error': errors
})

# Reset index for cleaner display
test_results = test_results.reset_index(drop=True)

# Display the test results
print("\nTest Period Results:")
print(test_results.to_string(index=False))

# Summary statistics for errors
print("\nError Summary:")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"Mean Error: {errors.mean():.2f}")
print(f"Standard Deviation of Error: {errors.std():.2f}")


# In[15]:


def plot_forecast_comparison(train_dates, y_train, test_dates, y_test, y_test_pred, forecast_dates, forecast_values, top_features, last_actual):
    # Ensure all sequences are lists (to avoid KeyError from Series)
    train_dates = list(train_dates)
    test_dates = list(test_dates)
    forecast_dates = list(forecast_dates)

    all_dates = train_dates + test_dates
    all_actuals = list(y_train) + list(y_test)

    plt.figure(figsize=(16, 9))
    
    # Historical data
    plt.plot(train_dates, y_train, 'o-', color='blue', label='Historical Stock Prices')
    
    # Test actual prices
    plt.plot(test_dates, y_test, 'o', color='purple', label='Actual Test Prices', markersize=8)
    
    # Test predicted prices
    plt.plot(test_dates, y_test_pred, 'D', color='red', label='Predicted Test Prices', markersize=6)
    
    # Forecast
    plt.plot(forecast_dates, forecast_values, 's--', color='green', label='Forecasted Stock Prices', markersize=6)

    # Forecast region background
    plt.axvspan(forecast_dates[0], forecast_dates[-1], color='green', alpha=0.1, label='Forecast Period')
    plt.axvspan(test_dates[0], test_dates[-1], color='pink', alpha=0.2, label='Test Period')

    # Horizontal line at last actual price
    plt.axhline(last_actual, color='lightcoral', linestyle='--')
    plt.text(test_dates[0], last_actual + 1.5, f'Last Price: ${last_actual:.2f}', color='darkred', fontsize=10)

    # Forecast annotation
    final_forecast = forecast_values[-1]
    percent_change = 100 * (final_forecast - last_actual) / last_actual
    plt.text(forecast_dates[-1], final_forecast, f"Last Predicted: ${final_forecast:.2f}\n% Change: {percent_change:.2f}%", 
             ha='right', va='top', fontsize=10, color='darkgreen')

    # Top features annotation
    box_text = '\n'.join([f"{i+1}. {feat}: {score:.4f}" for i, (feat, score) in enumerate(top_features.items())])
    plt.text(forecast_dates[-1], max(all_actuals + forecast_values), f"Top Features:\n{box_text}",
             bbox=dict(facecolor='lightgrey', alpha=0.5), fontsize=10, ha='right')

    # Model metrics
    mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    r2 = r2_score(y_test, y_test_pred)
    
    plt.text(forecast_dates[-1], min(all_actuals + forecast_values),
             f"Model Performance:\nR² = {r2:.4f}\nRMSE = ${rmse:.2f}\nMAPE = {mape:.2f}%",
             bbox=dict(facecolor='lightyellow', alpha=0.7), fontsize=10, ha='right', va='bottom')

    plt.title("UAL Stock Price - Historical Data and Forecast")
    plt.xlabel("Year")
    plt.ylabel("Stock Price ($)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# In[ ]:




