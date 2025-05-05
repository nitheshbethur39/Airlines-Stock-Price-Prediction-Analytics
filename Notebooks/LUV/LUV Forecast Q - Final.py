#!/usr/bin/env python
# coding: utf-8

# ## Data Prep

# In[ ]:


import pandas as pd
import numpy as np
def filter_luv_data(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter for rows where Stock_Ticker equals "LUV"
    luv_data = df[df["Stock Ticker"] == "LUV"]
    
    
    specific_columns_to_remove = ['UNIQUE_CARRIER', 'Stock Ticker', 'UNIQUE_CARRIER_NAME', 
                                 'Year','Month_str', 'AIRLINE_ID', 'Stock_End_Price', 'AirlineID','Month_str_t3_segment_merged_quarterly']
    luv_data = luv_data.drop(specific_columns_to_remove, axis=1)
    
    # Remove columns that have any values equal to 0
    columns_with_zeros = [col for col in luv_data.columns if (luv_data[col] == 0).any()]
    luv_data = luv_data.drop(columns_with_zeros, axis=1)
    
    # Remove columns with any empty/null values
    columns_with_nulls = luv_data.columns[luv_data.isna().any()].tolist()
    luv_data = luv_data.drop(columns_with_nulls, axis=1)
    
    # Save the filtered data to a new CSV file
    luv_data.to_csv(output_file, index=False)
    
    print(f"Filtered data saved to {output_file}")
    print(f"Number of rows in filtered data: {len(luv_data)}")
    print(f"Removed specific columns: {specific_columns_to_remove}")
    print(f"Removed {len(columns_with_zeros)} columns with any zeros: {columns_with_zeros}")
    print(f"Removed {len(columns_with_nulls)} columns with empty values: {columns_with_nulls}")
if __name__ == "__main__":
    input_file = "Merged_Airlines_With_Revenue.csv"
    output_file = "luv_Q1.csv"
    
    filter_luv_data(input_file, output_file)


# # Model Selection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
# Additional imports for new models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, RegressorMixin
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(file_path):
    """Load the quarterly data with minimal processing"""
    print("Loading quarterly data...")
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Add date column for time series processing
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + 
                              df['Quarter'].astype(str) + '-01', format='%Y-%m-%d')
    
    # Sort by date to ensure correct time order
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def preprocess_data(df):
    """Preprocess data with essential features for quarterly prediction"""
    print("\nPreprocessing quarterly data...")
    df_processed = df.copy()
    
    # Create volatility feature using (High-Low)/Low
    if 'High' in df_processed.columns and 'Low' in df_processed.columns:
        df_processed['Volatility'] = (df_processed['High'] - df_processed['Low']) / df_processed['Low']
    
    # Drop price-related columns that would leak the target
    # Do this BEFORE creating lag features to avoid having lag versions of these columns
    price_cols = ['High', 'Low','Open']
    df_processed.drop(columns=[col for col in price_cols if col in df_processed.columns], inplace=True)
    
    # Create quarter-specific features
    df_processed['Quarter_Sin'] = np.sin(2 * np.pi * df_processed['Quarter'] / 4)
    df_processed['Quarter_Cos'] = np.cos(2 * np.pi * df_processed['Quarter'] / 4)
    
    # Create return features
    df_processed['Return'] = df_processed['Adj Close'].pct_change()
    
    # Create quarter-over-quarter growth
    df_processed['QoQ_Growth'] = df_processed['Adj Close'].pct_change()
    
    # Create year-over-year growth (same quarter previous year)
    df_processed['YoY_Growth'] = df_processed['Adj Close'].pct_change(4)
    
    # Identify numeric features for lag creation
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['YEAR', 'Quarter', 'Return', 
                                                          'Volatility', 'Adj Close', 
                                                          'Quarter_Sin', 'Quarter_Cos', 
                                                          'QoQ_Growth', 'YoY_Growth']]
    
    # Create correlations with target to find most important features
    corr_with_target = df_processed[numeric_cols].corrwith(df_processed['Adj Close'], numeric_only=True).abs()
    important_features = corr_with_target.sort_values(ascending=False).head(10).index.tolist()
    
    # Always include 'Adj Close' for lagging
    important_features = list(set(important_features + ['Adj Close']))
    
    print(f"Creating lag features for top correlated features...")
    
    # Create lag features for important columns
    for col in important_features:
        # Previous quarter
        df_processed[f'{col}_LAG_1'] = df_processed[col].shift(1)
        
        # Previous year, same quarter (lag 4 for quarterly data)
        df_processed[f'{col}_LAG_4'] = df_processed[col].shift(4)
    
    # Create rolling statistics for Adj Close
    df_processed['ADJ_CLOSE_ROLL_MEAN_4'] = df_processed['Adj Close'].rolling(window=4, min_periods=1).mean()
    
    # Drop non-numeric columns that can't be used for modeling
    object_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    if 'Date' in object_cols:
        object_cols.remove('Date')  # Keep Date for time series analysis
    df_processed.drop(columns=object_cols, inplace=True)
    
    # Drop columns with too many NaNs (more than 25%)
    cols_to_drop = [col for col in df_processed.columns if df_processed[col].isna().sum() > 0.25 * len(df_processed)]
    df_processed.drop(columns=cols_to_drop, inplace=True)
    
    # Clean up any remaining NaNs
    df_processed = df_processed.dropna().reset_index(drop=True)
    
    # Separate features and target
    X = df_processed.drop(['Adj Close', 'Date'], axis=1, errors='ignore')
    y = df_processed['Adj Close']
    
    feature_names = X.columns.tolist()
    print(f"Total features after preprocessing: {len(feature_names)}")
    
    return X, y, feature_names, df_processed

def create_time_weights(n_samples, decay_factor=0.95):
    """Create exponential weights that give more importance to recent observations"""
    weights = np.ones(n_samples)
    for i in range(n_samples):
        weights[i] = decay_factor ** (n_samples - 1 - i)
    # Normalize weights to sum to n_samples
    weights = weights * n_samples / np.sum(weights)
    return weights

def split_time_series_data(X, y, df_processed, test_size=0.10):
    """Split data chronologically, keeping the last portion for testing"""
    print(f"\nSplitting quarterly data chronologically with test_size={test_size}...")
    
    # Calculate the split point
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_size))
    
    # Split the data
    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    
    # Create sample weights that give more importance to recent observations
    sample_weights = create_time_weights(len(X_train))
    
    # Get dates for reference
    train_dates = df_processed.iloc[:split_idx]['Date']
    test_dates = df_processed.iloc[split_idx:]['Date']
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, sample_weights, train_dates, test_dates

def scale_features(X_train, X_test):
    """Scale features using only training data statistics"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def perform_feature_selection(X_train, y_train, X_test, feature_names, sample_weights):
    print("\nPerforming feature selection on quarterly training data...")
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Use gradient boosting for feature selection
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    gb.fit(X_train_df, y_train, sample_weight=sample_weights)
    importance = gb.feature_importances_
    
    # Create a dataframe of feature importances
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Print top 10 features by importance
    print("\nTop 10 features by importance:")
    print(feature_importance.head(20))
    
    # Select features that cumulatively contribute to 90% of the importance
    cumulative_importance = np.cumsum(feature_importance['Importance'])
    n_features = max(np.argmax(cumulative_importance >= 0.90) + 1, 10)  # At least 10 features
    n_features = min(n_features, 25)  # Cap at 25 features maximum
    top_features = feature_importance.head(n_features)['Feature'].tolist()
    
    # Ensure key time-related features are included
    essential_features = [
        'YEAR', 'Quarter', 'Quarter_Sin', 'Quarter_Cos',
        'QoQ_Growth', 'YoY_Growth'
    ]
    
    for feature in essential_features:
        if feature in feature_names and feature not in top_features:
            top_features.append(feature)
    
    # Include lag features for Adj Close
    adj_close_lags = [col for col in feature_names if 'ADJ_CLOSE_LAG' in col]
    for col in adj_close_lags:
        if col not in top_features:
            top_features.append(col)
    
    # Include volatility and return
    for col in ['Volatility', 'Return']:
        if col in feature_names and col not in top_features:
            top_features.append(col)
    
    print(f"Selected {len(top_features)} features out of {len(feature_names)} total features")
    
    X_train_selected = X_train_df[top_features].copy()
    X_test_selected = X_test_df[top_features].copy()
    
    return {
        'original': (X_train_df, X_test_df, list(feature_names)),
        'selected': (X_train_selected, X_test_selected, top_features)
    }

# Custom wrapper for ARIMA model to make it compatible with scikit-learn API
class ARIMAWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.model = None
        
    def fit(self, X, y, sample_weight=None):
        # ARIMA only uses the target variable, not features
        self.model = SARIMAX(y, order=self.order, seasonal_order=(0, 0, 0, 0))
        self.result_ = self.model.fit(disp=False)
        return self
        
    def predict(self, X):
        # Predict one step ahead for each historical point plus future points
        n_train = len(self.result_.fittedvalues)
        n_test = len(X)
        
        # Get predictions for test set
        preds = self.result_.get_forecast(steps=n_test).predicted_mean
        return preds

# Create LSTM model builder
def create_lstm_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units//2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Create GRU model builder
def create_gru_model(input_shape, units=50, dropout=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(GRU(units=units//2, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# Custom wrapper for sequence models (LSTM and GRU)
class SequenceModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='lstm', sequence_length=4, units=50, dropout=0.2, 
                 learning_rate=0.001, epochs=100, batch_size=16):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def _prepare_sequences(self, X, y=None):
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        # Reshape to [samples, timesteps, features]
        samples = X.shape[0] - self.sequence_length + 1
        features = X.shape[1]
        
        X_seq = np.zeros((samples, self.sequence_length, features))
        for i in range(samples):
            X_seq[i] = X[i:i+self.sequence_length, :]
            
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            y_seq = y[self.sequence_length-1:]
            return X_seq, y_seq
        
        return X_seq
    
    def fit(self, X, y, sample_weight=None):
        # Prepare sequences
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        # Create input shape for the model
        input_shape = (self.sequence_length, X.shape[1])
        
        # Build model
        if self.model_type.lower() == 'lstm':
            self.model = create_lstm_model(input_shape, self.units, self.dropout, self.learning_rate)
        else:  # GRU
            self.model = create_gru_model(input_shape, self.units, self.dropout, self.learning_rate)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Fit model
        self.history = self.model.fit(
            X_seq, y_seq, 
            validation_split=0.2,
            epochs=self.epochs, 
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        # For prediction, we need to include some history
        X_seq = self._prepare_sequences(X)
        return self.model.predict(X_seq, verbose=0).flatten()

def train_evaluate_models(X_train, X_test, y_train, y_test, feature_names, sample_weights, train_dates, test_dates, reduction_strategy='original'):
    print(f"\nTraining and evaluating models using {reduction_strategy} features...")
    
    # Print dimensions to verify
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    # Define models
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'ARIMA': ARIMAWrapper(),
        'LSTM': SequenceModelWrapper(model_type='lstm'),
        'GRU': SequenceModelWrapper(model_type='gru')
    }
    
    # Define parameter grids
    param_grids = {
        'Ridge': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'Lasso': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]},
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1, 1.0], 
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'RandomForest': {
            'n_estimators': [50, 100], 
            'max_depth': [None, 5, 10]
        },
        'GradientBoosting': {
            'n_estimators': [50, 100], 
            'learning_rate': [0.01, 0.1], 
            'max_depth': [3, 5]
        },
        'SVR': {
            'C': [0.1, 1, 10], 
            'gamma': ['scale', 'auto']
        },
        'ARIMA': {'order': [(1,0,0), (1,1,0), (1,1,1), (2,1,0), (2,1,1)]},
        'LSTM': {
            'sequence_length': [4, 8],
            'units': [32, 64],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.01]
        },
        'GRU': {
            'sequence_length': [4, 8],
            'units': [32, 64],
            'dropout': [0.2, 0.3],
            'learning_rate': [0.001, 0.01]
        }
    }
    
    results = {}
    best_models = {}
    forecasts = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use TimeSeriesSplit for proper cross-validation with time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        try:
            # Special case for LSTM and GRU models - we'll use a simpler approach due to computational constraints
            if name in ['LSTM', 'GRU']:
                print(f"Training {name} without grid search due to computational constraints...")
                
                # Just create and fit one model with default parameters
                seq_model = SequenceModelWrapper(model_type=name.lower(), sequence_length=4, 
                                              units=50, dropout=0.2, 
                                              learning_rate=0.001,
                                              epochs=100, batch_size=16)
                
                # Adjust sample weights for sequence models
                if sample_weights is not None:
                    # Truncate sample weights for sequences
                    seq_weights = sample_weights[3:]  # Assuming sequence_length=4
                else:
                    seq_weights = None
                
                # Fit the model
                seq_model.fit(X_train, y_train)
                best_models[name] = seq_model
                
                # Make predictions
                # Note: Due to sequence requirements, predictions start from the sequence_length
                y_train_pred = seq_model.predict(X_train)
                y_test_pred = seq_model.predict(X_test)
                
                # Store predictions
                forecasts[name] = {'train': y_train_pred, 'test': y_test_pred}
                
                # Evaluate model performance
                # Note: For true comparison, we need to truncate actual values to match predictions
                y_train_trunc = y_train.iloc[3:].values  # Assuming sequence_length=4
                
                results[name] = {
                    'Best Parameters': {'sequence_length': 4, 'units': 50, 'dropout': 0.2, 'learning_rate': 0.001},
                    'Training': {
                        'MSE': mean_squared_error(y_train_trunc, y_train_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_train_trunc, y_train_pred)),
                        'MAE': mean_absolute_error(y_train_trunc, y_train_pred),
                        'R2': r2_score(y_train_trunc, y_train_pred)
                    },
                    'Testing': {
                        'MSE': mean_squared_error(y_test, y_test_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'MAE': mean_absolute_error(y_test, y_test_pred),
                        'R2': r2_score(y_test, y_test_pred)
                    }
                }
                
            elif name == 'ARIMA':
                print(f"Training ARIMA with grid search...")
                best_order = None
                best_aic = float('inf')
                
                # Find the best ARIMA parameters based on AIC
                for order in param_grids['ARIMA']['order']:
                    try:
                        arima_model = SARIMAX(y_train, order=order, seasonal_order=(0, 0, 0, 0))
                        arima_result = arima_model.fit(disp=False)
                        
                        if arima_result.aic < best_aic:
                            best_aic = arima_result.aic
                            best_order = order
                    except:
                        continue
                
                # Create and fit the best model
                arima_wrapper = ARIMAWrapper(order=best_order)
                arima_wrapper.fit(X_train, y_train)
                best_models[name] = arima_wrapper
                
                # Make predictions
                y_train_pred = arima_wrapper.predict(X_train)
                y_test_pred = arima_wrapper.predict(X_test)
                forecasts[name] = {'train': y_train_pred, 'test': y_test_pred}
                
                # Evaluate model performance
                results[name] = {
                    'Best Parameters': {'order': best_order},
                    'Training': {
                        'MSE': mean_squared_error(y_train, y_train_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'MAE': mean_absolute_error(y_train, y_train_pred),
                        'R2': r2_score(y_train, y_train_pred)
                    },
                    'Testing': {
                        'MSE': mean_squared_error(y_test, y_test_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'MAE': mean_absolute_error(y_test, y_test_pred),
                        'R2': r2_score(y_test, y_test_pred)
                    }
                }
                
            else:
                # Standard approach with grid search for traditional models
                # Perform grid search with sample weights for time-based importance
                grid_search = GridSearchCV(
                    model, 
                    param_grids[name], 
                    cv=tscv, 
                    scoring='neg_mean_squared_error', 
                    n_jobs=-1
                )
                
                # Fit the model with sample weights that prioritize recent data
                grid_search.fit(X_train, y_train, sample_weight=sample_weights)
                
                best_model = grid_search.best_estimator_
                best_models[name] = best_model
                
                # Make predictions
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                forecasts[name] = {'train': y_train_pred, 'test': y_test_pred}
                
                # Evaluate model performance
                results[name] = {
                    'Best Parameters': grid_search.best_params_,
                    'Training': {
                        'MSE': mean_squared_error(y_train, y_train_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'MAE': mean_absolute_error(y_train, y_train_pred),
                        'R2': r2_score(y_train, y_train_pred)
                    },
                    'Testing': {
                        'MSE': mean_squared_error(y_test, y_test_pred),
                        'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'MAE': mean_absolute_error(y_test, y_test_pred),
                        'R2': r2_score(y_test, y_test_pred)
                    }
                }
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue
    
    # Print evaluation results
    print(f"\nModel evaluation results ({reduction_strategy} features):")
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Best Parameters: {result['Best Parameters']}")
        print("  Training Metrics:")
        for metric, value in result['Training'].items():
            print(f"    {metric}: {value:.4f}")
        print("  Testing Metrics:")
        for metric, value in result['Testing'].items():
            print(f"    {metric}: {value:.4f}")
    
    # Find best model based on test R2
    valid_models = {k: v for k, v in results.items() if v['Testing']['R2'] > 0}  # Only consider models with positive R2
    if valid_models:
        best_model_name = max(valid_models, key=lambda x: valid_models[x]['Testing']['R2'])
    else:
        # If all models perform poorly, choose the one with least negative R2
        best_model_name = max(results, key=lambda x: results[x]['Testing']['R2'])
    
    print(f"\nBest model with {reduction_strategy} features: {best_model_name} with Test R2: {results[best_model_name]['Testing']['R2']:.4f}")
    
    # Plot actual vs predicted for best model
    plt.figure(figsize=(14, 8))
    
    # Plot train data
    plt.subplot(2, 1, 1)
    plt.plot(train_dates, y_train, 'b-', label='Actual (Train)')
    plt.plot(train_dates, forecasts[best_model_name]['train'], 'r--', label='Predicted (Train)')
    plt.title(f'Train Data: {best_model_name} Model ({reduction_strategy} features)')
    plt.ylabel('Adjusted Close Price ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot test data
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, y_test, 'g-', label='Actual (Test)')
    plt.plot(test_dates, forecasts[best_model_name]['test'], 'r--', label='Predicted (Test)')
    plt.title(f'Test Data: {best_model_name} Model ({reduction_strategy} features)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price ($)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return best_models, best_model_name, results, forecasts

def analyze_feature_importance(best_models, best_model_name, feature_names, X_train, y_train):
    print("\nAnalyzing feature importance...")
    best_model = best_models[best_model_name]
    
    # Get feature importance based on model type
    if best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
        importance = np.abs(best_model.coef_)
    elif best_model_name == 'SVR':
        # Use permutation importance for SVR
        result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=42)
        importance = result.importances_mean
    elif best_model_name in ['RandomForest', 'GradientBoosting']:
        # For tree-based models, use built-in feature importance
        importance = best_model.feature_importances_
    elif best_model_name in ['LSTM', 'GRU', 'ARIMA']:
        # For sequence models and ARIMA, use permutation importance
        try:
            result = permutation_importance(best_model, X_train, y_train, n_repeats=5, random_state=42)
            importance = result.importances_mean
        except:
            print(f"Cannot calculate feature importance for {best_model_name}. Using uniform weights.")
            importance = np.ones(len(feature_names)) / len(feature_names)
    else:
        # For other models, use permutation importance
        result = permutation_importance(best_model, X_train, y_train, n_repeats=10, random_state=42)
        importance = result.importances_mean
    
    # Create a DataFrame of feature importances
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Print top features by importance
    print(f"\nTop feature importance for {best_model_name}:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title(f'Top 15 Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.show()
    
    return feature_importance

def main(file_path):
    # Load the data
    df = load_data(file_path)
    
    # Preprocess data with feature engineering
    X, y, feature_names, df_processed = preprocess_data(df)
    
    # Split data chronologically (proper for time series)
    X_train_raw, X_test_raw, y_train, y_test, sample_weights, train_dates, test_dates = split_time_series_data(X, y, df_processed)
    
    # Scale features using only training data statistics
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)
    
    # Perform feature selection
    reduction_results = perform_feature_selection(X_train_scaled, y_train, X_test_scaled, feature_names, sample_weights)
    
    # Train and evaluate models with all features
    print("\n=== EVALUATING MODELS WITH ALL FEATURES ===")
    all_features_data = reduction_results['original']
    best_models_all, best_model_name_all, results_all, forecasts_all = train_evaluate_models(
        all_features_data[0], all_features_data[1], y_train, y_test, 
        all_features_data[2], sample_weights, train_dates, test_dates, 'all'
    )
    
    # Train and evaluate models with selected features
    print("\n=== EVALUATING MODELS WITH SELECTED FEATURES ===")
    selected_features_data = reduction_results['selected']
    best_models_selected, best_model_name_selected, results_selected, forecasts_selected = train_evaluate_models(
        selected_features_data[0], selected_features_data[1], y_train, y_test, 
        selected_features_data[2], sample_weights, train_dates, test_dates, 'selected'
    )
    
    # Analyze feature importance for best models
    print("\n=== TOP FEATURES FOR BEST MODEL WITH ALL FEATURES ===")
    feature_importance_all = analyze_feature_importance(
        best_models_all, best_model_name_all, all_features_data[2], 
        all_features_data[0], y_train
    )
    
    print("\n=== TOP FEATURES FOR BEST MODEL WITH SELECTED FEATURES ===")
    feature_importance_selected = analyze_feature_importance(
        best_models_selected, best_model_name_selected, selected_features_data[2], 
        selected_features_data[0], y_train
    )
    
    # Compare results
    print("\n=== COMPARISON OF ALL FEATURES VS SELECTED FEATURES ===")
    print(f"All Features - Best model: {best_model_name_all}")
    print(f"  Training R2: {results_all[best_model_name_all]['Training']['R2']:.4f}")
    print(f"  Testing R2: {results_all[best_model_name_all]['Testing']['R2']:.4f}")
    print(f"  Training RMSE: {results_all[best_model_name_all]['Training']['RMSE']:.4f}")
    print(f"  Testing RMSE: {results_all[best_model_name_all]['Testing']['RMSE']:.4f}")
    
    print(f"\nSelected Features - Best model: {best_model_name_selected}")
    print(f"  Training R2: {results_selected[best_model_name_selected]['Training']['R2']:.4f}")
    print(f"  Testing R2: {results_selected[best_model_name_selected]['Testing']['R2']:.4f}")
    print(f"  Training RMSE: {results_selected[best_model_name_selected]['Training']['RMSE']:.4f}")
    print(f"  Testing RMSE: {results_selected[best_model_name_selected]['Testing']['RMSE']:.4f}")
    
    # Determine overall best approach
    if results_selected[best_model_name_selected]['Testing']['R2'] > results_all[best_model_name_all]['Testing']['R2']:
        best_approach = "Selected Features"
        best_model_name = best_model_name_selected
        best_performance = results_selected[best_model_name_selected]['Testing']['R2']
    else:
        best_approach = "All Features"
        best_model_name = best_model_name_all
        best_performance = results_all[best_model_name_all]['Testing']['R2']
    
    print(f"\nBest overall approach: {best_approach} with {best_model_name}, Test R2: {best_performance:.4f}")
    
    print("\nQuarterly prediction analysis complete.")
    
    return {
        'best_approach': best_approach,
        'best_model_name': best_model_name,
        'best_performance': best_performance
    }

if __name__ == "__main__":
    file_path = "luv_Q1.csv"  # Quarterly data file
    results = main(file_path)


# ## Forecasting using the best model

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['Quarter'].astype(str) + '-01')
    df = df.sort_values('Date').reset_index(drop=True)
    df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    df['Volatility'] = (df['High'] - df['Low']) / df['Low']
    df['Return'] = df['Adj Close'].pct_change()
    df['QoQ_Growth'] = df['Adj Close'].pct_change()
    df['YoY_Growth'] = df['Adj Close'].pct_change(4)
    df['ADJ_CLOSE_ROLL_MEAN_4'] = df['Adj Close'].rolling(window=4).mean()
    df = create_lag_features(df, ['Adj Close'], lags=[1, 4])
    return df.dropna().reset_index(drop=True)

def create_lag_features(df, features, lags=[1, 4]):
    for col in features:
        for lag in lags:
            df[f'{col}_LAG_{lag}'] = df[col].shift(lag)
    return df

def analyze_historical_patterns(df):
    # Dynamically compute average quarter-over-quarter return
    quarter_returns = {}
    for q in [1, 2, 3, 4]:
        q_returns = []
        for year in df['YEAR'].unique():
            current = df[(df['YEAR'] == year) & (df['Quarter'] == q)]
            prev = df[(df['YEAR'] == year - 1) & (df['Quarter'] == q)]
            if not current.empty and not prev.empty:
                current_price = current['Adj Close'].values[0]
                prev_price = prev['Adj Close'].values[0]
                q_returns.append((current_price - prev_price) / prev_price)

        quarter_returns[q] = np.mean(q_returns) if q_returns else 0

    # Determine best quarter with weighted score
    recent_years = sorted(df['YEAR'].unique())[-5:]
    annual_returns = []
    quarter_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    weight = 1.0

    for year in reversed(recent_years):
        year_data = df[df['YEAR'] == year]
        if not year_data.empty:
            max_q = year_data.loc[year_data['Adj Close'].idxmax()]['Quarter']
            quarter_scores[int(max_q)] += weight
        weight *= 0.8
        if len(year_data) >= 2:
            start, end = year_data['Adj Close'].iloc[0], year_data['Adj Close'].iloc[-1]
            annual_returns.append((end / start) - 1)

    best_quarter = max(quarter_scores, key=quarter_scores.get)
    avg_annual_return = np.mean(annual_returns) if annual_returns else 0.08

    print("📊 Historical seasonality (avg QoQ return):")
    for q in range(1, 5):
        print(f"  Q{q}: {quarter_returns[q]:.4f}")

    print(f"\n🟩 Quarter with highest historical adj close (weighted): Q{best_quarter}")
    return {
        'seasonality': quarter_returns,
        'avg_annual_return': avg_annual_return,
        'best_quarter': best_quarter
    }

def simulate_feature_value(df, feature, last_val, n_simulations=100):
    qoq_changes = df[feature].pct_change().dropna()
    mu, sigma = qoq_changes.mean(), qoq_changes.std()
    if np.isnan(mu) or np.isnan(sigma):
        return last_val
    simulations = last_val * (1 + np.random.normal(mu, sigma, n_simulations))
    return np.mean(simulations)

def forecast_next_quarters_elasticnet_realistic(df, patterns, n_quarters=5):
    print(f"\nForecasting next {n_quarters} quarters using ElasticNet + Monte Carlo + Dynamic Seasonality...\n")

    all_possible_features = ['Open','ADJ_CLOSE_ROLL_MEAN_4', 'Part-time', 'LONG_TERM_DEBT',
                             'QoQ_Growth', 'YoY_Growth', 'Quarter', 'Quarter_Sin', 
                             'Quarter_Cos', 'Volatility', 'Return']
    
    important_features = [col for col in all_possible_features if col in df.columns]
    lag_features = [f for f in ['Adj Close_LAG_1', 'Adj Close_LAG_4'] if f in df.columns]
    features = important_features + lag_features
    target = 'Adj Close'

    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = ElasticNet(alpha=0.001, l1_ratio=0.9)
    model.fit(X_scaled, y)

    future_data = df.copy()
    last_known = future_data.iloc[-1].copy()
    forecasts = []

    latest_quarter = last_known['Quarter']
    latest_year = last_known['YEAR']
    best_quarter = patterns['best_quarter']

    for i in range(n_quarters):
        next_quarter = (latest_quarter % 4) + 1
        next_year = int(latest_year) + (1 if next_quarter == 1 and latest_quarter == 4 else 0)
        next_date = pd.to_datetime(f"{next_year}-{(next_quarter - 1) * 3 + 1}-01")

        row = {
            'Quarter': next_quarter,
            'Quarter_Sin': np.sin(2 * np.pi * next_quarter / 4),
            'Quarter_Cos': np.cos(2 * np.pi * next_quarter / 4)
        }

        for feat in important_features:
            if feat in last_known:
                if feat in ['LONG_TERM_DEBT', 'QoQ_Growth', 'YoY_Growth', 'Volatility', 'Return']:
                    row[feat] = simulate_feature_value(df, feat, last_known[feat])
                else:
                    row[feat] = last_known[feat]
            elif feat == 'ADJ_CLOSE_ROLL_MEAN_4':
                row[feat] = future_data['Adj Close'].rolling(4).mean().iloc[-1]

        if 'Adj Close_LAG_1' in lag_features:
            row['Adj Close_LAG_1'] = last_known['Adj Close']
        if 'Adj Close_LAG_4' in lag_features:
            row['Adj Close_LAG_4'] = future_data['Adj Close'].iloc[-4] if len(future_data) >= 4 else last_known['Adj Close']

        x_input = pd.DataFrame([row])[features]
        x_scaled = scaler.transform(x_input)
        pred_close = model.predict(x_scaled)[0]

        seasonality_adj = patterns['seasonality'].get(next_quarter, 0)
        randomness = np.random.normal(0, 0.015 * (i + 1))
        realism_factor = 1 + seasonality_adj + randomness
        boost = 0.015 if next_quarter == best_quarter else 0
        adj_close_realistic = pred_close * (realism_factor + boost)

        forecast = {
            'Date': next_date,
            'YEAR': next_year,
            'Quarter': next_quarter,
            'Adj Close': adj_close_realistic,
            'Raw Prediction': pred_close,
            'Seasonality Adj': seasonality_adj,
            'Randomness': randomness,
            'Boost (Top Qtr)': boost
        }

        forecast_row = last_known.copy()
        forecast_row['YEAR'] = next_year
        forecast_row['Quarter'] = next_quarter
        forecast_row['Date'] = next_date
        forecast_row['Adj Close'] = adj_close_realistic
        future_data = pd.concat([future_data, pd.DataFrame([forecast_row])], ignore_index=True)

        last_known = forecast_row
        forecasts.append(forecast)
        latest_quarter = next_quarter
        latest_year = next_year

    return pd.DataFrame(forecasts)

def display_forecast(df_hist, df_forecast):
    print("=== QUARTERLY FORECAST RESULTS ===")
    df_forecast['Quarter'] = 'Q' + df_forecast['Quarter'].astype(str)
    df_forecast['Period'] = df_forecast['YEAR'].astype(str) + '-' + df_forecast['Quarter']
    df_forecast['Adj Close'] = df_forecast['Adj Close'].round(2)

    print(df_forecast[['Period', 'Adj Close', 'Seasonality Adj', 'Boost (Top Qtr)', 'Randomness']].to_string(index=False))
    annual_return = (df_forecast['Adj Close'].iloc[-1] / df_hist['Adj Close'].iloc[-1] - 1) * 100
    print(f"\nExpected 1-year return: {annual_return:.2f}%")

    plt.figure(figsize=(14, 8))
    plt.plot(df_hist['Date'].iloc[-8:], df_hist['Adj Close'].iloc[-8:], 'b-', label='Historical', linewidth=2)
    plt.plot(df_forecast['Date'], df_forecast['Adj Close'], 'r--', label='Forecast', linewidth=2)
    plt.scatter(df_forecast['Date'], df_forecast['Adj Close'], color='red', s=80)
    plt.title('Southwest Airlines (LUV) Forecast (ElasticNet + Monte Carlo + Dynamic Seasonality)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = "luv_Q1.csv"
    df = load_and_prepare_data(file_path)
    patterns = analyze_historical_patterns(df)
    forecast_df = forecast_next_quarters_elasticnet_realistic(df, patterns, n_quarters=5)
    display_forecast(df, forecast_df)
    return forecast_df

# Run the full pipeline
forecast_df = main()


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['Quarter'].astype(str) + '-01')
    df = df.sort_values('Date').reset_index(drop=True)
    df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
    df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
    df['Volatility'] = (df['High'] - df['Low']) / df['Low']
    df['Return'] = df['Adj Close'].pct_change()
    df['QoQ_Growth'] = df['Adj Close'].pct_change()
    df['YoY_Growth'] = df['Adj Close'].pct_change(4)
    df['ADJ_CLOSE_ROLL_MEAN_4'] = df['Adj Close'].rolling(window=4).mean()
    df = create_lag_features(df, ['Adj Close'], lags=[1, 4])
    return df.dropna().reset_index(drop=True)

def create_lag_features(df, features, lags=[1, 4]):
    for col in features:
        for lag in lags:
            df[f'{col}_LAG_{lag}'] = df[col].shift(lag)
    return df

def analyze_historical_patterns(df):
    quarter_returns = {}
    for q in [1, 2, 3, 4]:
        q_returns = []
        for year in df['YEAR'].unique():
            current = df[(df['YEAR'] == year) & (df['Quarter'] == q)]
            prev = df[(df['YEAR'] == year - 1) & (df['Quarter'] == q)]
            if not current.empty and not prev.empty:
                current_price = current['Adj Close'].values[0]
                prev_price = prev['Adj Close'].values[0]
                q_returns.append((current_price - prev_price) / prev_price)
        quarter_returns[q] = np.mean(q_returns) if q_returns else 0

    recent_years = sorted(df['YEAR'].unique())[-5:]
    quarter_scores = {1: 0, 2: 0, 3: 0, 4: 0}
    weight = 1.0
    annual_returns = []

    for year in reversed(recent_years):
        year_data = df[df['YEAR'] == year]
        if not year_data.empty:
            max_q = year_data.loc[year_data['Adj Close'].idxmax()]['Quarter']
            quarter_scores[int(max_q)] += weight
        weight *= 0.8
        if len(year_data) >= 2:
            start, end = year_data['Adj Close'].iloc[0], year_data['Adj Close'].iloc[-1]
            annual_returns.append((end / start) - 1)

    best_quarter = max(quarter_scores, key=quarter_scores.get)
    avg_annual_return = np.mean(annual_returns) if annual_returns else 0.08

    print("📊 Historical seasonality (avg QoQ return):")
    for q in range(1, 5):
        print(f"  Q{q}: {quarter_returns[q]:.4f}")

    print(f"\n🟩 Quarter with highest historical adj close (weighted): Q{best_quarter}")
    return {
        'seasonality': quarter_returns,
        'avg_annual_return': avg_annual_return,
        'best_quarter': best_quarter
    }

def simulate_feature_value(df, feature, last_val, n_simulations=100):
    qoq_changes = df[feature].pct_change().dropna()
    mu, sigma = qoq_changes.mean(), qoq_changes.std()
    if np.isnan(mu) or np.isnan(sigma):
        return last_val
    simulations = last_val * (1 + np.random.normal(mu, sigma, n_simulations))
    return np.mean(simulations)

def forecast_next_quarters_lasso(df, patterns, n_quarters=5):
    print(f"\nForecasting next {n_quarters} quarters using Lasso + Monte Carlo + Dynamic Seasonality...\n")

    selected_features = [
        'Part-time', 'TRANS_EXPENSES_LAG_1', 'PROP_EQUIP_LAG_4', 'PROP_EQUIP_GROUND', 
        'TRANS_EXPENSE', 'AD_EXP_CARGO', 'ADJ_CLOSE_ROLL_MEAN_4', 'QoQ_Growth', 'YoY_Growth', 
        'Quarter_Sin', 'Quarter_Cos', 'Volatility', 'Return', 'Adj Close_LAG_1', 'Adj Close_LAG_4',
        'Quarter', 'YEAR'
    ]
    target = 'Adj Close'
    selected_features = [f for f in selected_features if f in df.columns]

    X = df[selected_features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Lasso(alpha=0.01)
    model.fit(X_scaled, y)

    future_data = df.copy()
    last_known = future_data.iloc[-1].copy()
    forecasts = []

    latest_quarter = last_known['Quarter']
    latest_year = last_known['YEAR']
    best_quarter = patterns['best_quarter']

    for i in range(n_quarters):
        next_quarter = (latest_quarter % 4) + 1
        next_year = int(latest_year) + (1 if next_quarter == 1 and latest_quarter == 4 else 0)
        next_date = pd.to_datetime(f"{next_year}-{(next_quarter - 1) * 3 + 1}-01")

        row = {
            'Quarter': next_quarter,
            'YEAR': next_year,
            'Quarter_Sin': np.sin(2 * np.pi * next_quarter / 4),
            'Quarter_Cos': np.cos(2 * np.pi * next_quarter / 4)
        }

        for feat in selected_features:
            if feat in last_known:
                if feat in ['QoQ_Growth', 'YoY_Growth', 'Volatility', 'Return']:
                    row[feat] = simulate_feature_value(df, feat, last_known[feat])
                elif feat == 'ADJ_CLOSE_ROLL_MEAN_4':
                    row[feat] = future_data['Adj Close'].rolling(4).mean().iloc[-1]
                else:
                    row[feat] = last_known[feat]

        row['Adj Close_LAG_1'] = last_known['Adj Close']
        row['Adj Close_LAG_4'] = future_data['Adj Close'].iloc[-4] if len(future_data) >= 4 else last_known['Adj Close']

        x_input = pd.DataFrame([row])[selected_features]
        x_scaled = scaler.transform(x_input)
        pred_close = model.predict(x_scaled)[0]

        seasonality_adj = patterns['seasonality'].get(next_quarter, 0)
        randomness = np.random.normal(0, 0.015 * (i + 1))
        realism_factor = 1 + seasonality_adj + randomness
        boost = 0.015 if next_quarter == best_quarter else 0
        adj_close_realistic = pred_close * (realism_factor + boost)

        forecast = {
            'Date': next_date,
            'YEAR': next_year,
            'Quarter': next_quarter,
            'Adj Close': adj_close_realistic,
            'Raw Prediction': pred_close,
            'Seasonality Adj': seasonality_adj,
            'Randomness': randomness,
            'Boost (Top Qtr)': boost
        }

        forecast_row = last_known.copy()
        forecast_row.update(forecast)
        future_data = pd.concat([future_data, pd.DataFrame([forecast_row])], ignore_index=True)
        last_known = forecast_row
        forecasts.append(forecast)
        latest_quarter = next_quarter
        latest_year = next_year

    return pd.DataFrame(forecasts)

def display_forecast(df_hist, df_forecast):
    print("=== QUARTERLY FORECAST RESULTS ===")
    df_forecast['Quarter'] = 'Q' + df_forecast['Quarter'].astype(str)
    df_forecast['Period'] = df_forecast['YEAR'].astype(str) + '-' + df_forecast['Quarter']
    df_forecast['Adj Close'] = df_forecast['Adj Close'].round(2)

    print(df_forecast[['Period', 'Adj Close', 'Seasonality Adj', 'Boost (Top Qtr)', 'Randomness']].to_string(index=False))
    annual_return = (df_forecast['Adj Close'].iloc[-1] / df_hist['Adj Close'].iloc[-1] - 1) * 100
    print(f"\nExpected 1-year return: {annual_return:.2f}%")

    plt.figure(figsize=(14, 8))
    plt.plot(df_hist['Date'].iloc[-8:], df_hist['Adj Close'].iloc[-8:], 'b-', label='Historical', linewidth=2)
    plt.plot(df_forecast['Date'], df_forecast['Adj Close'], 'r--', label='Forecast', linewidth=2)
    plt.scatter(df_forecast['Date'], df_forecast['Adj Close'], color='red', s=80)
    plt.title('Southwest Airlines (LUV) Forecast (Lasso + Monte Carlo + Dynamic Seasonality)', fontsize=15)
    plt.xlabel('Date')
    plt.ylabel('Adj Close Price ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    file_path = "luv_Q1.csv"
    df = load_and_prepare_data(file_path)
    patterns = analyze_historical_patterns(df)
    forecast_df = forecast_next_quarters_lasso(df, patterns, n_quarters=5)
    display_forecast(df, forecast_df)
    return forecast_df

# Run the updated forecast
forecast_df = main()

