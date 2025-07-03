import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def prepare_features_efficient(df):
    """Efficiently prepare features for modeling"""
    
    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Select only essential columns to reduce memory
    essential_cols = [
        'datetime', 'NO2', 'Light_Count', 'Medium_Count', 'Heavy_Count',
        'total_vehicles', 'weighted_total', 'heavy_vehicle_impact_ratio',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour'
    ]
    
    # Keep only available columns
    cols_to_use = [col for col in essential_cols if col in df.columns]
    df_clean = df[cols_to_use].copy()
    
    # Handle missing values efficiently
    df_clean['NO2'] = df_clean['NO2'].fillna(method='ffill').fillna(method='bfill')
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
    
    # Create efficient lag features (only most important lags)
    important_lags = [1, 3, 6, 12, 24]
    
    for lag in important_lags:
        df_clean[f'NO2_lag_{lag}h'] = df_clean['NO2'].shift(lag)
        df_clean[f'traffic_lag_{lag}h'] = df_clean['weighted_total'].shift(lag)
    
    # Simple rolling means (only key windows)
    for window in [3, 6, 24]:
        df_clean[f'NO2_ma_{window}h'] = df_clean['NO2'].rolling(window=window, min_periods=1).mean()
        df_clean[f'traffic_ma_{window}h'] = df_clean['weighted_total'].rolling(window=window, min_periods=1).mean()
    
    # Drop rows with NaN in lag features
    df_clean = df_clean.dropna()
    
    return df_clean

def train_random_forest_model(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest model"""
    
    # Initialize model with optimized parameters
    rf_model = RandomForestRegressor(
        n_estimators=100,  # Balanced for speed and accuracy
        max_depth=20,      # Limit depth for speed
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,         # Use all cores
        random_state=42
    )
    
    # Train model
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Random Forest - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    
    return rf_model, y_pred_test

def train_decision_tree_model(X_train, y_train, X_test, y_test):
    """Train and evaluate Decision Tree model"""
    
    # Initialize model
    dt_model = DecisionTreeRegressor(
        max_depth=15,      # Prevent overfitting
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Train model
    print("Training Decision Tree...")
    dt_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = dt_model.predict(X_train)
    y_pred_test = dt_model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Decision Tree - Train R²: {train_r2:.3f}, Test R²: {test_r2:.3f}")
    print(f"Test MAE: {test_mae:.2f}, Test RMSE: {test_rmse:.2f}")
    
    return dt_model, y_pred_test

def train_linear_model(X_train, y_train, X_test, y_test):
    """Train simple linear model as baseline"""
    
    # Scale features for linear model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ridge regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_test = ridge_model.predict(X_test_scaled)
    
    # Calculate metrics
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"Ridge Regression - Test R²: {test_r2:.3f}, MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")
    
    return ridge_model, scaler, y_pred_test

def evaluate_models_time_series(df_prepared):
    """Main function to train and evaluate all models"""
    
    print("="*60)
    print("EFFICIENT TIME SERIES MODELING FOR NO2 PREDICTION")
    print("="*60)
    
    # Define features and target
    feature_cols = [col for col in df_prepared.columns 
                   if col not in ['datetime', 'NO2'] and not col.startswith('NO2_')]
    
    X = df_prepared[feature_cols].values
    y = df_prepared['NO2'].values
    dates = df_prepared['datetime'].values
    
    # Simple train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"\nDataset size: {len(X)} samples")
    print(f"Features: {len(feature_cols)}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train models
    print("\n" + "-"*40)
    rf_model, rf_pred = train_random_forest_model(X_train, y_train, X_test, y_test)
    
    print("\n" + "-"*40)
    dt_model, dt_pred = train_decision_tree_model(X_train, y_train, X_test, y_test)
    
    print("\n" + "-"*40)
    ridge_model, scaler, ridge_pred = train_linear_model(X_train, y_train, X_test, y_test)
    
    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "-"*40)
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Feature Importance
    ax = axes[0, 0]
    top_features = feature_importance.head(15)
    ax.barh(range(len(top_features)), top_features['importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Top 15 Feature Importances (Random Forest)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs Actual
    ax = axes[0, 1]
    # Plot only last 7 days for clarity
    last_n = min(168, len(y_test))  # 168 hours = 7 days
    x_axis = range(last_n)
    
    ax.plot(x_axis, y_test[-last_n:], label='Actual', alpha=0.8, linewidth=2)
    ax.plot(x_axis, rf_pred[-last_n:], label='Random Forest', alpha=0.7)
    ax.plot(x_axis, dt_pred[-last_n:], label='Decision Tree', alpha=0.7)
    ax.plot(x_axis, ridge_pred[-last_n:], label='Ridge', alpha=0.7, linestyle='--')
    ax.set_xlabel('Hours')
    ax.set_ylabel('NO2 (μg/m³)')
    ax.set_title('Model Predictions - Last 7 Days')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot - Random Forest
    ax = axes[1, 0]
    ax.scatter(y_test, rf_pred, alpha=0.5, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual NO2 (μg/m³)')
    ax.set_ylabel('Predicted NO2 (μg/m³)')
    ax.set_title(f'Random Forest: Actual vs Predicted (R² = {r2_score(y_test, rf_pred):.3f})')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals = y_test - rf_pred
    ax.scatter(rf_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Predicted NO2 (μg/m³)')
    ax.set_ylabel('Residuals')
    ax.set_title('Random Forest: Residual Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    models_performance = pd.DataFrame({
        'Model': ['Random Forest', 'Decision Tree', 'Ridge Regression'],
        'R²': [
            r2_score(y_test, rf_pred),
            r2_score(y_test, dt_pred),
            r2_score(y_test, ridge_pred)
        ],
        'MAE': [
            mean_absolute_error(y_test, rf_pred),
            mean_absolute_error(y_test, dt_pred),
            mean_absolute_error(y_test, ridge_pred)
        ],
        'RMSE': [
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            np.sqrt(mean_squared_error(y_test, dt_pred)),
            np.sqrt(mean_squared_error(y_test, ridge_pred))
        ]
    })
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(models_performance.to_string(index=False))
    
    # Return best model (Random Forest based on typical performance)
    return rf_model, feature_cols, models_performance

def make_future_predictions(model, df_prepared, feature_cols, hours_ahead=24):
    """Make efficient future predictions"""
    
    print(f"\n{'='*60}")
    print(f"FORECASTING NEXT {hours_ahead} HOURS")
    print(f"{'='*60}")
    
    # Get last known data
    last_data = df_prepared.iloc[-1]
    predictions = []
    
    # Simple approach: use last known features with slight modifications
    base_features = df_prepared[feature_cols].iloc[-1].values.copy()
    
    for hour in range(hours_ahead):
        # Adjust hour_of_day
        future_hour = (last_data['hour_of_day'] + hour + 1) % 24
        hour_idx = feature_cols.index('hour_of_day') if 'hour_of_day' in feature_cols else None
        
        if hour_idx is not None:
            base_features[hour_idx] = future_hour
        
        # Make prediction
        pred = model.predict(base_features.reshape(1, -1))[0]
        predictions.append(pred)
        
        # Update lag features for next prediction (simplified)
        # This is a basic approach - in production, you'd update all lags properly
    
    # Create forecast dataframe
    future_dates = pd.date_range(
        start=last_data['datetime'] + timedelta(hours=1),
        periods=hours_ahead,
        freq='H'
    )
    
    forecast_df = pd.DataFrame({
        'datetime': future_dates,
        'NO2_forecast': predictions
    })
    
    # Simple visualization
    plt.figure(figsize=(12, 6))
    
    # Plot last week of actual data
    last_week = df_prepared.tail(168)
    plt.plot(last_week['datetime'], last_week['NO2'], label='Historical NO2', alpha=0.8)
    plt.plot(forecast_df['datetime'], forecast_df['NO2_forecast'], 
             label='Forecast', color='red', linestyle='--', linewidth=2)
    
    plt.axvline(x=last_data['datetime'], color='black', linestyle=':', alpha=0.5)
    plt.xlabel('Date')
    plt.ylabel('NO2 (μg/m³)')
    plt.title('NO2 Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nForecast Summary:")
    print(f"Current NO2: {last_data['NO2']:.1f} μg/m³")
    print(f"Forecasted Average: {np.mean(predictions):.1f} μg/m³")
    print(f"Forecasted Maximum: {np.max(predictions):.1f} μg/m³")
    print(f"Forecasted Minimum: {np.min(predictions):.1f} μg/m³")
    
    return forecast_df

def cross_validate_time_series(df_prepared, n_splits=5):
    """Perform time series cross-validation"""
    
    print("\n" + "="*60)
    print("TIME SERIES CROSS-VALIDATION")
    print("="*60)
    
    # Prepare features
    feature_cols = [col for col in df_prepared.columns 
                   if col not in ['datetime', 'NO2'] and not col.startswith('NO2_')]
    
    X = df_prepared[feature_cols].values
    y = df_prepared['NO2'].values
    
    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store results
    cv_results = {
        'RandomForest': {'r2': [], 'mae': []},
        'DecisionTree': {'r2': [], 'mae': []}
    }
    
    print(f"Performing {n_splits}-fold time series cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}:")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, max_depth=15, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        cv_results['RandomForest']['r2'].append(r2_score(y_test, rf_pred))
        cv_results['RandomForest']['mae'].append(mean_absolute_error(y_test, rf_pred))
        
        # Decision Tree
        dt = DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=42)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        cv_results['DecisionTree']['r2'].append(r2_score(y_test, dt_pred))
        cv_results['DecisionTree']['mae'].append(mean_absolute_error(y_test, dt_pred))
        
        print(f"  RF - R²: {cv_results['RandomForest']['r2'][-1]:.3f}, MAE: {cv_results['RandomForest']['mae'][-1]:.2f}")
        print(f"  DT - R²: {cv_results['DecisionTree']['r2'][-1]:.3f}, MAE: {cv_results['DecisionTree']['mae'][-1]:.2f}")
    
    # Summary
    print("\n" + "-"*40)
    print("CROSS-VALIDATION SUMMARY")
    print("-"*40)
    
    for model_name, results in cv_results.items():
        avg_r2 = np.mean(results['r2'])
        std_r2 = np.std(results['r2'])
        avg_mae = np.mean(results['mae'])
        std_mae = np.std(results['mae'])
        
        print(f"\n{model_name}:")
        print(f"  R²:  {avg_r2:.3f} ± {std_r2:.3f}")
        print(f"  MAE: {avg_mae:.2f} ± {std_mae:.2f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # R² scores
    ax1.boxplot([cv_results['RandomForest']['r2'], cv_results['DecisionTree']['r2']], 
                labels=['Random Forest', 'Decision Tree'])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Cross-Validation R² Scores')
    ax1.grid(True, alpha=0.3)
    
    # MAE scores
    ax2.boxplot([cv_results['RandomForest']['mae'], cv_results['DecisionTree']['mae']], 
                labels=['Random Forest', 'Decision Tree'])
    ax2.set_ylabel('MAE')
    ax2.set_title('Cross-Validation MAE Scores')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return cv_results

