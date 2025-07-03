import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# STL DECOMPOSITION FOR TRAFFIC DATA
# ========================================

def perform_traffic_stl_decomposition(final_dataset, period='daily'):
    """
    Perform STL decomposition on traffic data
    """
    # Prepare time series data
    if 'datetime' not in final_dataset.columns:
        print("Datetime column not found")
        return None
    
    # Sort by datetime
    ts_data = final_dataset.sort_values('datetime').copy()
    
    # Create different aggregation levels
    if period == 'hourly':
        # Hourly data
        traffic_ts = ts_data.groupby('datetime')['total_vehicles'].mean()
        seasonal_period = 24  # Daily seasonality
        title_suffix = "Hourly"
    elif period == 'daily':
        # Daily aggregation
        traffic_ts = ts_data.groupby(ts_data['datetime'].dt.date)['total_vehicles'].sum()
        traffic_ts.index = pd.to_datetime(traffic_ts.index)
        seasonal_period = 7  # Weekly seasonality
        title_suffix = "Daily"
    elif period == 'weekly':
        # Weekly aggregation
        ts_data['week'] = ts_data['datetime'].dt.to_period('W')
        traffic_ts = ts_data.groupby('week')['total_vehicles'].sum()
        traffic_ts.index = traffic_ts.index.to_timestamp()
        seasonal_period = 52  # Yearly seasonality
        title_suffix = "Weekly"
        
    # Check if we have enough data
    if len(traffic_ts) < 2 * seasonal_period:
        print(f"Insufficient data for {period} STL decomposition")
        return None
    
    # Handle missing values
    traffic_ts = traffic_ts.fillna(method='ffill').fillna(method='bfill')
    
    # Perform STL decomposition
    stl = STL(traffic_ts, seasonal=seasonal_period, robust=True)
    decomposition = stl.fit()
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'STL Decomposition - Traffic Volume ({title_suffix})', fontsize=16, fontweight='bold')
    
    # Original series
    axes[0].plot(traffic_ts.index, traffic_ts.values, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=12)
    axes[0].set_title(f'Original {title_suffix} Traffic Volume', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Trend component
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'r-', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=12)
    axes[1].set_title('Trend Component', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal component
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'g-', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=12)
    axes[2].set_title('Seasonal Component', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Residual component
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'orange', linewidth=1, alpha=0.8)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Residual', fontsize=12)
    axes[3].set_title('Residual Component', fontsize=14)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    trend_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid)
    seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
    
    return {
        'decomposition': decomposition,
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength,
        'period': period,
        'seasonal_period': seasonal_period
    }

# ========================================
# STL DECOMPOSITION FOR AIR QUALITY DATA
# ========================================

def perform_airquality_stl_decomposition(final_dataset, pollutant='NO2'):
    """
    Perform STL decomposition on air quality data
    """
    if pollutant not in final_dataset.columns:
        print(f"{pollutant} not found in dataset")
        return None
    
    # Prepare time series data
    ts_data = final_dataset.sort_values('datetime').copy()
    
    # Create daily aggregation for air quality (more stable for decomposition)
    pollution_ts = ts_data.groupby(ts_data['datetime'].dt.date)[pollutant].mean()
    pollution_ts.index = pd.to_datetime(pollution_ts.index)
    
    # Handle missing values
    pollution_ts = pollution_ts.fillna(method='ffill').fillna(method='bfill')
    
    # Check if we have enough data
    if len(pollution_ts) < 14:  # At least 2 weeks
        print(f"Insufficient data for {pollutant} STL decomposition")
        return None
    
    # Perform STL decomposition
    stl = STL(pollution_ts, seasonal=7, robust=True)  # Weekly seasonality
    decomposition = stl.fit()
    
    # Create visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle(f'STL Decomposition - {pollutant} Concentration', fontsize=16, fontweight='bold')
    
    # Original series
    axes[0].plot(pollution_ts.index, pollution_ts.values, 'purple', linewidth=1.5)
    axes[0].set_ylabel('Original', fontsize=12)
    axes[0].set_title(f'Original Daily {pollutant} Concentration (μg/m³)', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Add WHO/EU guidelines
    if pollutant == 'NO2':
        axes[0].axhline(y=40, color='red', linestyle='--', alpha=0.5, label='EU Annual Limit')
    elif pollutant == 'PM10':
        axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, label='EU Daily Limit')
    
    # Trend component
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, 'r-', linewidth=2)
    axes[1].set_ylabel('Trend', fontsize=12)
    axes[1].set_title('Trend Component', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Seasonal component
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, 'g-', linewidth=1.5)
    axes[2].set_ylabel('Seasonal', fontsize=12)
    axes[2].set_title('Weekly Seasonal Component', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Residual component
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, 'orange', linewidth=1, alpha=0.8)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Residual', fontsize=12)
    axes[3].set_title('Residual Component', fontsize=14)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    trend_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid)
    seasonal_strength = 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid)
    
    return {
        'decomposition': decomposition,
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength,
        'pollutant': pollutant
    }

# ========================================
# MULTI-SCALE DECOMPOSITION ANALYSIS
# ========================================

def perform_multiscale_decomposition(final_dataset):
    """
    Perform decomposition at multiple time scales
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Hourly pattern extraction (24-hour cycle)
    ax1 = fig.add_subplot(gs[0, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles']):
        # Calculate average hourly pattern
        hourly_pattern = final_dataset.groupby('hour_of_day')['total_vehicles'].mean()
        
        # Fit a smooth curve using Fourier series
        hours = np.arange(24)
        # Use first 3 harmonics for smooth pattern
        fourier_fit = np.zeros(24)
        for k in range(1, 4):
            a_k = 2/24 * np.sum(hourly_pattern.values * np.cos(2*np.pi*k*hours/24))
            b_k = 2/24 * np.sum(hourly_pattern.values * np.sin(2*np.pi*k*hours/24))
            fourier_fit += a_k * np.cos(2*np.pi*k*hours/24) + b_k * np.sin(2*np.pi*k*hours/24)
        
        fourier_fit += hourly_pattern.mean()  # Add DC component
        
        # Plot
        ax1.bar(hours, hourly_pattern.values, alpha=0.5, color='skyblue', label='Observed')
        ax1.plot(hours, fourier_fit, 'r-', linewidth=3, label='Fourier Fit')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('Daily Seasonal Pattern Extraction', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 3))
    
    # 2. Weekly pattern extraction
    ax2 = fig.add_subplot(gs[1, 0])
    
    if all(col in final_dataset.columns for col in ['day_of_week', 'total_vehicles']):
        # Calculate average daily pattern
        daily_pattern = final_dataset.groupby('day_of_week')['total_vehicles'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Normalize to show relative changes
        daily_normalized = daily_pattern / daily_pattern.mean()
        
        # Plot
        ax2.bar(range(7), daily_normalized.values, color=['steelblue']*5 + ['lightcoral']*2, alpha=0.8)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Day of Week', fontsize=12)
        ax2.set_ylabel('Relative Traffic Level', fontsize=12)
        ax2.set_title('Weekly Seasonal Pattern', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(days)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, val in enumerate(daily_normalized.values):
            ax2.text(i, val + 0.01, f'{(val-1)*100:+.1f}%', ha='center', fontsize=9)
    
    # 3. Monthly/Seasonal pattern
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'month' in final_dataset.columns:
        monthly_pattern = final_dataset.groupby('month')['total_vehicles'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Normalize
        monthly_normalized = monthly_pattern / monthly_pattern.mean()
        
        # Define season colors
        season_colors = ['lightblue', 'lightblue', 'lightgreen', 'lightgreen', 'lightgreen', 
                        'yellow', 'yellow', 'yellow', 'orange', 'orange', 'orange', 'lightblue']
        
        # Plot
        bars = ax3.bar(range(1, len(monthly_pattern)+1), monthly_normalized.values, 
                       color=season_colors[:len(monthly_pattern)], alpha=0.8, edgecolor='black')
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel('Relative Traffic Level', fontsize=12)
        ax3.set_title('Monthly/Seasonal Pattern', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(1, len(monthly_pattern)+1))
        ax3.set_xticklabels(months[:len(monthly_pattern)])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, val in enumerate(monthly_normalized.values):
            ax3.text(i+1, val + 0.01, f'{(val-1)*100:+.1f}%', ha='center', fontsize=9)
    
    # 4. Long-term trend analysis
    ax4 = fig.add_subplot(gs[2, :])
    
    if 'datetime' in final_dataset.columns:
        # Create daily aggregation
        daily_traffic = final_dataset.groupby(final_dataset['datetime'].dt.date)['total_vehicles'].sum()
        daily_traffic.index = pd.to_datetime(daily_traffic.index)
        
        # Apply different smoothing techniques
        # Moving average
        ma_7 = daily_traffic.rolling(window=7, center=True).mean()
        ma_30 = daily_traffic.rolling(window=30, center=True).mean()
        
        # Polynomial trend
        x = np.arange(len(daily_traffic))
        if len(x) > 3:
            poly_coeffs = np.polyfit(x, daily_traffic.values, 2)
            poly_trend = np.poly1d(poly_coeffs)(x)
        
        # Plot
        ax4.plot(daily_traffic.index, daily_traffic.values, alpha=0.3, color='gray', 
                linewidth=1, label='Daily Data')
        ax4.plot(ma_7.index, ma_7.values, 'b-', linewidth=2, label='7-day MA')
        ax4.plot(ma_30.index, ma_30.values, 'r-', linewidth=2.5, label='30-day MA')
        if len(x) > 3:
            ax4.plot(daily_traffic.index, poly_trend, 'g--', linewidth=2, label='Polynomial Trend')
        
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Total Daily Traffic', fontsize=12)
        ax4.set_title('Long-term Trend Analysis', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# ========================================
# DETRENDED AND DESEASONALIZED ANALYSIS
# ========================================

def analyze_detrended_deseasonalized_data(final_dataset, decomposition_results):
    """
    Analyze patterns after removing trend and seasonal components
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Detrended traffic analysis
    ax1 = fig.add_subplot(gs[0, :])
    
    if decomposition_results and 'decomposition' in decomposition_results:
        decomp = decomposition_results['decomposition']
        
        # Calculate detrended series
        detrended = decomp.observed - decomp.trend
        
        # Plot
        ax1.plot(detrended.index, detrended.values, 'b-', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.fill_between(detrended.index, 0, detrended.values, alpha=0.3, color='skyblue')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Detrended Values', fontsize=12)
        ax1.set_title('Detrended Traffic Volume (Original - Trend)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        std_dev = detrended.std()
        ax1.axhline(y=std_dev, color='red', linestyle=':', alpha=0.5, label=f'±1 STD: {std_dev:.0f}')
        ax1.axhline(y=-std_dev, color='red', linestyle=':', alpha=0.5)
        ax1.legend()
    
    # 2. Deseasonalized traffic analysis
    ax2 = fig.add_subplot(gs[1, :])
    
    if decomposition_results and 'decomposition' in decomposition_results:
        decomp = decomposition_results['decomposition']
        
        # Calculate deseasonalized series
        deseasonalized = decomp.observed - decomp.seasonal
        
        # Plot with trend
        ax2.plot(deseasonalized.index, deseasonalized.values, 'g-', linewidth=1.5, 
                alpha=0.8, label='Deseasonalized')
        ax2.plot(decomp.trend.index, decomp.trend.values, 'r-', linewidth=2.5, 
                label='Trend', alpha=0.8)
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Traffic Volume', fontsize=12)
        ax2.set_title('Deseasonalized Traffic Volume vs Trend', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. Residual analysis
    ax3 = fig.add_subplot(gs[2, 0])
    
    if decomposition_results and 'decomposition' in decomposition_results:
        residuals = decomposition_results['decomposition'].resid.dropna()
        
        # Histogram of residuals
        ax3.hist(residuals, bins=50, density=True, alpha=0.7, color='orange', edgecolor='black')
        
        # Fit normal distribution
        mu, std = residuals.mean(), residuals.std()
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax3.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2), 
                'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')
        
        ax3.set_xlabel('Residual Value', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Autocorrelation of residuals
    ax4 = fig.add_subplot(gs[2, 1])
    
    if decomposition_results and 'decomposition' in decomposition_results:
        residuals = decomposition_results['decomposition'].resid.dropna()
        
        # Calculate autocorrelation
        lags = range(1, min(25, len(residuals)//4))
        autocorr = [residuals.autocorr(lag=lag) for lag in lags]
        
        # Plot
        ax4.bar(lags, autocorr, color='purple', alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linewidth=0.5)
        
        # Add confidence bounds (95%)
        n = len(residuals)
        confidence = 1.96 / np.sqrt(n)
        ax4.axhline(y=confidence, color='red', linestyle='--', alpha=0.5, label='95% CI')
        ax4.axhline(y=-confidence, color='red', linestyle='--', alpha=0.5)
        
        ax4.set_xlabel('Lag', fontsize=12)
        ax4.set_ylabel('Autocorrelation', fontsize=12)
        ax4.set_title('Residual Autocorrelation', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Test for randomness
        significant_lags = sum(1 for ac in autocorr if abs(ac) > confidence)
        ax4.text(0.02, 0.98, f'Significant lags: {significant_lags}/{len(autocorr)}',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# ========================================
# COMPARATIVE DECOMPOSITION ANALYSIS
# ========================================
def compare_traffic_pollution_decomposition(final_dataset):
    """
    Compare decomposition patterns between traffic and pollution
    """
    # Prepare data
    ts_data = final_dataset.sort_values('datetime').copy()
    
    # Daily aggregation
    daily_traffic = ts_data.groupby(ts_data['datetime'].dt.date)['total_vehicles'].sum()
    daily_traffic.index = pd.to_datetime(daily_traffic.index)
    
    pollutants_available = []
    daily_pollutants = {}
    
    for pollutant in ['NO2', 'PM10', 'PM25']:
        if pollutant in ts_data.columns:
            daily_poll = ts_data.groupby(ts_data['datetime'].dt.date)[pollutant].mean()
            daily_poll.index = pd.to_datetime(daily_poll.index)
            # Remove any NaN values
            daily_poll = daily_poll.dropna()
            if len(daily_poll) > 14:  # Only include if we have enough data
                daily_pollutants[pollutant] = daily_poll
                pollutants_available.append(pollutant)
    
    if len(pollutants_available) == 0:
        print("No pollutant data available with sufficient observations")
        return
    
    # Create comparison plot
    n_vars = 1 + len(pollutants_available)
    fig, axes = plt.subplots(n_vars, 3, figsize=(18, 4*n_vars))
    if n_vars == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Comparative STL Decomposition: Traffic vs Pollutants', fontsize=16, fontweight='bold')
    
    # Store decomposition results for summary
    decomp_results = {}
    
    # Traffic decomposition
    if len(daily_traffic) > 14:
        try:
            # Fill any missing values
            daily_traffic = daily_traffic.fillna(method='ffill').fillna(method='bfill')
            
            stl_traffic = STL(daily_traffic, seasonal=7, robust=True)
            decomp_traffic = stl_traffic.fit()
            decomp_results['traffic'] = decomp_traffic
            
            # Normalize components for comparison
            trend_norm = (decomp_traffic.trend - decomp_traffic.trend.mean()) / decomp_traffic.trend.std()
            seasonal_norm = decomp_traffic.seasonal / decomp_traffic.observed.std()
            
            # Plot trend
            axes[0, 0].plot(trend_norm.index, trend_norm.values, 'b-', linewidth=2)
            axes[0, 0].set_title('Traffic - Normalized Trend', fontsize=12)
            axes[0, 0].set_ylabel('Traffic', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Plot seasonal pattern - show 2 complete weeks
            seasonal_cycle = decomp_traffic.seasonal[:14].values
            days = np.tile(['M', 'T', 'W', 'T', 'F', 'S', 'S'], 2)
            x_pos = np.arange(14)
            
            axes[0, 1].plot(x_pos, seasonal_norm[:14].values, 'b-', linewidth=2, marker='o')
            axes[0, 1].set_title('Traffic - Seasonal Pattern (2 weeks)', fontsize=12)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels(days)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(-0.5, 13.5)
            
            # Plot residual distribution
            residuals = decomp_traffic.resid.dropna()
            axes[0, 2].hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 2].set_title('Traffic - Residual Distribution', fontsize=12)
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, std = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            axes[0, 2].plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2), 
                           'r-', linewidth=2, alpha=0.8)
            
        except Exception as e:
            print(f"Error in traffic decomposition: {e}")
            axes[0, 0].text(0.5, 0.5, 'Error in decomposition', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 1].text(0.5, 0.5, 'Error in decomposition', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 2].text(0.5, 0.5, 'Error in decomposition', ha='center', va='center', transform=axes[0, 2].transAxes)
    
    # Pollutant decompositions
    colors = {'NO2': 'red', 'PM10': 'green', 'PM25': 'purple'}
    
    for i, pollutant in enumerate(pollutants_available):
        row = i + 1
        daily_data = daily_pollutants[pollutant]
        
        try:
            # Fill any missing values
            daily_data = daily_data.fillna(method='ffill').fillna(method='bfill')
            
            stl_poll = STL(daily_data, seasonal=7, robust=True)
            decomp_poll = stl_poll.fit()
            decomp_results[pollutant] = decomp_poll
            
            # Normalize components
            trend_norm = (decomp_poll.trend - decomp_poll.trend.mean()) / decomp_poll.trend.std()
            seasonal_norm = decomp_poll.seasonal / decomp_poll.observed.std()
            
            color = colors.get(pollutant, 'gray')
            
            # Plot trend
            axes[row, 0].plot(trend_norm.index, trend_norm.values, color=color, linewidth=2)
            axes[row, 0].set_title(f'{pollutant} - Normalized Trend', fontsize=12)
            axes[row, 0].set_ylabel(pollutant, fontsize=12, fontweight='bold')
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].tick_params(axis='x', rotation=45)
            
            # Plot seasonal pattern - ensure we have enough data
            if len(seasonal_norm) >= 14:
                seasonal_values = seasonal_norm[:14].values
            else:
                # Repeat the pattern to get 14 days
                n_repeat = int(np.ceil(14 / len(seasonal_norm)))
                seasonal_values = np.tile(seasonal_norm.values, n_repeat)[:14]
            
            x_pos = np.arange(14)
            axes[row, 1].plot(x_pos, seasonal_values, color=color, linewidth=2, marker='o')
            axes[row, 1].set_title(f'{pollutant} - Seasonal Pattern (2 weeks)', fontsize=12)
            axes[row, 1].set_xticks(x_pos)
            axes[row, 1].set_xticklabels(days)
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].set_xlim(-0.5, 13.5)
            
            # Add zero line for reference
            axes[row, 1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            # Plot residual distribution
            residuals = decomp_poll.resid.dropna()
            axes[row, 2].hist(residuals, bins=30, density=True, alpha=0.7, color=color, edgecolor='black')
            axes[row, 2].set_title(f'{pollutant} - Residual Distribution', fontsize=12)
            axes[row, 2].grid(True, alpha=0.3)
            
            # Add normal distribution overlay
            mu, std = residuals.mean(), residuals.std()
            if std > 0:  # Avoid division by zero
                x = np.linspace(residuals.min(), residuals.max(), 100)
                axes[row, 2].plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2), 
                               'k-', linewidth=2, alpha=0.8)
            
        except Exception as e:
            print(f"Error in {pollutant} decomposition: {e}")
            axes[row, 0].text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', 
                             transform=axes[row, 0].transAxes)
            axes[row, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                             transform=axes[row, 1].transAxes)
            axes[row, 2].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                             transform=axes[row, 2].transAxes)
    
    # Set common labels and formatting
    for ax in axes[:, 0]:
        ax.set_xlabel('Date', fontsize=10)
    
    for ax in axes[:, 1]:
        ax.set_xlabel('Day', fontsize=10)
        ax.set_ylabel('Normalized Seasonal', fontsize=10)
    
    for ax in axes[:, 2]:
        ax.set_xlabel('Residual Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print decomposition summary
    print("\n" + "="*60)
    print("DECOMPOSITION SUMMARY")
    print("="*60)
    
    for var_name, decomp in decomp_results.items():
        # Calculate strength metrics
        trend_strength = 1 - np.var(decomp.resid.dropna()) / np.var(decomp.trend.dropna() + decomp.resid.dropna())
        seasonal_strength = 1 - np.var(decomp.resid.dropna()) / np.var(decomp.seasonal + decomp.resid.dropna())
        
        print(f"\n{var_name}:")
        print(f"  Trend Strength: {trend_strength:.3f}")
        print(f"  Seasonal Strength: {seasonal_strength:.3f}")
        print(f"  Residual Std Dev: {decomp.resid.std():.3f}")
        
        # Check correlation between traffic and pollutant trends
        if var_name != 'traffic' and 'traffic' in decomp_results:
            # Align the series by common dates
            common_dates = decomp.trend.index.intersection(decomp_results['traffic'].trend.index)
            if len(common_dates) > 10:
                corr = decomp.trend[common_dates].corr(decomp_results['traffic'].trend[common_dates])
                print(f"  Trend correlation with traffic: {corr:.3f}")
    
    return decomp_results


def analyze_detrended_deseasonalized_airquality(final_dataset, pollutant='NO2'):
    """
    Analyze patterns after removing trend and seasonal components for air quality data
    """
    if pollutant not in final_dataset.columns:
        print(f"{pollutant} not found in dataset")
        return
    
    # Prepare time series data
    ts_data = final_dataset.sort_values('datetime').copy()
    
    # Create daily aggregation for more stable decomposition
    daily_data = ts_data.groupby(ts_data['datetime'].dt.date)[pollutant].mean()
    daily_data.index = pd.to_datetime(daily_data.index)
    daily_data = daily_data.dropna()
    
    if len(daily_data) < 14:
        print(f"Insufficient data for {pollutant} decomposition analysis")
        return
    
    # Perform STL decomposition
    stl = STL(daily_data, seasonal=7, robust=True)
    decomp = stl.fit()
    
    # Create visualization with corrected layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)
    
    # 1. Detrended data analysis (top, full width)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Calculate detrended series
    detrended = decomp.observed - decomp.trend
    
    # Plot
    ax1.plot(detrended.index, detrended.values, 'darkred', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(detrended.index, 0, detrended.values, 
                     where=(detrended.values > 0), interpolate=True, 
                     alpha=0.3, color='red', label='Above trend')
    ax1.fill_between(detrended.index, 0, detrended.values, 
                     where=(detrended.values <= 0), interpolate=True, 
                     alpha=0.3, color='blue', label='Below trend')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Detrended Values', fontsize=12)
    ax1.set_title(f'Detrended {pollutant} Concentration (Original - Trend)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    std_dev = detrended.std()
    ax1.axhline(y=std_dev, color='red', linestyle=':', alpha=0.5, label=f'±1 STD: {std_dev:.1f}')
    ax1.axhline(y=-std_dev, color='red', linestyle=':', alpha=0.5)
    ax1.axhline(y=2*std_dev, color='orange', linestyle=':', alpha=0.3, label=f'±2 STD: {2*std_dev:.1f}')
    ax1.axhline(y=-2*std_dev, color='orange', linestyle=':', alpha=0.3)
    ax1.legend(loc='best')
    
    # 2. Deseasonalized data analysis (middle, full width)
    ax2 = fig.add_subplot(gs[1, :])
    
    # Calculate deseasonalized series
    deseasonalized = decomp.observed - decomp.seasonal
    
    # Plot with trend
    ax2.plot(deseasonalized.index, deseasonalized.values, 'green', linewidth=1.5, 
            alpha=0.8, label='Deseasonalized')
    ax2.plot(decomp.trend.index, decomp.trend.values, 'darkred', linewidth=2.5, 
            label='Trend', alpha=0.8)
    
    # Add confidence band around trend
    trend_std = (deseasonalized - decomp.trend).std()
    ax2.fill_between(decomp.trend.index, 
                    decomp.trend - trend_std, 
                    decomp.trend + trend_std, 
                    alpha=0.2, color='red', label='±1 STD band')
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel(f'{pollutant} Concentration (μg/m³)', fontsize=12)
    ax2.set_title(f'Deseasonalized {pollutant} Concentration vs Trend', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add reference lines for air quality standards
    if pollutant == 'NO2':
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, linewidth=2, label='EU Annual Limit')
    elif pollutant == 'PM10':
        ax2.axhline(y=40, color='red', linestyle='--', alpha=0.5, linewidth=2, label='EU Annual Limit')
    
    # 3. Residual analysis (bottom left)
    ax3 = fig.add_subplot(gs[2, 0])
    
    residuals = decomp.resid.dropna()
    
    # Histogram of residuals
    n, bins, patches = ax3.hist(residuals, bins=30, density=True, alpha=0.7, 
                               color='darkorange', edgecolor='black')
    
    # Color bars based on value
    for i, patch in enumerate(patches):
        if bins[i] < -residuals.std():
            patch.set_facecolor('darkblue')
        elif bins[i] > residuals.std():
            patch.set_facecolor('darkred')
    
    # Fit normal distribution
    mu, std = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax3.plot(x, 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2), 
            'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')
    
    ax3.set_xlabel('Residual Value', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add normality test
    from scipy import stats as scipy_stats
    _, p_value = scipy_stats.normaltest(residuals)
    ax3.text(0.02, 0.98, f'Normality test p-value: {p_value:.4f}', 
            transform=ax3.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Autocorrelation of residuals (bottom right)
    ax4 = fig.add_subplot(gs[2, 1])
    
    # Calculate autocorrelation
    lags = range(1, min(25, len(residuals)//4))
    autocorr = [residuals.autocorr(lag=lag) for lag in lags]
    
    # Plot
    bars = ax4.bar(lags, autocorr, color='purple', alpha=0.7, edgecolor='black')
    
    # Color significant lags
    n = len(residuals)
    confidence = 1.96 / np.sqrt(n)
    
    for i, (lag, ac) in enumerate(zip(lags, autocorr)):
        if abs(ac) > confidence:
            bars[i].set_facecolor('darkred')
            bars[i].set_alpha(0.9)
    
    ax4.axhline(y=0, color='black', linewidth=0.5)
    
    # Add confidence bounds (95%)
    ax4.axhline(y=confidence, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax4.axhline(y=-confidence, color='red', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Lag', fontsize=12)
    ax4.set_ylabel('Autocorrelation', fontsize=12)
    ax4.set_title('Residual Autocorrelation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Test for randomness
    significant_lags = sum(1 for ac in autocorr if abs(ac) > confidence)
    ax4.text(0.02, 0.98, f'Significant lags: {significant_lags}/{len(autocorr)}',
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Add Ljung-Box test result
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=10, return_df=True)
    min_pvalue = lb_result['lb_pvalue'].min()
    ax4.text(0.02, 0.88, f'Ljung-Box test (10 lags)\nMin p-value: {min_pvalue:.4f}',
            transform=ax4.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle(f'{pollutant} Decomposition Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return decomp
