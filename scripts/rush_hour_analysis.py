import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# RUSH HOUR IDENTIFICATION AND ANALYSIS
# ========================================

def identify_rush_hours(final_dataset):
    """
    Identify rush hour periods based on traffic patterns
    """
    if 'hour_of_day' not in final_dataset.columns or 'total_vehicles' not in final_dataset.columns:
        print("Required columns not found")
        return None
    
    # Calculate hourly averages
    hourly_traffic = final_dataset.groupby('hour_of_day')['total_vehicles'].mean()
    
    # Find peaks using scipy
    peaks, properties = find_peaks(hourly_traffic.values, height=hourly_traffic.mean(), distance=3)
    
    # Identify morning and evening rush hours
    morning_peak = None
    evening_peak = None
    
    for peak in peaks:
        if 6 <= peak <= 10:  # Morning window
            morning_peak = peak
        elif 16 <= peak <= 20:  # Evening window
            evening_peak = peak
    
    # Define rush hour windows (±1 hour from peak)
    rush_hours = {
        'morning_peak': morning_peak,
        'morning_window': (max(0, morning_peak-1), min(23, morning_peak+1)) if morning_peak else (7, 9),
        'evening_peak': evening_peak,
        'evening_window': (max(0, evening_peak-1), min(23, evening_peak+1)) if evening_peak else (17, 19),
        'hourly_traffic': hourly_traffic
    }
    
    return rush_hours

def visualize_rush_hour_patterns(final_dataset):
    """
    Comprehensive visualization of rush hour patterns
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Get rush hour information
    rush_info = identify_rush_hours(final_dataset)
    
    # 1. Overall hourly pattern with rush hour identification
    ax1 = fig.add_subplot(gs[0, :])
    
    if rush_info and 'hour_of_day' in final_dataset.columns:
        hourly_stats = final_dataset.groupby('hour_of_day')['total_vehicles'].agg(['mean', 'std', 'count'])
        
        # Calculate confidence intervals
        ci = 1.96 * (hourly_stats['std'] / np.sqrt(hourly_stats['count']))
        
        # Plot main pattern
        ax1.plot(hourly_stats.index, hourly_stats['mean'], 'o-', 
                color='darkblue', linewidth=3, markersize=8, label='Average Traffic')
        ax1.fill_between(hourly_stats.index, 
                        hourly_stats['mean'] - ci, 
                        hourly_stats['mean'] + ci, 
                        alpha=0.3, color='skyblue', label='95% CI')
        
        # Highlight rush hours
        morning_start, morning_end = rush_info['morning_window']
        evening_start, evening_end = rush_info['evening_window']
        
        ax1.axvspan(morning_start, morning_end, alpha=0.2, color='orange', label='Morning Rush')
        ax1.axvspan(evening_start, evening_end, alpha=0.2, color='red', label='Evening Rush')
        
        # Mark peaks
        if rush_info['morning_peak']:
            ax1.axvline(rush_info['morning_peak'], color='orange', linestyle='--', linewidth=2)
            ax1.annotate(f"Morning Peak\n{rush_info['morning_peak']}:00", 
                        xy=(rush_info['morning_peak'], hourly_stats['mean'][rush_info['morning_peak']]),
                        xytext=(rush_info['morning_peak'], hourly_stats['mean'][rush_info['morning_peak']] + hourly_stats['std'].max()*0.3),
                        ha='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='orange'))
        
        if rush_info['evening_peak']:
            ax1.axvline(rush_info['evening_peak'], color='red', linestyle='--', linewidth=2)
            ax1.annotate(f"Evening Peak\n{rush_info['evening_peak']}:00", 
                        xy=(rush_info['evening_peak'], hourly_stats['mean'][rush_info['evening_peak']]),
                        xytext=(rush_info['evening_peak'], hourly_stats['mean'][rush_info['evening_peak']] + hourly_stats['std'].max()*0.3),
                        ha='center', fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                        arrowprops=dict(arrowstyle='->', color='red'))
        
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('Daily Traffic Pattern with Rush Hour Identification', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
    
    # 2. Morning vs Evening Rush Hour Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    
    if rush_info and all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles']):
        # Extract rush hour data
        morning_start, morning_end = rush_info['morning_window']
        evening_start, evening_end = rush_info['evening_window']
        
        morning_data = final_dataset[final_dataset['hour_of_day'].between(morning_start, morning_end)]['total_vehicles']
        evening_data = final_dataset[final_dataset['hour_of_day'].between(evening_start, evening_end)]['total_vehicles']
        
        # Create violin plots
        parts = ax2.violinplot([morning_data.dropna(), evening_data.dropna()], 
                              positions=[0, 1], showmeans=True, showmedians=True)
        
        # Customize colors
        colors = ['orange', 'red']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Add statistics
        morning_mean = morning_data.mean()
        evening_mean = evening_data.mean()
        
        ax2.text(0, morning_mean, f'{morning_mean:.0f}', ha='center', va='bottom', fontweight='bold')
        ax2.text(1, evening_mean, f'{evening_mean:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Morning Rush\n({}:00-{}:00)'.format(morning_start, morning_end),
                            'Evening Rush\n({}:00-{}:00)'.format(evening_start, evening_end)])
        ax2.set_ylabel('Vehicle Count', fontsize=12)
        ax2.set_title('Morning vs Evening Rush Hour Traffic Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add comparison text
        diff_pct = ((evening_mean - morning_mean) / morning_mean) * 100
        ax2.text(0.5, ax2.get_ylim()[1]*0.95, 
                f'Evening traffic is {diff_pct:+.1f}% compared to morning',
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 3. Rush Hour Intensity by Day of Week
    ax3 = fig.add_subplot(gs[1, 1])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'day_of_week', 'total_vehicles']):
        # Calculate rush hour intensity for each day
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        morning_intensity = []
        evening_intensity = []
        
        morning_start, morning_end = rush_info['morning_window']
        evening_start, evening_end = rush_info['evening_window']
        
        for day in range(7):
            day_data = final_dataset[final_dataset['day_of_week'] == day]
            
            morning_traffic = day_data[day_data['hour_of_day'].between(morning_start, morning_end)]['total_vehicles'].mean()
            evening_traffic = day_data[day_data['hour_of_day'].between(evening_start, evening_end)]['total_vehicles'].mean()
            
            morning_intensity.append(morning_traffic)
            evening_intensity.append(evening_traffic)
        
        # Create grouped bar chart
        x = np.arange(7)
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, morning_intensity, width, label='Morning Rush', color='orange', alpha=0.8)
        bars2 = ax3.bar(x + width/2, evening_intensity, width, label='Evening Rush', color='red', alpha=0.8)
        
        ax3.set_xlabel('Day of Week', fontsize=12)
        ax3.set_ylabel('Average Rush Hour Traffic', fontsize=12)
        ax3.set_title('Rush Hour Intensity by Day of Week', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(days)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Highlight weekends
        ax3.axvspan(4.5, 6.5, alpha=0.1, color='gray')
    
    # 4. Heavy Vehicle Rush Hour Patterns
    ax4 = fig.add_subplot(gs[2, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'Heavy_Count', 'Light_Count']):
        # Calculate heavy vehicle percentage by hour
        hourly_heavy = final_dataset.groupby('hour_of_day').agg({
            'Heavy_Count': 'sum',
            'Light_Count': 'sum',
            'total_vehicles': 'sum'
        })
        hourly_heavy['heavy_percentage'] = (hourly_heavy['Heavy_Count'] / hourly_heavy['total_vehicles']) * 100
        
        # Create dual-axis plot
        ax4_twin = ax4.twinx()
        
        # Plot heavy vehicle count
        line1 = ax4.plot(hourly_heavy.index, hourly_heavy['Heavy_Count'], 
                        'o-', color='darkgreen', linewidth=3, markersize=8, label='Heavy Vehicle Count')
        ax4.set_ylabel('Heavy Vehicle Count', color='darkgreen', fontsize=12)
        ax4.tick_params(axis='y', labelcolor='darkgreen')
        
        # Plot heavy vehicle percentage
        line2 = ax4_twin.plot(hourly_heavy.index, hourly_heavy['heavy_percentage'], 
                             's-', color='purple', linewidth=3, markersize=8, label='Heavy Vehicle %')
        ax4_twin.set_ylabel('Heavy Vehicle Percentage (%)', color='purple', fontsize=12)
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        
                    
        # Highlight rush hours
        ax4.axvspan(morning_start, morning_end, alpha=0.1, color='orange')
        ax4.axvspan(evening_start, evening_end, alpha=0.1, color='red')
        
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_title('Heavy Vehicle Patterns Throughout the Day', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(24))
        ax4.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        # Find heavy vehicle peak hours
        heavy_peak_hour = hourly_heavy['Heavy_Count'].idxmax()
        heavy_pct_peak_hour = hourly_heavy['heavy_percentage'].idxmax()
        
        ax4.text(0.02, 0.98, f'Heavy vehicle peak: {heavy_peak_hour}:00\nHighest percentage: {heavy_pct_peak_hour}:00 ({hourly_heavy["heavy_percentage"][heavy_pct_peak_hour]:.1f}%)',
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Rush Hour Duration Analysis
    ax5 = fig.add_subplot(gs[3, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'day_of_week', 'total_vehicles']):
        # Calculate rush hour duration for each day
        rush_hour_data = []
        
        for day in range(7):
            day_data = final_dataset[final_dataset['day_of_week'] == day]
            hourly_avg = day_data.groupby('hour_of_day')['total_vehicles'].mean()
            
            # Define threshold as 80% of daily peak
            threshold = hourly_avg.max() * 0.8
            
            # Find morning rush duration
            morning_hours = hourly_avg[6:12]  # 6 AM to 12 PM
            morning_rush = morning_hours[morning_hours >= threshold]
            morning_duration = len(morning_rush) if len(morning_rush) > 0 else 0
            
            # Find evening rush duration
            evening_hours = hourly_avg[15:21]  # 3 PM to 9 PM
            evening_rush = evening_hours[evening_hours >= threshold]
            evening_duration = len(evening_rush) if len(evening_rush) > 0 else 0
            
            rush_hour_data.append({
                'day': day,
                'morning_duration': morning_duration,
                'evening_duration': evening_duration,
                'total_duration': morning_duration + evening_duration
            })
        
        rush_df = pd.DataFrame(rush_hour_data)
        
        # Create stacked bar chart
        x = np.arange(7)
        bars1 = ax5.bar(x, rush_df['morning_duration'], label='Morning Rush Duration', color='orange', alpha=0.8)
        bars2 = ax5.bar(x, rush_df['evening_duration'], bottom=rush_df['morning_duration'], 
                        label='Evening Rush Duration', color='red', alpha=0.8)
        
        ax5.set_xlabel('Day of Week', fontsize=12)
        ax5.set_ylabel('Rush Hour Duration (hours)', fontsize=12)
        ax5.set_title('Rush Hour Duration by Day of Week (≥80% of Daily Peak)', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(days)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add total duration labels
        for i, total in enumerate(rush_df['total_duration']):
            ax5.text(i, total + 0.1, f'{total}h', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# ========================================
# CORRELATION WITH AIR QUALITY PEAKS
# ========================================

def analyze_traffic_pollution_correlation(final_dataset):
    """
    Analyze correlation between traffic peaks and pollution peaks with lag analysis
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Hourly correlation pattern
    ax1 = fig.add_subplot(gs[0, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles', 'NO2']):
        # Calculate hourly averages
        hourly_data = final_dataset.groupby('hour_of_day').agg({
            'total_vehicles': 'mean',
            'NO2': 'mean'
        })
        
        # Normalize for visualization
        traffic_norm = (hourly_data['total_vehicles'] - hourly_data['total_vehicles'].min()) / \
                      (hourly_data['total_vehicles'].max() - hourly_data['total_vehicles'].min())
        no2_norm = (hourly_data['NO2'] - hourly_data['NO2'].min()) / \
                   (hourly_data['NO2'].max() - hourly_data['NO2'].min())
        
        # Plot normalized patterns
        ax1.plot(hourly_data.index, traffic_norm, 'o-', color='darkblue', 
                linewidth=3, markersize=8, label='Traffic (normalized)')
        ax1.plot(hourly_data.index, no2_norm, 's-', color='darkred', 
                linewidth=3, markersize=8, label='NO2 (normalized)')
        
        # Calculate correlation
        correlation = hourly_data['total_vehicles'].corr(hourly_data['NO2'])
        
        # Highlight rush hours
        rush_info = identify_rush_hours(final_dataset)
        if rush_info:
            morning_start, morning_end = rush_info['morning_window']
            evening_start, evening_end = rush_info['evening_window']
            ax1.axvspan(morning_start, morning_end, alpha=0.1, color='orange')
            ax1.axvspan(evening_start, evening_end, alpha=0.1, color='red')
        
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Normalized Values', fontsize=12)
        ax1.set_title(f'Hourly Traffic vs NO2 Patterns (Correlation: {correlation:.3f})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add lag indicators
        traffic_peak = hourly_data['total_vehicles'].idxmax()
        no2_peak = hourly_data['NO2'].idxmax()
        lag = no2_peak - traffic_peak
        
        ax1.annotate(f'Traffic Peak\n{traffic_peak}:00', 
                    xy=(traffic_peak, traffic_norm[traffic_peak]),
                    xytext=(traffic_peak, 1.1),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='blue'))
        
        ax1.annotate(f'NO2 Peak\n{no2_peak}:00\n(Lag: {lag}h)', 
                    xy=(no2_peak, no2_norm[no2_peak]),
                    xytext=(no2_peak, 1.2),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    # 2. Lag correlation analysis
    ax2 = fig.add_subplot(gs[1, 0])
    
    if all(col in final_dataset.columns for col in ['total_vehicles', 'NO2']):
        # Calculate cross-correlation for different lags
        max_lag = 12  # Check up to 12 hours lag
        lags = range(-max_lag, max_lag + 1)
        correlations = []
        
        for lag in lags:
            if lag < 0:
                # Traffic leads pollution
                corr = final_dataset['total_vehicles'].iloc[:lag].corr(
                    final_dataset['NO2'].iloc[-lag:])
            elif lag > 0:
                # Pollution leads traffic (unusual but check)
                corr = final_dataset['total_vehicles'].iloc[lag:].corr(
                    final_dataset['NO2'].iloc[:-lag])
            else:
                # No lag
                corr = final_dataset['total_vehicles'].corr(final_dataset['NO2'])
            
            correlations.append(corr)
        
        # Plot cross-correlation
        bars = ax2.bar(lags, correlations, color=['red' if c < 0 else 'green' for c in correlations], 
                       alpha=0.7, edgecolor='black')
        
        # Highlight maximum correlation
        max_corr_idx = np.argmax(np.abs(correlations))
        max_lag_value = lags[max_corr_idx]
        max_corr_value = correlations[max_corr_idx]
        
        ax2.axvline(max_lag_value, color='gold', linestyle='--', linewidth=2)
        ax2.text(max_lag_value, max_corr_value + 0.02, 
                f'Max: {max_corr_value:.3f}\nat lag {max_lag_value}h',
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax2.set_xlabel('Lag (hours) - Negative = Traffic leads NO2', fontsize=12)
        ax2.set_ylabel('Correlation Coefficient', fontsize=12)
        ax2.set_title('Cross-correlation: Traffic vs NO2', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-max_lag-0.5, max_lag+0.5)
    
    # 3. PM10 lag analysis (if available)
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'PM10' in final_dataset.columns:
        # Similar analysis for PM10
        correlations_pm10 = []
        
        for lag in lags:
            if lag < 0:
                corr = final_dataset['total_vehicles'].iloc[:lag].corr(
                    final_dataset['PM10'].iloc[-lag:])
            elif lag > 0:
                corr = final_dataset['total_vehicles'].iloc[lag:].corr(
                    final_dataset['PM10'].iloc[:-lag])
            else:
                corr = final_dataset['total_vehicles'].corr(final_dataset['PM10'])
            
            correlations_pm10.append(corr)
        
        bars = ax3.bar(lags, correlations_pm10, 
                       color=['red' if c < 0 else 'green' for c in correlations_pm10], 
                       alpha=0.7, edgecolor='black')
        
        # Highlight maximum correlation
        max_corr_idx = np.argmax(np.abs(correlations_pm10))
        max_lag_value = lags[max_corr_idx]
        max_corr_value = correlations_pm10[max_corr_idx]
        
        ax3.axvline(max_lag_value, color='gold', linestyle='--', linewidth=2)
        ax3.text(max_lag_value, max_corr_value + 0.02, 
                f'Max: {max_corr_value:.3f}\nat lag {max_lag_value}h',
                ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax3.set_xlabel('Lag (hours) - Negative = Traffic leads PM10', fontsize=12)
        ax3.set_ylabel('Correlation Coefficient', fontsize=12)
        ax3.set_title('Cross-correlation: Traffic vs PM10', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-max_lag-0.5, max_lag+0.5)
    
    # 4. Rush hour pollution response
    ax4 = fig.add_subplot(gs[2, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles', 'NO2']):
        # Create lagged NO2 columns
        for lag in [0, 1, 2, 3, 4]:
            final_dataset[f'NO2_lag_{lag}h'] = final_dataset['NO2'].shift(lag)
        
        # Get rush hour info
        rush_info = identify_rush_hours(final_dataset)
        morning_start, morning_end = rush_info['morning_window']
        
        # Calculate average response during and after morning rush hour
        response_data = []
        
        for hour in range(morning_start-1, morning_start+6):  # From 1 hour before to 5 hours after start
            hour_data = final_dataset[final_dataset['hour_of_day'] == hour]
            
            traffic_avg = hour_data['total_vehicles'].mean()
            no2_avg = hour_data['NO2'].mean()
            
            response_data.append({
                'hour': hour,
                'traffic': traffic_avg,
                'NO2': no2_avg,
                'time_from_rush_start': hour - morning_start
            })
        
        response_df = pd.DataFrame(response_data)
        
        # Normalize for comparison
        response_df['traffic_norm'] = (response_df['traffic'] - response_df['traffic'].min()) / \
                                      (response_df['traffic'].max() - response_df['traffic'].min())
        response_df['NO2_norm'] = (response_df['NO2'] - response_df['NO2'].min()) / \
                                  (response_df['NO2'].max() - response_df['NO2'].min())
        
        # Plot response curves
        ax4.plot(response_df['time_from_rush_start'], response_df['traffic_norm'], 
                'o-', color='darkblue', linewidth=3, markersize=10, label='Traffic')
        ax4.plot(response_df['time_from_rush_start'], response_df['NO2_norm'], 
                's-', color='darkred', linewidth=3, markersize=10, label='NO2')
        
        # Shade rush hour period
        ax4.axvspan(0, morning_end - morning_start, alpha=0.2, color='orange', label='Rush Hour')
        
        ax4.set_xlabel('Hours from Rush Hour Start', fontsize=12)
        ax4.set_ylabel('Normalized Values', fontsize=12)
        ax4.set_title('Pollution Response to Morning Rush Hour', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xticks(range(-1, 6))
        
        # Add annotations for key points
        traffic_peak_idx = response_df['traffic_norm'].idxmax()
        no2_peak_idx = response_df['NO2_norm'].idxmax()
        
        ax4.annotate('Traffic Peak', 
                    xy=(response_df.loc[traffic_peak_idx, 'time_from_rush_start'], 
                        response_df.loc[traffic_peak_idx, 'traffic_norm']),
                    xytext=(response_df.loc[traffic_peak_idx, 'time_from_rush_start'] + 0.5, 0.8),
                    arrowprops=dict(arrowstyle='->', color='blue'))
        
        ax4.annotate('NO2 Peak', 
                    xy=(response_df.loc[no2_peak_idx, 'time_from_rush_start'], 
                        response_df.loc[no2_peak_idx, 'NO2_norm']),
                    xytext=(response_df.loc[no2_peak_idx, 'time_from_rush_start'] + 0.5, 0.5),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    plt.show()

# ========================================
# POLLUTANT-SPECIFIC RESPONSE ANALYSIS
# ========================================

def analyze_pollutant_response_times(final_dataset):
    """
    Analyze response time variations by pollutant type
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Prepare pollutants list
    pollutants = []
    if 'NO2' in final_dataset.columns:
        pollutants.append('NO2')
    if 'PM10' in final_dataset.columns:
        pollutants.append('PM10')
    if 'PM25' in final_dataset.columns:
        pollutants.append('PM25')
    
    # 1. Response time comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'total_vehicles' in final_dataset.columns and len(pollutants) > 0:
        response_times = {}
        colors = {'NO2': 'red', 'PM10': 'blue', 'PM25': 'green'}
        
        for pollutant in pollutants:
            # Calculate cross-correlation
            max_lag = 8
            lags = range(0, max_lag + 1)
            correlations = []
            
            for lag in lags:
                if lag == 0:
                    corr = final_dataset['total_vehicles'].corr(final_dataset[pollutant])
                else:
                    corr = final_dataset['total_vehicles'].iloc[:-lag].corr(
                        final_dataset[pollutant].iloc[lag:])
                correlations.append(corr)
            
            # Find optimal lag
            optimal_lag = np.argmax(correlations)
            response_times[pollutant] = {
                'lag': optimal_lag,
                'correlation': correlations[optimal_lag],
                'correlations': correlations
            }
            
            # Plot correlation curves
            ax1.plot(lags, correlations, 'o-', color=colors.get(pollutant, 'gray'), 
                    linewidth=2, markersize=8, label=f'{pollutant} (Peak lag: {optimal_lag}h)')
        
        ax1.set_xlabel('Lag (hours)', fontsize=12)
        ax1.set_ylabel('Correlation Coefficient', fontsize=12)
        ax1.set_title('Pollutant Response Time to Traffic Changes', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add vertical lines for optimal lags
        for pollutant, data in response_times.items():
            ax1.axvline(data['lag'], color=colors.get(pollutant, 'gray'), 
                       linestyle='--', alpha=0.5)
    
    # 2. Rush hour impact by pollutant
    ax2 = fig.add_subplot(gs[1, 0])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'is_rush_hour']):
        # Define rush hours if not already in dataset
        if 'is_rush_hour' not in final_dataset.columns:
            rush_info = identify_rush_hours(final_dataset)
            morning_start, morning_end = rush_info['morning_window']
            evening_start, evening_end = rush_info['evening_window']
            
            final_dataset['is_rush_hour'] = final_dataset['hour_of_day'].apply(
                lambda x: (morning_start <= x <= morning_end) or (evening_start <= x <= evening_end)
            )
        
        # Calculate impact for each pollutant
        impact_data = []
        
        for pollutant in pollutants:
            rush_hour_mean = final_dataset[final_dataset['is_rush_hour']][pollutant].mean()
            non_rush_mean = final_dataset[~final_dataset['is_rush_hour']][pollutant].mean()
            impact = ((rush_hour_mean - non_rush_mean) / non_rush_mean) * 100
            
            impact_data.append({
                'pollutant': pollutant,
                'rush_hour': rush_hour_mean,
                'non_rush': non_rush_mean,
                'impact_pct': impact
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # Create bar plot
        x = np.arange(len(pollutants))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, impact_df['non_rush'], width, 
                        label='Non-Rush Hour', color='lightblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, impact_df['rush_hour'], width, 
                        label='Rush Hour', color='darkred', alpha=0.8)
        
        # Add percentage labels
        for i, row in impact_df.iterrows():
            ax2.text(i, row['rush_hour'] + 1, f'+{row["impact_pct"]:.1f}%', 
                    ha='center', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Pollutant', fontsize=12)
        ax2.set_ylabel('Average Concentration (μg/m³)', fontsize=12)
        ax2.set_title('Rush Hour Impact on Different Pollutants', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(pollutants)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Weekday vs Weekend response
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'is_weekend' in final_dataset.columns and 'hour_of_day' in final_dataset.columns:
        # Compare rush hour effects on weekdays vs weekends
        for pollutant in pollutants[:1]:  # Show first pollutant for clarity
            weekday_hourly = final_dataset[~final_dataset['is_weekend']].groupby('hour_of_day')[pollutant].mean()
            weekend_hourly = final_dataset[final_dataset['is_weekend']].groupby('hour_of_day')[pollutant].mean()
            
            ax3.plot(weekday_hourly.index, weekday_hourly.values, 'o-', 
                    color='darkblue', linewidth=2, label='Weekday')
            ax3.plot(weekend_hourly.index, weekend_hourly.values, 's-', 
                    color='darkgreen', linewidth=2, label='Weekend')
        
        # Highlight rush hours
        rush_info = identify_rush_hours(final_dataset)
        if rush_info:
            morning_start, morning_end = rush_info['morning_window']
            evening_start, evening_end = rush_info['evening_window']
            ax3.axvspan(morning_start, morning_end, alpha=0.1, color='orange')
            ax3.axvspan(evening_start, evening_end, alpha=0.1, color='red')
        
        ax3.set_xlabel('Hour of Day', fontsize=12)
        ax3.set_ylabel(f'{pollutants[0]} Concentration (μg/m³)', fontsize=12)
        ax3.set_title(f'Weekday vs Weekend {pollutants[0]} Patterns', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 4. Response intensity heatmap
    ax4 = fig.add_subplot(gs[2, :])
    
    if len(pollutants) > 0 and 'hour_of_day' in final_dataset.columns:
        # Create matrix of hourly correlations
        hours = range(24)
        correlation_matrix = pd.DataFrame(index=hours, columns=pollutants)
        
        for hour in hours:
            hour_data = final_dataset[final_dataset['hour_of_day'] == hour]
            if len(hour_data) > 30:  # Ensure sufficient data
                for pollutant in pollutants:
                    correlation_matrix.loc[hour, pollutant] = hour_data['total_vehicles'].corr(hour_data[pollutant])
        
        # Convert to numeric and plot heatmap
        correlation_matrix = correlation_matrix.astype(float)
        
        sns.heatmap(correlation_matrix.T, cmap='RdBu_r', center=0, 
                   annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'},
                   vmin=-1, vmax=1, ax=ax4)
        
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_ylabel('Pollutant', fontsize=12)
        ax4.set_title('Hourly Traffic-Pollution Correlation Patterns', fontsize=14, fontweight='bold')
        
        # Add rush hour indicators
        rush_info = identify_rush_hours(final_dataset)
        if rush_info:
            morning_start, morning_end = rush_info['morning_window']
            evening_start, evening_end = rush_info['evening_window']
            
            for hour in range(morning_start, morning_end + 1):
                ax4.add_patch(Rectangle((hour, -0.5), 1, len(pollutants), 
                                      fill=False, edgecolor='orange', lw=2))
            
            for hour in range(evening_start, evening_end + 1):
                ax4.add_patch(Rectangle((hour, -0.5), 1, len(pollutants), 
                                      fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    plt.show()

# ========================================
# SUMMARY STATISTICS AND INSIGHTS
# ========================================

def generate_rush_hour_summary(final_dataset):
    """
    Generate comprehensive summary of rush hour patterns and impacts
    """
    print("=" * 80)
    print("RUSH HOUR ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Identify rush hours
    rush_info = identify_rush_hours(final_dataset)
    
    if rush_info:
        print(f"\n📍 RUSH HOUR IDENTIFICATION:")
        print(f"   Morning Rush: {rush_info['morning_window'][0]}:00 - {rush_info['morning_window'][1]}:00")
        print(f"   Morning Peak: {rush_info['morning_peak']}:00")
        print(f"   Evening Rush: {rush_info['evening_window'][0]}:00 - {rush_info['evening_window'][1]}:00")
        print(f"   Evening Peak: {rush_info['evening_peak']}:00")
    
    # Traffic statistics
    if all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles', 'is_rush_hour']):
        if 'is_rush_hour' not in final_dataset.columns:
            morning_start, morning_end = rush_info['morning_window']
            evening_start, evening_end = rush_info['evening_window']
            
            final_dataset['is_rush_hour'] = final_dataset['hour_of_day'].apply(
                lambda x: (morning_start <= x <= morning_end) or (evening_start <= x <= evening_end)
            )
        
        rush_traffic = final_dataset[final_dataset['is_rush_hour']]['total_vehicles'].mean()
        non_rush_traffic = final_dataset[~final_dataset['is_rush_hour']]['total_vehicles'].mean()
        
        print(f"\n🚗 TRAFFIC STATISTICS:")
        print(f"   Rush Hour Average: {rush_traffic:.0f} vehicles/hour")
        print(f"   Non-Rush Average: {non_rush_traffic:.0f} vehicles/hour")
        print(f"   Rush Hour Increase: {((rush_traffic - non_rush_traffic) / non_rush_traffic * 100):.1f}%")
    
    # Pollution impact
    pollutants = ['NO2', 'PM10', 'PM25']
    print(f"\n🌫️ POLLUTION IMPACT:")
    
    for pollutant in pollutants:
        if pollutant in final_dataset.columns:
            rush_pollution = final_dataset[final_dataset['is_rush_hour']][pollutant].mean()
            non_rush_pollution = final_dataset[~final_dataset['is_rush_hour']][pollutant].mean()
            impact = ((rush_pollution - non_rush_pollution) / non_rush_pollution) * 100
            
            print(f"   {pollutant}:")
            print(f"      Rush Hour: {rush_pollution:.1f} μg/m³")
            print(f"      Non-Rush: {non_rush_pollution:.1f} μg/m³")
            print(f"      Increase: {impact:.1f}%")
    
    print("\n" + "=" * 80)
