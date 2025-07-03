import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calplot  # For calendar heatmaps
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========================================
# HOURLY PATTERNS (24-HOUR CYCLES)
# ========================================

def visualize_hourly_patterns(final_dataset, df_no2):
    """
    Visualize 24-hour patterns for traffic and air quality
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Traffic hourly pattern with confidence intervals
    ax1 = fig.add_subplot(gs[0, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'total_vehicles']):
        hourly_stats = final_dataset.groupby('hour_of_day')['total_vehicles'].agg(['mean', 'std', 'count'])
        
        # Calculate confidence intervals
        confidence = 0.95
        z_score = 1.96  # for 95% confidence
        hourly_stats['ci'] = z_score * (hourly_stats['std'] / np.sqrt(hourly_stats['count']))
        
        hours = hourly_stats.index
        means = hourly_stats['mean']
        ci = hourly_stats['ci']
        
        # Main line plot
        ax1.plot(hours, means, 'o-', color='darkblue', linewidth=3, markersize=8, label='Mean Traffic')
        ax1.fill_between(hours, means - ci, means + ci, alpha=0.3, color='skyblue', label='95% CI')
        
        # Add rush hour shading
        ax1.axvspan(6, 9, alpha=0.1, color='orange', label='Morning Rush')
        ax1.axvspan(17, 20, alpha=0.1, color='red', label='Evening Rush')
        
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('24-Hour Traffic Pattern with Confidence Intervals', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(24))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Add annotations for peak hours
        peak_hour = means.idxmax()
        ax1.annotate(f'Peak: {means[peak_hour]:.0f}', 
                    xy=(peak_hour, means[peak_hour]), 
                    xytext=(peak_hour, means[peak_hour] + means.std()*0.5),
                    ha='center',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. NO2 hourly pattern
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 'data' in df_no2.columns:
        # Convert to datetime and extract hour
        df_no2['datetime'] = pd.to_datetime(df_no2['data'])
        df_no2['hour'] = df_no2['datetime'].dt.hour
        
        # Get NO2 station columns
        no2_stations = [col for col in df_no2.columns if not col.startswith('v_') and col not in ['data', 'datetime', 'hour']]
        
        for station in no2_stations[:5]:  # Limit to 5 stations for clarity
            if station in df_no2.columns:
                hourly_no2 = df_no2.groupby('hour')[station].mean()
                ax2.plot(hourly_no2.index, hourly_no2.values, 'o-', linewidth=2, 
                        markersize=6, label=station, alpha=0.8)
        
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('NO2 Concentration (μg/m³)', fontsize=12)
        ax2.set_title('24-Hour NO2 Pattern by Station', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, 24, 3))
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 3. Traffic vs NO2 correlation by hour
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'hour_of_day' in final_dataset.columns and 'NO2' in final_dataset.columns:
        # Calculate hourly averages
        hourly_traffic = final_dataset.groupby('hour_of_day')['total_vehicles'].mean()
        hourly_no2 = final_dataset.groupby('hour_of_day')['NO2'].mean()
        
        # Normalize for dual axis
        ax3_twin = ax3.twinx()
        
        # Plot traffic
        line1 = ax3.plot(hourly_traffic.index, hourly_traffic.values, 'o-', 
                         color='darkblue', linewidth=3, markersize=8, label='Traffic')
        ax3.set_ylabel('Average Vehicle Count', color='darkblue', fontsize=12)
        ax3.tick_params(axis='y', labelcolor='darkblue')
        
        # Plot NO2
        line2 = ax3_twin.plot(hourly_no2.index, hourly_no2.values, 's-', 
                             color='darkred', linewidth=3, markersize=8, label='NO2')
        ax3_twin.set_ylabel('NO2 Concentration (μg/m³)', color='darkred', fontsize=12)
        ax3_twin.tick_params(axis='y', labelcolor='darkred')
        
        ax3.set_xlabel('Hour of Day', fontsize=12)
        ax3.set_title('Hourly Traffic vs NO2 Patterns', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(0, 24, 3))
        ax3.grid(True, alpha=0.3)
        
        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper right')
    
    # 4. Hourly heatmap by vehicle type
    ax4 = fig.add_subplot(gs[2, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'Light_Count', 'Medium_Count', 'Heavy_Count']):
        vehicle_types = ['Light_Count', 'Medium_Count', 'Heavy_Count']
        hourly_by_type = final_dataset.groupby('hour_of_day')[vehicle_types].mean()
        
        # Normalize each vehicle type to percentage
        hourly_pct = hourly_by_type.div(hourly_by_type.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        sns.heatmap(hourly_pct.T, cmap='YlOrRd', annot=True, fmt='.1f', 
                   cbar_kws={'label': 'Percentage (%)'}, ax=ax4)
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_ylabel('Vehicle Type', fontsize=12)
        ax4.set_title('Vehicle Type Distribution by Hour (%)', fontsize=14, fontweight='bold')
        ax4.set_yticklabels(['Light', 'Medium', 'Heavy'], rotation=0)
    
    plt.tight_layout()
    plt.show()

# ========================================
# WEEKLY PATTERNS (WEEKDAY VS WEEKEND)
# ========================================

def visualize_weekly_patterns(final_dataset, df_no2, df_pm10):
    """
    Visualize weekly patterns including weekday vs weekend comparisons
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Daily traffic pattern
    ax1 = fig.add_subplot(gs[0, :])
    
    if all(col in final_dataset.columns for col in ['day_of_week', 'total_vehicles']):
        daily_stats = final_dataset.groupby('day_of_week')['total_vehicles'].agg(['mean', 'std', 'count'])
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Create bar plot with error bars
        x_pos = np.arange(7)
        bars = ax1.bar(x_pos, daily_stats['mean'], yerr=daily_stats['std'], 
                       capsize=5, edgecolor='black', linewidth=1)
        
        # Color weekends differently
        for i in range(5):
            bars[i].set_color('steelblue')
            bars[i].set_alpha(0.8)
        for i in [5, 6]:
            bars[i].set_color('lightcoral')
            bars[i].set_alpha(0.8)
        
        ax1.set_xlabel('Day of Week', fontsize=12)
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('Weekly Traffic Pattern', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(days)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage change labels
        for i, (idx, row) in enumerate(daily_stats.iterrows()):
            if i > 0:
                pct_change = ((row['mean'] - daily_stats.iloc[i-1]['mean']) / daily_stats.iloc[i-1]['mean']) * 100
                ax1.text(i, row['mean'] + row['std'], f'{pct_change:+.1f}%', 
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Weekday vs Weekend hourly patterns
    ax2 = fig.add_subplot(gs[1, :2])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'is_weekend', 'total_vehicles']):
        weekday_hourly = final_dataset[~final_dataset['is_weekend']].groupby('hour_of_day')['total_vehicles'].mean()
        weekend_hourly = final_dataset[final_dataset['is_weekend']].groupby('hour_of_day')['total_vehicles'].mean()
        
        ax2.plot(weekday_hourly.index, weekday_hourly.values, 'o-', 
                color='darkblue', linewidth=3, markersize=8, label='Weekday')
        ax2.plot(weekend_hourly.index, weekend_hourly.values, 's-', 
                color='darkred', linewidth=3, markersize=8, label='Weekend')
        
        ax2.fill_between(weekday_hourly.index, 0, weekday_hourly.values, alpha=0.2, color='darkblue')
        ax2.fill_between(weekend_hourly.index, 0, weekend_hourly.values, alpha=0.2, color='darkred')
        
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Average Vehicle Count', fontsize=12)
        ax2.set_title('Weekday vs Weekend Hourly Traffic Patterns', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, 24, 3))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight maximum difference
        diff = weekday_hourly - weekend_hourly
        max_diff_hour = diff.idxmax()
        ax2.annotate(f'Max difference\nat {max_diff_hour}:00', 
                    xy=(max_diff_hour, weekday_hourly[max_diff_hour]), 
                    xytext=(max_diff_hour + 3, weekday_hourly[max_diff_hour]),
                    arrowprops=dict(arrowstyle='->', color='green'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 3. Day of week effect on air quality
    ax3 = fig.add_subplot(gs[1, 2])
    
    if 'day_of_week' in final_dataset.columns and 'NO2' in final_dataset.columns:
        daily_no2 = final_dataset.groupby('day_of_week')['NO2'].agg(['mean', 'std'])
        
        x_pos = np.arange(7)
        bars = ax3.bar(x_pos, daily_no2['mean'], yerr=daily_no2['std'], 
                       capsize=5, edgecolor='black', linewidth=1)
        
        # Color based on pollution level
        for i, bar in enumerate(bars):
            if daily_no2['mean'].iloc[i] > 40:  # EU annual limit
                bar.set_color('red')
                bar.set_alpha(0.8)
            else:
                bar.set_color('green')
                bar.set_alpha(0.8)
        
        ax3.axhline(40, color='red', linestyle='--', label='EU Annual Limit')
        ax3.set_xlabel('Day of Week', fontsize=12)
        ax3.set_ylabel('NO2 Concentration (μg/m³)', fontsize=12)
        ax3.set_title('Daily NO2 Levels', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
    
    # 4. Weekday vs Weekend pollution distribution
    ax4 = fig.add_subplot(gs[2, 0])
        
    if 'is_weekend' in final_dataset.columns and 'NO2' in final_dataset.columns:
        weekday_no2 = final_dataset[~final_dataset['is_weekend']]['NO2'].dropna()
        weekend_no2 = final_dataset[final_dataset['is_weekend']]['NO2'].dropna()
        
        # Create violin plots
        parts = ax4.violinplot([weekday_no2, weekend_no2], positions=[0, 1], 
                              showmeans=True, showmedians=True, showextrema=True)
        
        # Customize colors
        colors = ['steelblue', 'lightcoral']
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Weekday', 'Weekend'])
        ax4.set_ylabel('NO2 Concentration (μg/m³)', fontsize=12)
        ax4.set_title('NO2 Distribution: Weekday vs Weekend', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        weekday_mean = weekday_no2.mean()
        weekend_mean = weekend_no2.mean()
        reduction = ((weekday_mean - weekend_mean) / weekday_mean) * 100
        ax4.text(0.5, ax4.get_ylim()[1]*0.95, f'Weekend reduction: {reduction:.1f}%', 
                ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # 5. Weekly pattern heatmap
    ax5 = fig.add_subplot(gs[2, 1:])
    
    if all(col in final_dataset.columns for col in ['day_of_week', 'hour_of_day', 'total_vehicles']):
        # Create pivot table for heatmap
        weekly_heatmap = final_dataset.pivot_table(
            values='total_vehicles', 
            index='hour_of_day', 
            columns='day_of_week', 
            aggfunc='mean'
        )
        
        # Create custom colormap
        sns.heatmap(weekly_heatmap, cmap='RdYlBu_r', annot=False, 
                   cbar_kws={'label': 'Average Vehicle Count'}, ax=ax5,
                   linewidths=0.5, linecolor='gray')
        
        ax5.set_xlabel('Day of Week', fontsize=12)
        ax5.set_ylabel('Hour of Day', fontsize=12)
        ax5.set_title('Weekly Traffic Intensity Heatmap', fontsize=14, fontweight='bold')
        ax5.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add rush hour indicators
        for day in range(7):
            ax5.add_patch(Rectangle((day, 7), 1, 2, fill=False, edgecolor='red', lw=2))
            ax5.add_patch(Rectangle((day, 17), 1, 3, fill=False, edgecolor='orange', lw=2))
    
    plt.tight_layout()
    plt.show()

# ========================================
# MONTHLY AND SEASONAL TRENDS
# ========================================

def visualize_monthly_seasonal_trends(final_dataset, df_no2, df_pm10):
    """
    Visualize monthly and seasonal patterns in traffic and air quality
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Monthly traffic trends
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'month' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        monthly_stats = final_dataset.groupby('month')['total_vehicles'].agg(['mean', 'std', 'count'])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Create line plot with confidence intervals
        x = monthly_stats.index
        y = monthly_stats['mean']
        yerr = monthly_stats['std'] / np.sqrt(monthly_stats['count'])  # Standard error
        
        ax1.plot(x, y, 'o-', color='darkblue', linewidth=3, markersize=10)
        ax1.fill_between(x, y - yerr, y + yerr, alpha=0.3, color='skyblue')
        
        # Add seasonal backgrounds
        # Winter
        ax1.axvspan(0.5, 2.5, alpha=0.1, color='blue', label='Winter')
        ax1.axvspan(11.5, 12.5, alpha=0.1, color='blue')
        # Spring
        ax1.axvspan(2.5, 5.5, alpha=0.1, color='green', label='Spring')
        # Summer
        ax1.axvspan(5.5, 8.5, alpha=0.1, color='orange', label='Summer')
        # Fall
        ax1.axvspan(8.5, 11.5, alpha=0.1, color='brown', label='Fall')
        
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Average Vehicle Count', fontsize=12)
        ax1.set_title('Monthly Traffic Trends with Seasonal Indicators', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(months)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax1.plot(x, p(x), "--", color='red', alpha=0.8, linewidth=2, label='Trend')
    
    # 2. Seasonal comparison box plots
    ax2 = fig.add_subplot(gs[1, 0])
    
    if 'season' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        season_data = [final_dataset[final_dataset['season'] == season]['total_vehicles'].dropna() 
                      for season in season_order]
        
        bp = ax2.boxplot(season_data, labels=season_order, patch_artist=True)
        
        # Customize box colors
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Vehicle Count', fontsize=12)
        ax2.set_title('Seasonal Traffic Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add mean markers
        for i, data in enumerate(season_data):
            mean_val = data.mean()
            ax2.scatter(i+1, mean_val, color='red', s=100, zorder=3, marker='D', label='Mean' if i == 0 else '')
        ax2.legend()
    
    # 3. Monthly air quality trends
    ax3 = fig.add_subplot(gs[1, 1])
    
    if 'month' in final_dataset.columns and 'NO2' in final_dataset.columns:
        monthly_no2 = final_dataset.groupby('month')['NO2'].agg(['mean', 'std'])
        
        # Create bar plot with error bars
        x = monthly_no2.index
        bars = ax3.bar(x, monthly_no2['mean'], yerr=monthly_no2['std'], 
                       capsize=5, edgecolor='black', linewidth=1)
        
        # Color bars based on pollution level
        for i, bar in enumerate(bars):
            if monthly_no2['mean'].iloc[i] > 40:  # EU annual limit
                bar.set_color('red')
                bar.set_alpha(0.8)
            elif monthly_no2['mean'].iloc[i] > 25:  # WHO guideline
                bar.set_color('orange')
                bar.set_alpha(0.8)
            else:
                bar.set_color('green')
                bar.set_alpha(0.8)
        
        ax3.axhline(40, color='red', linestyle='--', label='EU Annual Limit', linewidth=2)
        ax3.axhline(25, color='orange', linestyle='--', label='WHO Guideline', linewidth=2)
        
        ax3.set_xlabel('Month', fontsize=12)
        ax3.set_ylabel('NO2 Concentration (μg/m³)', fontsize=12)
        ax3.set_title('Monthly NO2 Levels', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(months)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend()
    
    # 4. Seasonal pollution patterns
    ax4 = fig.add_subplot(gs[2, :])
    
    if all(col in final_dataset.columns for col in ['season', 'hour_of_day', 'NO2']):
        # Create subplots for each season
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        colors = ['blue', 'green', 'orange', 'brown']
        
        for i, (season, color) in enumerate(zip(season_order, colors)):
            season_data = final_dataset[final_dataset['season'] == season]
            hourly_no2 = season_data.groupby('hour_of_day')['NO2'].mean()
            
            ax4.plot(hourly_no2.index, hourly_no2.values, 'o-', 
                    color=color, linewidth=2, markersize=6, label=season, alpha=0.8)
        
        ax4.set_xlabel('Hour of Day', fontsize=12)
        ax4.set_ylabel('NO2 Concentration (μg/m³)', fontsize=12)
        ax4.set_title('Seasonal Hourly NO2 Patterns', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(0, 24, 3))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add shaded regions for typical high pollution hours
        ax4.axvspan(7, 9, alpha=0.05, color='red')
        ax4.axvspan(17, 19, alpha=0.05, color='red')
    
    plt.tight_layout()
    plt.show()

    """
    Create custom calendar heatmap with improved layout
    """
    import matplotlib
    import calendar
    from matplotlib.patches import Rectangle
    
    # Set matplotlib to use default font
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    if 'datetime' not in final_dataset.columns:
        print("Datetime column not found in final_dataset")
        return
    
    # Prepare daily data
    daily_traffic = final_dataset.groupby(final_dataset['datetime'].dt.date)['total_vehicles'].sum()
    daily_traffic.index = pd.to_datetime(daily_traffic.index)
    
    # Get date range
    start_date = daily_traffic.index.min()
    end_date = daily_traffic.index.max()
    years = sorted(daily_traffic.index.year.unique())
    
    # Calculate number of years
    n_years = len(years)
    
    # Create figure with subplots for each year
    fig = plt.figure(figsize=(20, 4 * n_years + 2))
    
    # Traffic Calendar Heatmap
    for year_idx, year in enumerate(years):
        ax = plt.subplot(n_years, 1, year_idx + 1)
        
        # Filter data for this year
        year_data = daily_traffic[daily_traffic.index.year == year]
        
        # Create calendar layout
        month_positions = []
        days_data = []
        
        for month in range(1, 13):
            # Get calendar for this month
            month_cal = calendar.monthcalendar(year, month)
            month_positions.append(len(days_data))
            
            # Process each week
            for week_num, week in enumerate(month_cal):
                for day_num, day in enumerate(week):
                    if day == 0:
                        days_data.append(np.nan)
                    else:
                        try:
                            date = pd.Timestamp(year, month, day)
                            if date in year_data.index:
                                days_data.append(year_data[date])
                            else:
                                days_data.append(np.nan)
                        except:
                            days_data.append(np.nan)
        
        # Reshape data for heatmap (7 days x n_weeks)
        n_weeks = len(days_data) // 7
        heatmap_data = np.array(days_data[:n_weeks*7]).reshape(n_weeks, 7).T
        
        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set labels
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, (month_name, pos) in enumerate(zip(month_names, month_positions)):
            ax.text(pos, -0.7, month_name, ha='left', va='top', fontsize=10)
        
        # Add vertical lines between months
        for pos in month_positions[1:]:
            ax.axvline(x=pos-0.5, color='white', linewidth=2)
        
        ax.set_title(f'{year} - Daily Traffic Volume', fontsize=12, pad=10)
        ax.set_xlim(-0.5, n_weeks-0.5)
        
        # Remove x-axis labels
        ax.set_xticks([])
        
        # Add colorbar for the last subplot
        if year_idx == n_years - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
            cbar.set_label('Total Daily Traffic Volume', fontsize=10)
    
    plt.suptitle('Daily Traffic Volume Calendar Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # NO2 Calendar Heatmap
    if 'NO2' in final_dataset.columns:
        daily_no2 = final_dataset.groupby(final_dataset['datetime'].dt.date)['NO2'].mean()
        daily_no2.index = pd.to_datetime(daily_no2.index)
        
        fig = plt.figure(figsize=(20, 4 * n_years + 2))
        
        for year_idx, year in enumerate(years):
            ax = plt.subplot(n_years, 1, year_idx + 1)
            
            # Filter data for this year
            year_data = daily_no2[daily_no2.index.year == year]
            
            # Create calendar layout
            month_positions = []
            days_data = []
            
            for month in range(1, 13):
                # Get calendar for this month
                month_cal = calendar.monthcalendar(year, month)
                month_positions.append(len(days_data))
                
                # Process each week
                for week_num, week in enumerate(month_cal):
                    for day_num, day in enumerate(week):
                        if day == 0:
                            days_data.append(np.nan)
                        else:
                            try:
                                date = pd.Timestamp(year, month, day)
                                if date in year_data.index:
                                    days_data.append(year_data[date])
                                else:
                                    days_data.append(np.nan)
                            except:
                                days_data.append(np.nan)
            
            # Reshape data for heatmap (7 days x n_weeks)
            n_weeks = len(days_data) // 7
            heatmap_data = np.array(days_data[:n_weeks*7]).reshape(n_weeks, 7).T
            
            # Create heatmap with diverging colormap centered at EU limit
            vmin = np.nanmin(heatmap_data)
            vmax = np.nanmax(heatmap_data)
            vcenter = 40  # EU annual limit for NO2
            
            # Create normalized colormap
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                          norm=norm, interpolation='nearest')
            
            # Set labels
            ax.set_yticks(range(7))
            ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            # Add month labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, (month_name, pos) in enumerate(zip(month_names, month_positions)):
                ax.text(pos, -0.7, month_name, ha='left', va='top', fontsize=10)
            
            # Add vertical lines between months
            for pos in month_positions[1:]:
                ax.axvline(x=pos-0.5, color='white', linewidth=2)
            
            ax.set_title(f'{year} - Daily NO2 Concentration', fontsize=12, pad=10)
            ax.set_xlim(-0.5, n_weeks-0.5)
            
            # Remove x-axis labels
            ax.set_xticks([])
            
            # Add colorbar for the last subplot
            if year_idx == n_years - 1:
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
                cbar.set_label('Average Daily NO2 (μg/m³)', fontsize=10)
                # Add EU limit line on colorbar
                cbar.ax.axvline(x=40, color='red', linewidth=2, linestyle='--')
                cbar.ax.text(40, 1.5, 'EU Limit', ha='center', va='bottom', fontsize=8, color='red')
        
        plt.suptitle('Daily NO2 Concentration Calendar Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


    """
    Create custom calendar heatmap with improved layout
    """
    import matplotlib
    import calendar
    from matplotlib.patches import Rectangle
    
    # Set matplotlib to use default font
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    if 'datetime' not in final_dataset.columns:
        print("Datetime column not found in final_dataset")
        return
    
    # Prepare daily data
    daily_traffic = final_dataset.groupby(final_dataset['datetime'].dt.date)['total_vehicles'].sum()
    daily_traffic.index = pd.to_datetime(daily_traffic.index)
    
    # Get date range
    start_date = daily_traffic.index.min()
    end_date = daily_traffic.index.max()
    years = sorted(daily_traffic.index.year.unique())
    
    # Calculate number of years
    n_years = len(years)
    
    # Create figure with subplots for each year
    fig = plt.figure(figsize=(20, 4 * n_years + 2))
    
    # Traffic Calendar Heatmap
    for year_idx, year in enumerate(years):
        ax = plt.subplot(n_years, 1, year_idx + 1)
        
        # Filter data for this year
        year_data = daily_traffic[daily_traffic.index.year == year]
        
        # Create calendar layout
        month_positions = []
        days_data = []
        
        for month in range(1, 13):
            # Get calendar for this month
            month_cal = calendar.monthcalendar(year, month)
            month_positions.append(len(days_data))
            
            # Process each week
            for week_num, week in enumerate(month_cal):
                for day_num, day in enumerate(week):
                    if day == 0:
                        days_data.append(np.nan)
                    else:
                        try:
                            date = pd.Timestamp(year, month, day)
                            if date in year_data.index:
                                days_data.append(year_data[date])
                            else:
                                days_data.append(np.nan)
                        except:
                            days_data.append(np.nan)
        
        # Reshape data for heatmap (7 days x n_weeks)
        n_weeks = len(days_data) // 7
        heatmap_data = np.array(days_data[:n_weeks*7]).reshape(n_weeks, 7).T
        
        # Create heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        
        # Set labels
        ax.set_yticks(range(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add month labels
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, (month_name, pos) in enumerate(zip(month_names, month_positions)):
            ax.text(pos, -0.7, month_name, ha='left', va='top', fontsize=10)
        
        # Add vertical lines between months
        for pos in month_positions[1:]:
            ax.axvline(x=pos-0.5, color='white', linewidth=2)
        
        ax.set_title(f'{year} - Daily Traffic Volume', fontsize=12, pad=10)
        ax.set_xlim(-0.5, n_weeks-0.5)
        
        # Remove x-axis labels
        ax.set_xticks([])
        
        # Add colorbar for the last subplot
        if year_idx == n_years - 1:
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
            cbar.set_label('Total Daily Traffic Volume', fontsize=10)
    
    plt.suptitle('Daily Traffic Volume Calendar Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # NO2 Calendar Heatmap
    if 'NO2' in final_dataset.columns:
        daily_no2 = final_dataset.groupby(final_dataset['datetime'].dt.date)['NO2'].mean()
        daily_no2.index = pd.to_datetime(daily_no2.index)
        
        fig = plt.figure(figsize=(20, 4 * n_years + 2))
        
        for year_idx, year in enumerate(years):
            ax = plt.subplot(n_years, 1, year_idx + 1)
            
            # Filter data for this year
            year_data = daily_no2[daily_no2.index.year == year]
            
            # Create calendar layout
            month_positions = []
            days_data = []
            
            for month in range(1, 13):
                # Get calendar for this month
                month_cal = calendar.monthcalendar(year, month)
                month_positions.append(len(days_data))
                
                # Process each week
                for week_num, week in enumerate(month_cal):
                    for day_num, day in enumerate(week):
                        if day == 0:
                            days_data.append(np.nan)
                        else:
                            try:
                                date = pd.Timestamp(year, month, day)
                                if date in year_data.index:
                                    days_data.append(year_data[date])
                                else:
                                    days_data.append(np.nan)
                            except:
                                days_data.append(np.nan)
            
            # Reshape data for heatmap (7 days x n_weeks)
            n_weeks = len(days_data) // 7
            heatmap_data = np.array(days_data[:n_weeks*7]).reshape(n_weeks, 7).T
            
            # Create heatmap with diverging colormap centered at EU limit
            vmin = np.nanmin(heatmap_data)
            vmax = np.nanmax(heatmap_data)
            vcenter = 40  # EU annual limit for NO2
            
            # Create normalized colormap
            from matplotlib.colors import TwoSlopeNorm
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
            
            im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r', 
                          norm=norm, interpolation='nearest')
            
            # Set labels
            ax.set_yticks(range(7))
            ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            
            # Add month labels
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for i, (month_name, pos) in enumerate(zip(month_names, month_positions)):
                ax.text(pos, -0.7, month_name, ha='left', va='top', fontsize=10)
            
            # Add vertical lines between months
            for pos in month_positions[1:]:
                ax.axvline(x=pos-0.5, color='white', linewidth=2)
            
            ax.set_title(f'{year} - Daily NO2 Concentration', fontsize=12, pad=10)
            ax.set_xlim(-0.5, n_weeks-0.5)
            
            # Remove x-axis labels
            ax.set_xticks([])
            
            # Add colorbar for the last subplot
            if year_idx == n_years - 1:
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.05)
                cbar.set_label('Average Daily NO2 (μg/m³)', fontsize=10)
                # Add EU limit line on colorbar
                cbar.ax.axvline(x=40, color='red', linewidth=2, linestyle='--')
                cbar.ax.text(40, 1.5, 'EU Limit', ha='center', va='bottom', fontsize=8, color='red')
        
        plt.suptitle('Daily NO2 Concentration Calendar Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
# ========================================
# ADVANCED TIME SERIES VISUALIZATIONS
# ========================================

def create_advanced_timeseries_plots(final_dataset):
    """
    Create advanced time series visualizations including decomposition and rolling statistics
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 1. Time series with rolling statistics
    ax1 = fig.add_subplot(gs[0, :])
    
    if 'datetime' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        # Sort by datetime
        ts_data = final_dataset.sort_values('datetime').copy()
        
        # Calculate rolling statistics
        ts_data['rolling_mean_24h'] = ts_data['total_vehicles'].rolling(window=24, center=True).mean()
        ts_data['rolling_std_24h'] = ts_data['total_vehicles'].rolling(window=24, center=True).std()
        ts_data['rolling_mean_7d'] = ts_data['total_vehicles'].rolling(window=24*7, center=True).mean()
        
        # Plot
        ax1.plot(ts_data['datetime'], ts_data['total_vehicles'], 
                alpha=0.3, color='gray', linewidth=0.5, label='Hourly Data')
        ax1.plot(ts_data['datetime'], ts_data['rolling_mean_24h'], 
                color='blue', linewidth=2, label='24-hour Moving Average')
        ax1.plot(ts_data['datetime'], ts_data['rolling_mean_7d'], 
                color='red', linewidth=2, label='7-day Moving Average')
        
        # Add confidence bands
        ax1.fill_between(ts_data['datetime'], 
                        ts_data['rolling_mean_24h'] - 2*ts_data['rolling_std_24h'],
                        ts_data['rolling_mean_24h'] + 2*ts_data['rolling_std_24h'],
                        alpha=0.2, color='blue', label='95% CI')
        
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Vehicle Count', fontsize=12)
        ax1.set_title('Traffic Time Series with Rolling Statistics', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Day-of-week effect visualization
    ax2 = fig.add_subplot(gs[1, 0])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'day_of_week', 'total_vehicles']):
        # Calculate average hourly pattern for each day
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = plt.cm.rainbow(np.linspace(0, 1, 7))
        
        for day, color, label in zip(range(7), colors, days):
            day_data = final_dataset[final_dataset['day_of_week'] == day]
            hourly_avg = day_data.groupby('hour_of_day')['total_vehicles'].mean()
            ax2.plot(hourly_avg.index, hourly_avg.values, 'o-', 
                    color=color, label=label, alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Average Vehicle Count', fontsize=12)
        ax2.set_title('Hourly Patterns by Day of Week', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 3))
    
    
    # 3. Pollution lag analysis
    ax4 = fig.add_subplot(gs[2, :])
    
    if all(col in final_dataset.columns for col in ['total_vehicles', 'NO2']):
        # Calculate cross-correlation
        lags = range(0, 25)
        cross_corr = []
        
        for lag in lags:
            if lag == 0:
                corr = final_dataset['total_vehicles'].corr(final_dataset['NO2'])
            else:
                # Shift NO2 by lag hours
                shifted_no2 = final_dataset['NO2'].shift(-lag)
                corr = final_dataset['total_vehicles'].corr(shifted_no2)
            cross_corr.append(corr)
        
        ax4.bar(lags, cross_corr, color='darkgreen', alpha=0.8)
        ax4.axhline(0, color='black', linewidth=0.5)
        
        # Find and annotate maximum correlation
        max_lag = np.argmax(np.abs(cross_corr))
        max_corr = cross_corr[max_lag]
        ax4.annotate(f'Max correlation at lag {max_lag}h: {max_corr:.3f}',
                    xy=(max_lag, max_corr),
                    xytext=(max_lag + 5, max_corr),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        ax4.set_xlabel('Lag (hours)', fontsize=12)
        ax4.set_ylabel('Cross-correlation', fontsize=12)
        ax4.set_title('Traffic-NO2 Cross-correlation (Traffic leads NO2)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 4. Seasonal decomposition preview
    ax5 = fig.add_subplot(gs[3, :])
    
    if 'datetime' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        # Create hourly time series
        hourly_ts = final_dataset.groupby('datetime')['total_vehicles'].mean()
        
        # Simple trend extraction using rolling mean
        trend = hourly_ts.rolling(window=24*7, center=True).mean()
        detrended = hourly_ts - trend
        
        # Plot original and trend
        ax5.plot(hourly_ts.index, hourly_ts.values, alpha=0.5, label='Original', color='gray')
        ax5.plot(trend.index, trend.values, linewidth=3, label='Trend', color='red')
        
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylabel('Vehicle Count', fontsize=12)
        ax5.set_title('Traffic Time Series Trend Component', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
