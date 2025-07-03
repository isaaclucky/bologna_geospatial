import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define WHO/EU Air Quality Standards
AQ_STANDARDS = {
    'NO2': {
        'WHO_annual': 10,  # μg/m³
        'WHO_24h': 25,     # μg/m³
        'EU_annual': 40,   # μg/m³
        'EU_hourly': 200   # μg/m³
    },
    'PM10': {
        'WHO_annual': 15,  # μg/m³
        'WHO_24h': 45,     # μg/m³
        'EU_annual': 40,   # μg/m³
        'EU_24h': 50       # μg/m³
    },
    'PM25': {
        'WHO_annual': 5,   # μg/m³
        'WHO_24h': 15,     # μg/m³
        'EU_annual': 25,   # μg/m³
    }
}

# ========================================
# 2.1 TRAFFIC DATA EXPLORATION
# ========================================

def explore_traffic_distributions(df_bol, df_regional, final_dataset, sample_size=50000):
    """
    Optimized traffic data exploration with distribution analysis
    
    Parameters:
    -----------
    sample_size : int, optional
        Maximum number of rows to sample for visualization (default: 50000)
        Set to None to use all data
    """
    
    # Pre-define vehicle types to avoid repeated list creation
    vehicle_types = ['Light_Count', 'Medium_Count', 'Heavy_Count']
    colors = ['skyblue', 'orange', 'green']
    regional_mapping = {'Light_Count': 'Light', 'Medium_Count': 'Medium', 'Heavy_Count': 'Heavy'}
    
    # Sample data if too large (for visualization purposes)
    def smart_sample(df, n_samples):
        if df is None or len(df) == 0:
            return df
        if sample_size is not None and len(df) > n_samples:
            return df.sample(n=n_samples, random_state=42)
        return df
    
    # Sample datasets
    df_bol_sample = smart_sample(df_bol, sample_size)
    df_regional_sample = smart_sample(df_regional, sample_size)
    
    # Pre-calculate statistics for all vehicle types at once
    bol_stats = {}
    regional_stats = {}
    
    # Vectorized calculation of statistics for Bologna
    for vtype in vehicle_types:
        if vtype in df_bol.columns:
            data = df_bol[vtype].dropna()
            bol_stats[vtype] = {
                'mean': data.mean(),
                'median': data.median(),
                'data': df_bol_sample[vtype].dropna() if vtype in df_bol_sample.columns else pd.Series()
            }
    
    # Vectorized calculation of statistics for Regional
    for vtype, regional_col in regional_mapping.items():
        if regional_col in df_regional.columns:
            data = df_regional[regional_col].dropna()
            regional_stats[regional_col] = {
                'mean': data.mean(),
                'median': data.median(),
                'data': df_regional_sample[regional_col].dropna() if regional_col in df_regional_sample.columns else pd.Series()
            }
    
    # Create figure for vehicle count distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Traffic Volume Distributions by Vehicle Type and Data Source', fontsize=16)
    
    # Optimized histogram plotting with fewer bins for large datasets
    n_bins = 30 if sample_size and sample_size > 10000 else 50
    
    # Bologna traffic distributions
    for idx, (vtype, color) in enumerate(zip(vehicle_types, colors)):
        ax = axes[0, idx]
        if vtype in bol_stats and len(bol_stats[vtype]['data']) > 0:
            # Use pre-calculated data
            data = bol_stats[vtype]['data']
            mean_val = bol_stats[vtype]['mean']
            median_val = bol_stats[vtype]['median']
            
            # Faster histogram using numpy
            counts, bins = np.histogram(data, bins=n_bins)
            ax.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, color=color, edgecolor='black')
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='black', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax.set_xlabel(f'{vtype}')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Bologna - {vtype}')
            ax.legend()
            ax.set_yscale('log')
    
    # Regional traffic distributions
    for idx, (vtype, color) in enumerate(zip(vehicle_types, colors)):
        ax = axes[1, idx]
        regional_col = regional_mapping.get(vtype, vtype)
        if regional_col in regional_stats and len(regional_stats[regional_col]['data']) > 0:
            # Use pre-calculated data
            data = regional_stats[regional_col]['data']
            mean_val = regional_stats[regional_col]['mean']
            median_val = regional_stats[regional_col]['median']
            
            # Faster histogram using numpy
            counts, bins = np.histogram(data, bins=n_bins)
            ax.bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7, color=color, edgecolor='black')
            
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='black', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax.set_xlabel(f'{regional_col} Count')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Regional - {regional_col}')
            ax.legend()
            ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Optimized KDE plots for Total Traffic Volume
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Kernel Density Plots - Total Traffic Volume', fontsize=16)
    
    # Pre-calculate total traffic using vectorized operations
    # Bologna total traffic
    if all(col in df_bol_sample.columns for col in vehicle_types):
        # Vectorized sum
        bol_total = df_bol_sample[vehicle_types].sum(axis=1).dropna()
        
        if len(bol_total) > 0:
            ax = axes[0]
            # Use fewer points for KDE calculation
            kde_points = min(1000, len(bol_total))
            if len(bol_total) > kde_points:
                bol_total_kde = bol_total.sample(n=kde_points, random_state=42)
            else:
                bol_total_kde = bol_total
            
            # Faster KDE using seaborn
            sns.kdeplot(data=bol_total_kde, ax=ax, color='navy', linewidth=2, fill=True, alpha=0.3)
            ax.set_xlabel('Total Vehicle Count')
            ax.set_ylabel('Density')
            ax.set_title('Bologna Traffic Density')
            ax.grid(True, alpha=0.3)
    
    # Regional total traffic
    regional_cols = ['Light', 'Medium', 'Heavy']
    if all(col in df_regional_sample.columns for col in regional_cols):
        # Vectorized sum
        regional_total = df_regional_sample[regional_cols].sum(axis=1).dropna()
        
        if len(regional_total) > 0:
            ax = axes[1]
            # Use fewer points for KDE calculation
            kde_points = min(1000, len(regional_total))
            if len(regional_total) > kde_points:
                regional_total_kde = regional_total.sample(n=kde_points, random_state=42)
            else:
                regional_total_kde = regional_total
            
            # Faster KDE using seaborn
            sns.kdeplot(data=regional_total_kde, ax=ax, color='darkgreen', linewidth=2, fill=True, alpha=0.3)
            ax.set_xlabel('Total Vehicle Count')
            ax.set_ylabel('Density')
            ax.set_title('Regional Traffic Density')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Optimized Heavy Vehicle Proportion Analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Heavy Vehicle Proportion Analysis', fontsize=16)
    
    # Bologna heavy vehicle proportion - vectorized calculation
    if all(col in df_bol.columns for col in ['Heavy_Count'] + vehicle_types):
        # Calculate proportions using vectorized operations
        total_traffic = df_bol[vehicle_types].sum(axis=1)
        # Avoid division by zero
        mask = total_traffic > 0
        heavy_prop = np.zeros(len(df_bol))
        heavy_prop[mask] = df_bol.loc[mask, 'Heavy_Count'] / total_traffic[mask]
        
        # Sample for visualization
        if sample_size and len(heavy_prop) > sample_size:
            sample_indices = np.random.choice(len(heavy_prop), sample_size, replace=False)
            heavy_prop_sample = heavy_prop[sample_indices]
        else:
            heavy_prop_sample = heavy_prop
        
        # Remove zeros and invalid values for better visualization
        heavy_prop_sample = heavy_prop_sample[(heavy_prop_sample > 0) & (heavy_prop_sample <= 1)]
        
        ax = axes[0]
        if len(heavy_prop_sample) > 0:
            ax.hist(heavy_prop_sample, bins=30, alpha=0.7, color='darkred', edgecolor='black')
            ax.axvline(heavy_prop_sample.mean(), color='yellow', linestyle='--', linewidth=2, 
                      label=f'Mean: {heavy_prop_sample.mean():.3f}')
            ax.set_xlabel('Heavy Vehicle Proportion')
            ax.set_ylabel('Frequency')
            ax.set_title('Bologna - Heavy Vehicle Proportion')
            ax.legend()
    
    # Regional heavy vehicle proportion - vectorized calculation
    if all(col in df_regional.columns for col in ['Heavy'] + regional_cols):
        # Calculate proportions using vectorized operations
        total_traffic = df_regional[regional_cols].sum(axis=1)
        # Avoid division by zero
        mask = total_traffic > 0
        heavy_prop = np.zeros(len(df_regional))
        heavy_prop[mask] = df_regional.loc[mask, 'Heavy'] / total_traffic[mask]
        
        # Sample for visualization
        if sample_size and len(heavy_prop) > sample_size:
            sample_indices = np.random.choice(len(heavy_prop), sample_size, replace=False)
            heavy_prop_sample = heavy_prop[sample_indices]
        else:
            heavy_prop_sample = heavy_prop
        
        # Remove zeros and invalid values for better visualization
        heavy_prop_sample = heavy_prop_sample[(heavy_prop_sample > 0) & (heavy_prop_sample <= 1)]
        
        ax = axes[1]
        if len(heavy_prop_sample) > 0:
            ax.hist(heavy_prop_sample, bins=30, alpha=0.7, color='darkred', edgecolor='black')
            ax.axvline(heavy_prop_sample.mean(), color='yellow', linestyle='--', linewidth=2,
                      label=f'Mean: {heavy_prop_sample.mean():.3f}')
            ax.set_xlabel('Heavy Vehicle Proportion')
            ax.set_ylabel('Frequency')
            ax.set_title('Regional - Heavy Vehicle Proportion')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print sampling information
    if sample_size:
        print(f"\n📊 Data Sampling Information:")
        print(f"Original Bologna data: {len(df_bol):,} rows")
        print(f"Sampled Bologna data: {len(df_bol_sample):,} rows")
        print(f"Original Regional data: {len(df_regional):,} rows")
        print(f"Sampled Regional data: {len(df_regional_sample):,} rows")

def generate_traffic_summary_statistics(df_bol, df_regional, final_dataset):
    """
    Generate comprehensive summary statistics for traffic data
    """
    
    print("=" * 80)
    print("TRAFFIC DATA SUMMARY STATISTICS")
    print("=" * 80)
    
    # Summary by Data Source
    print("\n1. SUMMARY BY DATA SOURCE")
    print("-" * 50)
    
    # Bologna statistics
    if 'Light_Count' in df_bol.columns:
        print("\nBologna Traffic Statistics:")
        bologna_stats = df_bol[['Light_Count', 'Medium_Count', 'Heavy_Count']].describe()
        print(bologna_stats.round(2))
    
    # Regional statistics
    if 'Light' in df_regional.columns:
        print("\nRegional Traffic Statistics:")
        regional_stats = df_regional[['Light', 'Medium', 'Heavy']].describe()
        print(regional_stats.round(2))
    
    # Combined statistics from final dataset
    if 'data_source' in final_dataset.columns:
        print("\n2. SUMMARY BY DATA SOURCE (Final Dataset)")
        print("-" * 50)
        
        for source in final_dataset['data_source'].unique():
            print(f"\nData Source: {source}")
            source_data = final_dataset[final_dataset['data_source'] == source]
            if 'total_vehicles' in source_data.columns:
                print(source_data[['Light_Count', 'Medium_Count', 'Heavy_Count', 'total_vehicles']].describe().round(2))
    
    # Summary by Station/Monitoring Point
    print("\n3. SUMMARY BY MONITORING STATION")
    print("-" * 50)
    
    # Bologna stations
    if 'id_uni' in df_bol.columns:
        print("\nTop 10 Bologna Monitoring Points by Traffic Volume:")
        station_summary = df_bol.groupby('id_uni').agg({
            'Light_Count': ['mean', 'std', 'sum'],
            'Medium_Count': ['mean', 'std', 'sum'],
            'Heavy_Count': ['mean', 'std', 'sum']
        }).round(2)
        
        # Calculate total traffic per station
        total_traffic = df_bol.groupby('id_uni')[['Light_Count', 'Medium_Count', 'Heavy_Count']].sum().sum(axis=1)
        top_stations = total_traffic.nlargest(10)
        print(station_summary.loc[top_stations.index])
    
    # Time-based summaries
    print("\n4. TEMPORAL SUMMARY STATISTICS")
    print("-" * 50)
    
    if 'hour_of_day' in final_dataset.columns:
        print("\nTraffic by Hour of Day:")
        hourly_stats = final_dataset.groupby('hour_of_day')['total_vehicles'].agg(['mean', 'std', 'min', 'max']).round(2)
        print(hourly_stats)
    
    if 'day_of_week' in final_dataset.columns:
        print("\nTraffic by Day of Week (0=Monday, 6=Sunday):")
        daily_stats = final_dataset.groupby('day_of_week')['total_vehicles'].agg(['mean', 'std', 'min', 'max']).round(2)
        print(daily_stats)
    
    if 'is_weekend' in final_dataset.columns:
        print("\nWeekday vs Weekend Comparison:")
        weekend_stats = final_dataset.groupby('is_weekend')['total_vehicles'].agg(['mean', 'std', 'count']).round(2)
        weekend_stats.index = ['Weekday', 'Weekend']
        print(weekend_stats)


# ========================================
# 2.2 AIR QUALITY DATA EXPLORATION
# ========================================

def explore_air_quality_distributions(df_no2, df_pm10, df_pm25, stations_df):
    """
    Comprehensive air quality data exploration with distribution analysis
    """
    
    # Prepare data - melt dataframes for easier plotting
    no2_stations = [col for col in df_no2.columns if not col.startswith('v_') and col != 'data']
    pm10_stations = [col for col in df_pm10.columns if not col.startswith('v_') and col != 'data']
    pm25_stations = [col for col in df_pm25.columns if not col.startswith('v_') and col != 'data']
    
    # Create comprehensive pollution distribution plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Air Quality Pollutant Distributions', fontsize=16)
    
    # NO2 Distribution
    ax = axes[0, 0]
    for station in no2_stations:
        if station in df_no2.columns:
            data = df_no2[station].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=station, density=True)
    ax.set_xlabel('NO2 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('NO2 Distribution by Station')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add WHO/EU standards lines
    ax.axvline(AQ_STANDARDS['NO2']['EU_hourly'], color='red', linestyle='--', linewidth=2, label='EU Hourly Limit')
    ax.axvline(AQ_STANDARDS['NO2']['EU_annual'], color='orange', linestyle='--', linewidth=2, label='EU Annual Limit')
    
    # NO2 Box plots
    ax = axes[0, 1]
    no2_data = []
    for station in no2_stations:
        if station in df_no2.columns:
            no2_data.append(df_no2[station].dropna())
    ax.boxplot(no2_data, labels=no2_stations, vert=True)
    ax.set_ylabel('NO2 Concentration (μg/m³)')
    ax.set_title('NO2 Distribution Comparison')
    ax.axhline(AQ_STANDARDS['NO2']['EU_hourly'], color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(AQ_STANDARDS['NO2']['EU_annual'], color='orange', linestyle='--', linewidth=1, alpha=0.7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # PM10 Distribution
    ax = axes[1, 0]
    for station in pm10_stations:
        if station in df_pm10.columns:
            data = df_pm10[station].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=station, density=True)
    ax.set_xlabel('PM10 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('PM10 Distribution by Station')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add WHO/EU standards lines
    ax.axvline(AQ_STANDARDS['PM10']['EU_24h'], color='red', linestyle='--', linewidth=2, label='EU Daily Limit')
    ax.axvline(AQ_STANDARDS['PM10']['EU_annual'], color='orange', linestyle='--', linewidth=2, label='EU Annual Limit')
    

    # PM10 Box plots
    ax = axes[1, 1]
    pm10_data = []
    for station in pm10_stations:
        if station in df_pm10.columns:
            pm10_data.append(df_pm10[station].dropna())
    ax.boxplot(pm10_data, labels=pm10_stations, vert=True)
    ax.set_ylabel('PM10 Concentration (μg/m³)')
    ax.set_title('PM10 Distribution Comparison')
    ax.axhline(AQ_STANDARDS['PM10']['EU_24h'], color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(AQ_STANDARDS['PM10']['EU_annual'], color='orange', linestyle='--', linewidth=1, alpha=0.7)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # PM2.5 Distribution
    ax = axes[2, 0]
    for station in pm25_stations:
        if station in df_pm25.columns:
            data = df_pm25[station].dropna()
            ax.hist(data, bins=50, alpha=0.5, label=station, density=True)
    ax.set_xlabel('PM2.5 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('PM2.5 Distribution by Station')
    ax.legend()
    
    # Add WHO/EU standards lines
    ax.axvline(AQ_STANDARDS['PM25']['EU_annual'], color='orange', linestyle='--', linewidth=2, label='EU Annual Limit')
    ax.axvline(AQ_STANDARDS['PM25']['WHO_24h'], color='green', linestyle='--', linewidth=2, label='WHO Daily Limit')
    
    # PM2.5 Box plots
    ax = axes[2, 1]
    pm25_data = []
    for station in pm25_stations:
        if station in df_pm25.columns:
            pm25_data.append(df_pm25[station].dropna())
    if pm25_data:  # Only plot if data exists
        ax.boxplot(pm25_data, labels=pm25_stations, vert=True)
        ax.set_ylabel('PM2.5 Concentration (μg/m³)')
        ax.set_title('PM2.5 Distribution Comparison')
        ax.axhline(AQ_STANDARDS['PM25']['EU_annual'], color='orange', linestyle='--', linewidth=1, alpha=0.7)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Kernel Density Plots for all pollutants
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Pollutant Concentration Kernel Density Estimates', fontsize=16)
    
    # NO2 KDE
    ax = axes[0]
    for station in no2_stations:
        if station in df_no2.columns:
            data = df_no2[station].dropna()
            data.plot(kind='density', ax=ax, label=station, linewidth=2)
    ax.set_xlabel('NO2 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('NO2 Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PM10 KDE
    ax = axes[1]
    for station in pm10_stations:
        if station in df_pm10.columns:
            data = df_pm10[station].dropna()
            data.plot(kind='density', ax=ax, label=station, linewidth=2)
    ax.set_xlabel('PM10 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('PM10 Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # PM2.5 KDE
    ax = axes[2]
    for station in pm25_stations:
        if station in df_pm25.columns:
            data = df_pm25[station].dropna()
            data.plot(kind='density', ax=ax, label=station, linewidth=2)
    ax.set_xlabel('PM2.5 Concentration (μg/m³)')
    ax.set_ylabel('Density')
    ax.set_title('PM2.5 Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
 

def analyze_air_quality_standards_compliance(df_no2, df_pm10, df_pm25):
    """
    Analyze compliance with WHO/EU air quality standards
    """
    
    print("=" * 80)
    print("AIR QUALITY STANDARDS COMPLIANCE ANALYSIS")
    print("=" * 80)
    
    # NO2 Compliance Analysis
    print("\n1. NO2 COMPLIANCE ANALYSIS")
    print("-" * 50)
    
    no2_stations = [col for col in df_no2.columns if not col.startswith('v_') and col != 'data' and col != 'datetime']
    
    for station in no2_stations:
        if station in df_no2.columns:
            data = df_no2[station].dropna()
            
            # Calculate exceedances
            eu_hourly_exceed = (data > AQ_STANDARDS['NO2']['EU_hourly']).sum()
            eu_hourly_percent = (eu_hourly_exceed / len(data)) * 100
            
            # Annual average
            annual_avg = data.mean()
            
            print(f"\n{station}:")
            print(f"  - Annual Average: {annual_avg:.2f} μg/m³ (EU Limit: {AQ_STANDARDS['NO2']['EU_annual']} μg/m³)")
            print(f"  - Status: {'⚠️ EXCEEDS' if annual_avg > AQ_STANDARDS['NO2']['EU_annual'] else '✓ COMPLIANT'}")
            print(f"  - Hourly Exceedances: {eu_hourly_exceed} times ({eu_hourly_percent:.2f}% of measurements)")
            print(f"  - Max Hourly: {data.max():.2f} μg/m³")
    
    # PM10 Compliance Analysis
    print("\n2. PM10 COMPLIANCE ANALYSIS")
    print("-" * 50)
    
    pm10_stations = [col for col in df_pm10.columns if not col.startswith('v_') and col != 'data']
    
    for station in pm10_stations:
        if station in df_pm10.columns:
            data = df_pm10[station].dropna()
            
            # Calculate exceedances
            eu_daily_exceed = (data > AQ_STANDARDS['PM10']['EU_24h']).sum()
            eu_daily_percent = (eu_daily_exceed / len(data)) * 100
            
            # Annual average
            annual_avg = data.mean()
            
            print(f"\n{station}:")
            print(f"  - Annual Average: {annual_avg:.2f} μg/m³ (EU Limit: {AQ_STANDARDS['PM10']['EU_annual']} μg/m³)")
            print(f"  - Status: {'⚠️ EXCEEDS' if annual_avg > AQ_STANDARDS['PM10']['EU_annual'] else '✓ COMPLIANT'}")
            print(f"  - Daily Exceedances: {eu_daily_exceed} days ({eu_daily_percent:.2f}% of days)")
            print(f"  - Note: EU allows max 35 exceedances/year. Status: {'⚠️ EXCEEDS' if eu_daily_exceed > 35 else '✓ COMPLIANT'}")
            print(f"  - Max Daily: {data.max():.2f} μg/m³")
    
    # PM2.5 Compliance Analysis
    print("\n3. PM2.5 COMPLIANCE ANALYSIS")
    print("-" * 50)
    
    pm25_stations = [col for col in df_pm25.columns if not col.startswith('v_') and col != 'data']
    
    for station in pm25_stations:
        if station in df_pm25.columns:
            data = df_pm25[station].dropna()
            
            # Annual average
            annual_avg = data.mean()
            
            # WHO guideline exceedances
            who_daily_exceed = (data > AQ_STANDARDS['PM25']['WHO_24h']).sum()
            who_daily_percent = (who_daily_exceed / len(data)) * 100
            
            print(f"\n{station}:")
            print(f"  - Annual Average: {annual_avg:.2f} μg/m³ (EU Limit: {AQ_STANDARDS['PM25']['EU_annual']} μg/m³)")
            print(f"  - Status: {'⚠️ EXCEEDS' if annual_avg > AQ_STANDARDS['PM25']['EU_annual'] else '✓ COMPLIANT'}")
            print(f"  - WHO Daily Guideline Exceedances: {who_daily_exceed} days ({who_daily_percent:.2f}% of days)")
            print(f"  - Max Daily: {data.max():.2f} μg/m³")

    """
    Analyze validation flags to understand data quality patterns
    """
    
    print("=" * 80)
    print("DATA VALIDATION FLAG ANALYSIS")
    print("=" * 80)
    
    # Function to analyze validation flags for a dataset
    def analyze_flags(df, pollutant_name):
        print(f"\n{pollutant_name} Validation Flag Analysis:")
        print("-" * 50)
        
        # Get validation flag columns
        v_columns = [col for col in df.columns if col.startswith('v_')]
        station_columns = [col.replace('v_', '') for col in v_columns]
        
        for v_col, station in zip(v_columns, station_columns):
            if v_col in df.columns:
                # Count flag values
                flag_counts = df[v_col].value_counts().sort_index()
                total_records = len(df)
                
                print(f"\n{station}:")
                print(f"  Total Records: {total_records}")
                
                for flag_value, count in flag_counts.items():
                    percentage = (count / total_records) * 100
                    print(f"  Flag {flag_value}: {count} records ({percentage:.2f}%)")
                
                # Check for patterns in invalid data
                if station in df.columns:
                    invalid_mask = df[v_col] != 1  # Assuming 1 is valid
                    if invalid_mask.sum() > 0:
                        invalid_data = df.loc[invalid_mask, station]
                        print(f"  Invalid Data Statistics:")
                        print(f"    - Count: {invalid_mask.sum()}")
                        print(f"    - Mean: {invalid_data.mean():.2f}")
                        print(f"    - Std: {invalid_data.std():.2f}")
                        print(f"    - Min: {invalid_data.min():.2f}")
                        print(f"    - Max: {invalid_data.max():.2f}")
    
    # Analyze each pollutant
    analyze_flags(df_no2, "NO2")
    analyze_flags(df_pm10, "PM10")
    analyze_flags(df_pm25, "PM2.5")
    


def create_station_comparison_plot(df_no2, df_pm10, df_pm25, stations_df):
    """
    Create comprehensive station-wise pollutant level comparisons
    """
    
    # Calculate station statistics
    station_stats = {}
    
    # NO2 statistics
    no2_stations = [col for col in df_no2.columns if not col.startswith('v_') and col != 'data']
    for station in no2_stations:
        if station in df_no2.columns:
            data = df_no2[station].dropna()
            station_stats[station] = {
                'NO2_mean': data.mean(),
                'NO2_median': data.median(),
                'NO2_std': data.std(),
                'NO2_p95': data.quantile(0.95),
                'NO2_max': data.max()
            }
    
    # PM10 statistics
    pm10_stations = [col for col in df_pm10.columns if not col.startswith('v_') and col != 'data']
    for station in pm10_stations:
        if station in df_pm10.columns:
            data = df_pm10[station].dropna()
            if station not in station_stats:
                station_stats[station] = {}
            station_stats[station].update({
                'PM10_mean': data.mean(),
                'PM10_median': data.median(),
                'PM10_std': data.std(),
                'PM10_p95': data.quantile(0.95),
                'PM10_max': data.max()
            })
    
    # PM2.5 statistics
    pm25_stations = [col for col in df_pm25.columns if not col.startswith('v_') and col != 'data']
    for station in pm25_stations:
        if station in df_pm25.columns:
            data = df_pm25[station].dropna()
            if station not in station_stats:
                station_stats[station] = {}
            station_stats[station].update({
                'PM25_mean': data.mean(),
                'PM25_median': data.median(),
                'PM25_std': data.std(),
                'PM25_p95': data.quantile(0.95),
                'PM25_max': data.max()
            })
    
    # Convert to DataFrame for easier plotting
    stats_df = pd.DataFrame(station_stats).T
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Station-wise Air Quality Comparison', fontsize=16)
    
    # NO2 Mean Comparison
    ax = axes[0, 0]
    if 'NO2_mean' in stats_df.columns:
        stats_df['NO2_mean'].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.axhline(AQ_STANDARDS['NO2']['EU_annual'], color='red', linestyle='--', label='EU Annual Limit')
        ax.set_title('NO2 - Mean Concentration')
        ax.set_ylabel('Concentration (μg/m³)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # PM10 Mean Comparison
    ax = axes[0, 1]
    if 'PM10_mean' in stats_df.columns:
        stats_df['PM10_mean'].plot(kind='bar', ax=ax, color='orange', edgecolor='black')
        ax.axhline(AQ_STANDARDS['PM10']['EU_annual'], color='red', linestyle='--', label='EU Annual Limit')
        ax.set_title('PM10 - Mean Concentration')
        ax.set_ylabel('Concentration (μg/m³)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # PM2.5 Mean Comparison
    ax = axes[0, 2]
    if 'PM25_mean' in stats_df.columns:
        stats_df['PM25_mean'].plot(kind='bar', ax=ax, color='green', edgecolor='black')
        ax.axhline(AQ_STANDARDS['PM25']['EU_annual'], color='red', linestyle='--', label='EU Annual Limit')
        ax.set_title('PM2.5 - Mean Concentration')
        ax.set_ylabel('Concentration (μg/m³)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 95th Percentile Comparisons
    # NO2 P95
    ax = axes[1, 0]
    if 'NO2_p95' in stats_df.columns:
        stats_df['NO2_p95'].plot(kind='bar', ax=ax, color='darkblue', edgecolor='black')
        ax.axhline(AQ_STANDARDS['NO2']['EU_hourly'], color='red', linestyle='--', label='EU Hourly Limit')
        ax.set_title('NO2 - 95th Percentile')
        ax.set_ylabel('Concentration (μg/m³)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # PM10 P95
    ax = axes[1, 1]
    if 'PM10_p95' in stats_df.columns:
        stats_df['PM10_p95'].plot(kind='bar', ax=ax, color='darkorange', edgecolor='black')
        ax.axhline(AQ_STANDARDS['PM10']['EU_24h'], color='red', linestyle='--', label='EU Daily Limit')
        ax.set_title('PM10 - 95th Percentile')
        ax.set_ylabel('Concentration (μg/m³)')
        ax.legend()
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Coefficient of Variation (CV) for all pollutants
    ax = axes[1, 2]
    cv_data = pd.DataFrame()
    if 'NO2_mean' in stats_df.columns and 'NO2_std' in stats_df.columns:
        cv_data['NO2_CV'] = stats_df['NO2_std'] / stats_df['NO2_mean']
    if 'PM10_mean' in stats_df.columns and 'PM10_std' in stats_df.columns:
        cv_data['PM10_CV'] = stats_df['PM10_std'] / stats_df['PM10_mean']
    if 'PM25_mean' in stats_df.columns and 'PM25_std' in stats_df.columns:
        cv_data['PM25_CV'] = stats_df['PM25_std'] / stats_df['PM25_mean']
    
    if not cv_data.empty:
        cv_data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Coefficient of Variation by Station')
        ax.set_ylabel('CV (Std/Mean)')
        ax.legend(['NO2', 'PM10', 'PM2.5'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics table
    print("\n" + "=" * 80)
    print("STATION-WISE POLLUTANT STATISTICS SUMMARY")
    print("=" * 80)
    print(stats_df.round(2))


def visualize_traffic_summary_statistics(df_bol, df_regional, final_dataset):
    """
    Comprehensive summary statistics for traffic data
    """
    
    # Set up the visualization style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors_palette = sns.color_palette("husl", 8)
    
    # Create a figure with subplots for different statistics
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # ========================================
    # 1. DATA SOURCE COMPARISON
    # ========================================
    
    # Prepare data for comparison
    vehicle_types_bol = ['Light_Count', 'Medium_Count', 'Heavy_Count']
    vehicle_types_reg = ['Light', 'Medium', 'Heavy']
    
    # Calculate summary statistics for both sources
    if all(col in df_bol.columns for col in vehicle_types_bol):
        bol_summary = df_bol[vehicle_types_bol].describe().T
        bol_summary['source'] = 'Bologna'
        bol_summary['vehicle_type'] = ['Light', 'Medium', 'Heavy']
    
    if all(col in df_regional.columns for col in vehicle_types_reg):
        reg_summary = df_regional[vehicle_types_reg].describe().T
        reg_summary['source'] = 'Regional'
        reg_summary['vehicle_type'] = ['Light', 'Medium', 'Heavy']
    
    # Plot 1: Mean vehicle counts by source
    ax1 = fig.add_subplot(gs[0, :2])
    
    if 'bol_summary' in locals() and 'reg_summary' in locals():
        x = np.arange(3)
        width = 0.35
        
        ax1.bar(x - width/2, bol_summary['mean'], width, label='Bologna', color=colors_palette[0], alpha=0.8)
        ax1.bar(x + width/2, reg_summary['mean'], width, label='Regional', color=colors_palette[1], alpha=0.8)
        
        ax1.set_xlabel('Vehicle Type')
        ax1.set_ylabel('Mean Count')
        ax1.set_title('Average Vehicle Counts by Data Source', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['Light', 'Medium', 'Heavy'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation comparison
    ax2 = fig.add_subplot(gs[0, 2])
    
    if 'bol_summary' in locals() and 'reg_summary' in locals():
        bol_cv = bol_summary['std'] / bol_summary['mean']
        reg_cv = reg_summary['std'] / reg_summary['mean']
        
        x = np.arange(3)
        ax2.bar(x - width/2, bol_cv, width, label='Bologna', color=colors_palette[2], alpha=0.8)
        ax2.bar(x + width/2, reg_cv, width, label='Regional', color=colors_palette[3], alpha=0.8)
        
        ax2.set_xlabel('Vehicle Type')
        ax2.set_ylabel('Coefficient of Variation')
        ax2.set_title('Traffic Variability by Source', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Light', 'Medium', 'Heavy'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ========================================
    # 2. TOP STATIONS ANALYSIS
    # ========================================
    
    # Plot 3: Top 10 Bologna stations by total traffic
    ax3 = fig.add_subplot(gs[1, :])
    
    if 'id_uni' in df_bol.columns and all(col in df_bol.columns for col in vehicle_types_bol):
        # Calculate total traffic per station
        station_totals = df_bol.groupby('id_uni')[vehicle_types_bol].sum()
        station_totals['total'] = station_totals.sum(axis=1)
        top_10_stations = station_totals.nlargest(10, 'total')
        
        # Create stacked bar chart
        bottom = np.zeros(10)
        colors = [colors_palette[4], colors_palette[5], colors_palette[6]]
        
        for idx, (vtype, color) in enumerate(zip(['Light_Count', 'Medium_Count', 'Heavy_Count'], colors)):
            values = top_10_stations[vtype].values
            ax3.bar(range(10), values, bottom=bottom, label=vtype.replace('_Count', ''), 
                   color=color, alpha=0.8, edgecolor='white', linewidth=1)
            bottom += values
        
        ax3.set_xlabel('Station ID')
        ax3.set_ylabel('Total Vehicle Count')
        ax3.set_title('Top 10 Bologna Monitoring Stations by Traffic Volume', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(10))
        ax3.set_xticklabels(top_10_stations.index, rotation=45)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')
    
    # ========================================
    # 3. TEMPORAL PATTERNS
    # ========================================
    
    # Plot 4: Hourly traffic pattern
    ax4 = fig.add_subplot(gs[2, 0])
    
    if 'hour_of_day' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        hourly_stats = final_dataset.groupby('hour_of_day')['total_vehicles'].agg(['mean', 'std'])
        
        # Plot with confidence interval
        hours = hourly_stats.index
        means = hourly_stats['mean']
        stds = hourly_stats['std']
        
        ax4.plot(hours, means, 'o-', color=colors_palette[0], linewidth=2, markersize=6)
        ax4.fill_between(hours, means - stds, means + stds, alpha=0.2, color=colors_palette[0])
        
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Average Vehicle Count')
        ax4.set_title('24-Hour Traffic Pattern', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(0, 24, 3))
        ax4.grid(True, alpha=0.3)
        
        # Highlight rush hours
        ax4.axvspan(7, 9, alpha=0.1, color='red', label='Morning Rush')
        ax4.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Rush')
        ax4.legend(loc='upper right')
    
    # Plot 5: Daily traffic pattern
    ax5 = fig.add_subplot(gs[2, 1])
    
    if 'day_of_week' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        daily_stats = final_dataset.groupby('day_of_week')['total_vehicles'].agg(['mean', 'std'])
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create bar plot with error bars
        x_pos = np.arange(7)
        ax5.bar(x_pos, daily_stats['mean'], yerr=daily_stats['std'], 
               color=colors_palette[1], alpha=0.8, capsize=5, edgecolor='black', linewidth=1)
        
        ax5.set_xlabel('Day of Week')
        ax5.set_ylabel('Average Vehicle Count')
        ax5.set_title('Weekly Traffic Pattern', fontsize=14, fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(days)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Highlight weekend
        ax5.axvspan(4.5, 6.5, alpha=0.1, color='green')
        ax5.text(5.5, ax5.get_ylim()[1]*0.95, 'Weekend', ha='center', fontsize=10, style='italic')
    
    # Plot 6: Weekend vs Weekday comparison
    ax6 = fig.add_subplot(gs[2, 2])
    
    if 'is_weekend' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        weekend_data = final_dataset[final_dataset['is_weekend'] == True]['total_vehicles']
        weekday_data = final_dataset[final_dataset['is_weekend'] == False]['total_vehicles']
        
        # Create violin plot
        parts = ax6.violinplot([weekday_data.dropna(), weekend_data.dropna()], 
                               positions=[0, 1], showmeans=True, showmedians=True)
        
        # Customize colors
        for pc, color in zip(parts['bodies'], [colors_palette[2], colors_palette[3]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(['Weekday', 'Weekend'])
        ax6.set_ylabel('Vehicle Count')
        ax6.set_title('Weekday vs Weekend Distribution', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        weekday_mean = weekday_data.mean()
        weekend_mean = weekend_data.mean()
        reduction = ((weekday_mean - weekend_mean) / weekday_mean) * 100
        ax6.text(0.5, ax6.get_ylim()[1]*0.95, f'Weekend reduction: {reduction:.1f}%', 
                ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # ========================================
    # 4. HEATMAP OF HOURLY PATTERNS BY DAY
    # ========================================
    
    ax7 = fig.add_subplot(gs[3, :])
    
    if all(col in final_dataset.columns for col in ['hour_of_day', 'day_of_week', 'total_vehicles']):
        # Create pivot table for heatmap
        heatmap_data = final_dataset.pivot_table(
            values='total_vehicles', 
            index='hour_of_day', 
            columns='day_of_week', 
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='.0f', 
                   cbar_kws={'label': 'Average Vehicle Count'}, ax=ax7)
        
        ax7.set_xlabel('Day of Week')
        ax7.set_ylabel('Hour of Day')
        ax7.set_title('Traffic Intensity Heatmap: Hour vs Day of Week', fontsize=14, fontweight='bold')
        ax7.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # Add grid for better readability
        ax7.set_xticks(np.arange(7) + 0.5, minor=True)
        ax7.set_yticks(np.arange(24) + 0.5, minor=True)
        ax7.grid(which="minor", color="white", linestyle='-', linewidth=0.5)
    
    plt.suptitle('Traffic Data Summary Statistics Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    
    # ========================================
    # VEHICLE TYPE PROPORTION VISUALIZATION
    # ========================================
    
    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig3.suptitle('Vehicle Type Distribution Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Pie chart for Bologna vehicle distribution
    ax = axes[0, 0]
    if all(col in df_bol.columns for col in vehicle_types_bol):
        total_by_type = df_bol[vehicle_types_bol].sum()
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        wedges, texts, autotexts = ax.pie(total_by_type, labels=['Light', 'Medium', 'Heavy'], 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Bologna - Vehicle Type Distribution')
        
        # Add total count in center
        total_count = total_by_type.sum()
        ax.text(0, 0, f'Total:\n{total_count:,.0f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot 2: Pie chart for Regional vehicle distribution
    ax = axes[0, 1]
    if all(col in df_regional.columns for col in vehicle_types_reg):
        total_by_type = df_regional[vehicle_types_reg].sum()
        colors = ['#66c2a5', '#fc8d62', '#8da0cb']
        wedges, texts, autotexts = ax.pie(total_by_type, labels=['Light', 'Medium', 'Heavy'], 
                                          colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title('Regional - Vehicle Type Distribution')
        
        # Add total count in center
        total_count = total_by_type.sum()
        ax.text(0, 0, f'Total:\n{total_count:,.0f}', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Plot 3: Temporal heavy vehicle percentage
    ax = axes[1, 0]
    if all(col in final_dataset.columns for col in ['hour_of_day', 'Heavy_Count', 'total_vehicles']):
        hourly_heavy = final_dataset.groupby('hour_of_day').agg({
            'Heavy_Count': 'sum',
            'total_vehicles': 'sum'
        })
        hourly_heavy['heavy_percentage'] = (hourly_heavy['Heavy_Count'] / hourly_heavy['total_vehicles']) * 100
        
        ax.plot(hourly_heavy.index, hourly_heavy['heavy_percentage'], 'o-', 
                color='darkred', linewidth=2, markersize=8)
        ax.fill_between(hourly_heavy.index, 0, hourly_heavy['heavy_percentage'], 
                       alpha=0.3, color='darkred')
        
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Heavy Vehicle Percentage (%)')
        ax.set_title('Heavy Vehicle Percentage by Hour')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(0, 24, 3))
        
        # Add average line
        avg_heavy = hourly_heavy['heavy_percentage'].mean()
        ax.axhline(avg_heavy, color='black', linestyle='--', alpha=0.5, 
                  label=f'Daily Average: {avg_heavy:.1f}%')
        ax.legend()
    
    # Plot 4: Seasonal/Monthly patterns if available
    ax = axes[1, 1]
    if 'month' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        monthly_stats = final_dataset.groupby('month')['total_vehicles'].agg(['mean', 'std'])
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        if len(monthly_stats) > 0:
            x_pos = monthly_stats.index - 1  # Adjust for 0-based indexing
            ax.bar(x_pos, monthly_stats['mean'], yerr=monthly_stats['std'], 
                   color='skyblue', alpha=0.8, capsize=5, edgecolor='navy', linewidth=1)
            
            ax.set_xlabel('Month')
            ax.set_ylabel('Average Vehicle Count')
            ax.set_title('Monthly Traffic Patterns')
            ax.set_xticks(range(len(monthly_stats)))
            ax.set_xticklabels([months[i] for i in x_pos])
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    else:
        # If no monthly data, show data source comparison
        if 'data_source' in final_dataset.columns:
            source_stats = final_dataset.groupby('data_source')['total_vehicles'].agg(['mean', 'count'])
            ax.bar(source_stats.index, source_stats['mean'], color=['#1f77b4', '#ff7f0e'], alpha=0.8)
            ax.set_xlabel('Data Source')
            ax.set_ylabel('Average Vehicle Count')
            ax.set_title('Average Traffic by Data Source')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for i, (idx, row) in enumerate(source_stats.iterrows()):
                ax.text(i, row['mean'], f"n={row['count']:,}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # ========================================
    # STATISTICAL INSIGHTS SUMMARY
    # ========================================
    
    print("\n" + "="*80)
    print("VISUAL SUMMARY STATISTICS INSIGHTS")
    print("="*80)
    
    # Calculate and display key insights
    if 'hour_of_day' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        hourly_stats = final_dataset.groupby('hour_of_day')['total_vehicles'].mean()
        peak_hour = hourly_stats.idxmax()
        low_hour = hourly_stats.idxmin()
        
        print(f"\n🚦 TRAFFIC PATTERNS:")
        print(f"   - Peak traffic hour: {peak_hour}:00 ({hourly_stats[peak_hour]:.0f} vehicles)")
        print(f"   - Lowest traffic hour: {low_hour}:00 ({hourly_stats[low_hour]:.0f} vehicles)")
        print(f"   - Peak to low ratio: {hourly_stats[peak_hour]/hourly_stats[low_hour]:.1f}x")
    
    if 'is_weekend' in final_dataset.columns and 'total_vehicles' in final_dataset.columns:
        weekend_avg = final_dataset[final_dataset['is_weekend'] == True]['total_vehicles'].mean()
        weekday_avg = final_dataset[final_dataset['is_weekend'] == False]['total_vehicles'].mean()
        
        print(f"\n📅 WEEKLY PATTERNS:")
        print(f"   - Weekday average: {weekday_avg:.0f} vehicles/hour")
        print(f"   - Weekend average: {weekend_avg:.0f} vehicles/hour")
        print(f"   - Weekend reduction: {((weekday_avg - weekend_avg)/weekday_avg)*100:.1f}%")
    
    if all(col in df_bol.columns for col in vehicle_types_bol):
        heavy_ratio_bol = df_bol['Heavy_Count'].sum() / df_bol[vehicle_types_bol].sum().sum()
        print(f"\n🚛 VEHICLE TYPE INSIGHTS:")
        print(f"   - Bologna heavy vehicle ratio: {heavy_ratio_bol*100:.1f}%")
    
    if all(col in df_regional.columns for col in vehicle_types_reg):
        heavy_ratio_reg = df_regional['Heavy'].sum() / df_regional[vehicle_types_reg].sum().sum()
        print(f"   - Regional heavy vehicle ratio: {heavy_ratio_reg*100:.1f}%")
    
    print("\n" + "="*80)
