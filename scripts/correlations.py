import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def analyze_traffic_air_quality_relationships(final_dataset):
    """
    Comprehensive analysis of traffic-air quality relationships
    """
    
    # 5.1.2 Correlation Heatmaps
    def create_correlation_heatmaps(df):
        """Create various correlation heatmaps"""
        
        # Prepare data
        traffic_vars = ['Light_Count', 'Medium_Count', 'Heavy_Count', 'total_vehicles', 
                       'weighted_total', 'heavy_vehicle_impact_ratio']
        pollutant_vars = ['NO2']  # Add PM10, PM2.5 if available
        
        # 1. Pearson and Spearman Correlations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Select variables
        vars_to_correlate = traffic_vars + pollutant_vars
        correlation_data = df[vars_to_correlate].dropna()
        
        # Pearson correlation
        pearson_corr = correlation_data.corr(method='pearson')
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax1)
        ax1.set_title('Pearson Correlation Matrix', fontsize=14)
        
        # Spearman correlation
        spearman_corr = correlation_data.corr(method='spearman')
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=True, fmt='.3f', 
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=0.5, ax=ax2)
        ax2.set_title('Spearman Correlation Matrix', fontsize=14)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Lagged Correlations (1-24 hours)
        create_lagged_correlation_heatmap(df)
        
    
    def create_lagged_correlation_heatmap(df):
        """Create heatmap of lagged correlations"""
        
        # Define lags to analyze
        lags = [1, 2, 3, 6, 12, 24]
        
        # Initialize correlation matrix
        traffic_var = 'weighted_total'  # Main traffic variable
        pollutant = 'NO2'
        
        if pollutant not in df.columns:
            print(f"Warning: {pollutant} not found in dataset")
            return
            
        corr_results = pd.DataFrame(index=lags, columns=['Pearson', 'Spearman', 'Sample Size'])
        
        # Calculate correlations for each lag
        for lag in lags:
            lag_col = f'traffic_lag_{lag}h'
            if lag_col in df.columns:
                valid_data = df[[lag_col, pollutant]].dropna()
                if len(valid_data) > 30:  # Minimum sample size
                    pearson_corr, pearson_p = pearsonr(valid_data[lag_col], valid_data[pollutant])
                    spearman_corr, spearman_p = spearmanr(valid_data[lag_col], valid_data[pollutant])
                    
                    corr_results.loc[lag, 'Pearson'] = pearson_corr
                    corr_results.loc[lag, 'Spearman'] = spearman_corr
                    corr_results.loc[lag, 'Sample Size'] = len(valid_data)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot correlation values
        corr_results[['Pearson', 'Spearman']].plot(kind='bar', ax=ax1)
        ax1.set_xlabel('Lag (hours)', fontsize=12)
        ax1.set_ylabel('Correlation Coefficient', fontsize=12)
        ax1.set_title(f'Lagged Correlations: Traffic (t-lag) vs {pollutant} (t)', fontsize=14)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.legend(['Pearson', 'Spearman'])
        ax1.grid(True, alpha=0.3)
        
        # Create comprehensive lag matrix
        lag_matrix = pd.DataFrame(index=[f'Traffic_lag_{i}h' for i in range(1, 25)],
                                 columns=[f'{pollutant}_lag_{i}h' for i in range(0, 13)])
        
        for traffic_lag in range(1, 25):
            traffic_col = f'traffic_lag_{traffic_lag}h'
            if traffic_col not in df.columns:
                continue
                
            for no2_lag in range(0, 13):
                if no2_lag == 0:
                    no2_col = pollutant
                else:
                    no2_col = f'{pollutant}_lag_{no2_lag}h'
                
                if no2_col in df.columns:
                    valid_data = df[[traffic_col, no2_col]].dropna()
                    if len(valid_data) > 30:
                        corr, _ = pearsonr(valid_data[traffic_col], valid_data[no2_col])
                        lag_matrix.loc[f'Traffic_lag_{traffic_lag}h', f'{pollutant}_lag_{no2_lag}h'] = corr
        
        # Convert to numeric and plot
        lag_matrix = lag_matrix.astype(float)
        
        sns.heatmap(lag_matrix, cmap='coolwarm', center=0, vmin=-0.5, vmax=0.5,
                   cbar_kws={'label': 'Correlation Coefficient'}, ax=ax2)
        ax2.set_title(f'Cross-Lagged Correlation Matrix: Traffic vs {pollutant}', fontsize=14)
        ax2.set_xlabel(f'{pollutant} Lag', fontsize=12)
        ax2.set_ylabel('Traffic Lag', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    
    # 5.1.3 Advanced Correlation Analysis
    def analyze_vehicle_type_impact(df):
        """Analyze the specific impact of different vehicle types"""
        
        pollutant = 'NO2'
        if pollutant not in df.columns:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Relative contribution analysis
        ax = axes[0, 0]
        vehicle_counts = df[['Light_Count', 'Medium_Count', 'Heavy_Count']].sum()
        vehicle_correlations = pd.Series({
            'Light': df['Light_Count'].corr(df[pollutant]),
            'Medium': df['Medium_Count'].corr(df[pollutant]),
            'Heavy': df['Heavy_Count'].corr(df[pollutant])
        })
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        vehicle_counts.plot(kind='bar', ax=ax, alpha=0.7, color='lightblue')
        vehicle_correlations.plot(kind='line', ax=ax2, color='red', marker='o', linewidth=2)
        
        ax.set_xlabel('Vehicle Type', fontsize=12)
        ax.set_ylabel('Total Count', fontsize=12, color='blue')
        ax2.set_ylabel(f'Correlation with {pollutant}', fontsize=12, color='red')
        ax.set_title('Vehicle Count vs Pollution Correlation by Type', fontsize=14)
        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 2. Heavy vehicle percentage impact
        ax = axes[0, 1]
        df['heavy_percentage'] = (df['Heavy_Count'] / df['total_vehicles'] * 100).replace([np.inf, -np.inf], 0)
        
        # Bin heavy percentage
        bins = [0, 5, 10, 15, 20, 100]
        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
        df['heavy_percentage_bin'] = pd.cut(df['heavy_percentage'], bins=bins, labels=labels)
        
        # Calculate mean pollution by bin
        pollution_by_heavy = df.groupby('heavy_percentage_bin')[pollutant].agg(['mean', 'std', 'count'])
        
        pollution_by_heavy['mean'].plot(kind='bar', ax=ax, yerr=pollution_by_heavy['std'], 
                                       capsize=5, color='orange', alpha=0.7)
        ax.set_xlabel('Heavy Vehicle Percentage', fontsize=12)
        ax.set_ylabel(f'Mean {pollutant} (μg/m³)', fontsize=12)
        ax.set_title('Pollution Levels by Heavy Vehicle Percentage', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add sample sizes
        for i, (idx, row) in enumerate(pollution_by_heavy.iterrows()):
            ax.text(i, row['mean'] + row['std'] + 1, f'n={row["count"]}', 
                   ha='center', va='bottom', fontsize=9)
        
        # 3. Time-of-day vehicle mix analysis
        ax = axes[1, 0]
        
        # Calculate vehicle mix by hour
        vehicle_mix = df.groupby('hour_of_day')[['Light_Count', 'Medium_Count', 'Heavy_Count']].mean()
        vehicle_mix_pct = vehicle_mix.div(vehicle_mix.sum(axis=1), axis=0) * 100
        
        vehicle_mix_pct.plot(kind='area', ax=ax, alpha=0.7)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Vehicle Mix (%)', fontsize=12)
        ax.set_title('Vehicle Type Distribution by Hour', fontsize=14)
        ax.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 4. Weighted impact comparison
        ax = axes[1, 1]
        
        # Compare different weighting schemes
        correlations = pd.DataFrame({
            'Unweighted Total': [df['total_vehicles'].corr(df[pollutant])],
            'Weighted Total': [df['weighted_total'].corr(df[pollutant])],
            'Heavy Only': [df['Heavy_Count'].corr(df[pollutant])],
            'Heavy Impact Ratio': [df['heavy_vehicle_impact_ratio'].corr(df[pollutant])]
        }).T
        
        correlations.plot(kind='barh', ax=ax, color='green', alpha=0.7)
        ax.set_xlabel(f'Correlation with {pollutant}', fontsize=12)
        ax.set_ylabel('Traffic Metric', fontsize=12)
        ax.set_title('Comparison of Traffic Metrics', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add correlation values
        for i, v in enumerate(correlations.values):
            ax.text(v[0] + 0.01, i, f'{v[0]:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    # 5.1.4 Statistical Summary
    def create_statistical_summary(df):
        """Create comprehensive statistical summary of relationships"""
        
        traffic_vars = ['Light_Count', 'Medium_Count', 'Heavy_Count', 'total_vehicles', 
                       'weighted_total', 'heavy_vehicle_impact_ratio']
        pollutant = 'NO2'
        
        if pollutant not in df.columns:
            return
            
        # Create summary table
        summary_stats = pd.DataFrame(columns=['Pearson_r', 'Pearson_p', 'Spearman_r', 
                                             'Spearman_p', 'R²', 'RMSE'])
        
        for var in traffic_vars:
            valid_data = df[[var, pollutant]].dropna()
            
            if len(valid_data) > 30:
                # Correlations
                pearson_r, pearson_p = pearsonr(valid_data[var], valid_data[pollutant])
                spearman_r, spearman_p = spearmanr(valid_data[var], valid_data[pollutant])
                
                # Linear regression
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score, mean_squared_error
                
                X = valid_data[var].values.reshape(-1, 1)
                y = valid_data[pollutant].values
                
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                r2 = r2_score(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))
                
                summary_stats.loc[var] = [pearson_r, pearson_p, spearman_r, spearman_p, r2, rmse]
        
        # Display summary
        print("\n" + "="*80)
        print(f"STATISTICAL SUMMARY: Traffic Variables vs {pollutant}")
        print("="*80)
        print(summary_stats.round(4))
        
        # Identify strongest predictors
        print("\n" + "-"*50)
        print("STRONGEST PREDICTORS (by absolute Pearson correlation):")
        print("-"*50)
        
        sorted_vars = summary_stats['Pearson_r'].abs().sort_values(ascending=False)
        for i, (var, corr) in enumerate(sorted_vars.items(), 1):
            p_val = summary_stats.loc[var, 'Pearson_p']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"{i}. {var}: r = {summary_stats.loc[var, 'Pearson_r']:.3f} {significance}")
        
        print("\n*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
    
    
    print("\nCreating correlation heatmaps...")
    create_correlation_heatmaps(final_dataset)
    
    print("\nAnalyzing vehicle type impacts...")
    analyze_vehicle_type_impact(final_dataset)
    
    print("\nGenerating statistical summary...")
    create_statistical_summary(final_dataset)
    
    return final_dataset
