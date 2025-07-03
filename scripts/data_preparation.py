import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import gc  



# 1. PREPARE AND COMBINE TRAFFIC DATA
def prepare_traffic_data(df_bol, df_regional, bologna_meta, regional_meta):
    """Prepare and standardize traffic data from both sources"""
    
    # Bologna data preparation
    df_bol_geo = df_bol.merge(
        bologna_meta[['codice', 'geometry']], 
        left_on='id_uni', 
        right_on='codice', 
        how='left'
    )
    
    df_bol_geo['station_id'] = df_bol_geo['id_uni']
    df_bol_geo['total_vehicles'] = (
        df_bol_geo['Light_Count'] + 
        df_bol_geo['Medium_Count'] + 
        df_bol_geo['Heavy_Count']
    )
    df_bol_geo['data_source'] = 'bologna'
    
    # Regional data preparation
    df_regional_geo = df_regional.merge(
        regional_meta[['NAME', 'geometry']], 
        left_on='MTSStationID', 
        right_on='NAME', 
        how='left'
    )
    
    df_regional_geo['station_id'] = df_regional_geo['MTSStationID']
    df_regional_geo['Light_Count'] = df_regional_geo['Light']
    df_regional_geo['Medium_Count'] = df_regional_geo['Medium']
    df_regional_geo['Heavy_Count'] = df_regional_geo['Heavy']
    df_regional_geo['total_vehicles'] = (
        df_regional_geo['Light'] + 
        df_regional_geo['Medium'] + 
        df_regional_geo['Heavy'] + 
        df_regional_geo.get('Others', 0)
    )
    df_regional_geo['data_source'] = 'regional'
    df_regional_geo['datetime'] = pd.to_datetime(df_regional_geo['datetime'])
    
    # Combine with common columns
    common_cols = [
        'datetime', 'station_id', 'Light_Count', 'Medium_Count', 
        'Heavy_Count', 'total_vehicles', 'geometry', 'data_source'
    ]
    
    # Aggregate bidirectional traffic for Bologna data
    df_bol_agg = df_bol_geo.groupby(['datetime', 'station_id', 'data_source']).agg({
        'Light_Count': 'sum',
        'Medium_Count': 'sum',
        'Heavy_Count': 'sum',
        'total_vehicles': 'sum',
        'geometry': 'first'  
    }).reset_index()
    
    # Combine datasets
    combined_traffic = pd.concat([
        df_bol_agg[common_cols],
        df_regional_geo[common_cols]
    ], ignore_index=True)
    
    return combined_traffic

# 2. CREATE BUFFER ZONES
def create_buffer_zones(air_stations, zone_distances_km=[0.5, 1.5, 3.0]):
    """Create buffer zones around air quality monitoring stations"""
    
    # Convert to GeoDataFrame and project to UTM for accurate distances
    stations_gdf = gpd.GeoDataFrame(air_stations, geometry='geometry', crs='EPSG:4326')
    stations_gdf_utm = stations_gdf.to_crs('EPSG:32633')  # UTM zone 33N for Bologna
    
    # Create buffer zones for each station
    buffer_zones = []
    
    for idx, station in stations_gdf_utm.iterrows():
        station_name = station['station']
        station_geom = station['geometry']
        
        # Create buffers for each zone
        prev_buffer = None
        for i, distance_km in enumerate(zone_distances_km):
            distance_m = distance_km * 1000
            
            # Create buffer
            buffer = station_geom.buffer(distance_m)
            
            # Create zone (ring) by subtracting previous buffer
            if prev_buffer is not None:
                zone = buffer.difference(prev_buffer)
            else:
                zone = buffer
            
            buffer_zones.append({
                'station_name': station_name,
                'zone_id': i + 1,
                'zone_name': f'zone_{i+1}',
                'inner_radius_km': 0 if i == 0 else zone_distances_km[i-1],
                'outer_radius_km': distance_km,
                'geometry': zone,
                'zone_weight': 1.0 / (i + 1)  # Decreasing weight with distance
            })
            
            prev_buffer = buffer
    
    # Convert back to WGS84 for consistency
    zones_gdf = gpd.GeoDataFrame(buffer_zones, crs='EPSG:32633')
    zones_gdf = zones_gdf.to_crs('EPSG:4326')
    
    return zones_gdf

# # 3. ASSIGN TRAFFIC TO ZONES
# def assign_traffic_to_zones(traffic_data, buffer_zones):
#     """Assign traffic monitoring points to buffer zones"""
    
#     # Convert traffic data to GeoDataFrame
#     traffic_gdf = gpd.GeoDataFrame(
#         traffic_data, 
#         geometry='geometry', 
#         crs='EPSG:4326'
#     )
    
#     # Spatial join to find which zone each traffic point belongs to
#     traffic_with_zones = gpd.sjoin(
#         traffic_gdf,
#         buffer_zones[['station_name', 'zone_id', 'zone_name', 'zone_weight', 'geometry']],
#         how='left',
#         predicate='within'
#     )
    
#     # Handle traffic points that might be in multiple zones (overlap)
#     # Keep the closest zone (highest weight)
#     traffic_with_zones = traffic_with_zones.sort_values('zone_weight', ascending=False)
#     traffic_with_zones = traffic_with_zones.drop_duplicates(
#         subset=['datetime', 'station_id'], 
#         keep='first'
#     )
    
#     return traffic_with_zones


# 3. BATCH PROCESSING FOR SPATIAL JOINS
def assign_traffic_to_zones(traffic_data, buffer_zones, batch_size=50000):
    """Assign traffic to zones in batches to save memory"""
    
    traffic_gdf = gpd.GeoDataFrame(traffic_data, geometry='geometry', crs='EPSG:4326')
    
    # Process in batches
    results = []
    n_batches = len(traffic_gdf) // batch_size + 1
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(traffic_gdf))
        
        batch = traffic_gdf.iloc[start_idx:end_idx]
        
        # Find which zone each point belongs to
        batch_result = gpd.sjoin(batch, buffer_zones, how='left', predicate='within')
        
        # Keep only the closest zone (smallest zone_id)
        batch_result = batch_result.sort_values('zone_id').drop_duplicates(
            subset=['datetime', 'station_id'], keep='first'
        )
        
        results.append(batch_result)
        
        # Clean up
        del batch, batch_result
        gc.collect()
    
    return pd.concat(results, ignore_index=True)


# 4. CALCULATE WEIGHTED TRAFFIC METRICS
def calculate_weighted_traffic(traffic_with_zones):
    """Calculate weighted traffic metrics for each air quality station"""
    
    # Apply weights based on zone and vehicle type
    vehicle_weights = {
        'Light': 1.0,
        'Medium': 2.5,  # Medium vehicles produce ~2.5x pollution
        'Heavy': 4.0    # Heavy vehicles produce ~4x pollution
    }
    
    # Calculate weighted vehicle counts
    traffic_with_zones['weighted_light'] = (
        traffic_with_zones['Light_Count'] * 
        traffic_with_zones['zone_weight'] * 
        vehicle_weights['Light']
    )
    traffic_with_zones['weighted_medium'] = (
        traffic_with_zones['Medium_Count'] * 
        traffic_with_zones['zone_weight'] * 
        vehicle_weights['Medium']
    )
    traffic_with_zones['weighted_heavy'] = (
        traffic_with_zones['Heavy_Count'] * 
        traffic_with_zones['zone_weight'] * 
        vehicle_weights['Heavy']
    )
    
    # Total weighted traffic impact
    traffic_with_zones['weighted_total'] = (
        traffic_with_zones['weighted_light'] + 
        traffic_with_zones['weighted_medium'] + 
        traffic_with_zones['weighted_heavy']
    )
    
    # Add source weight (local data might be more accurate)
    source_weights = {'bologna': 1.2, 'regional': 1.0}
    traffic_with_zones['final_weight'] = (
        traffic_with_zones['weighted_total'] * 
        traffic_with_zones['data_source'].map(source_weights)
    )
    
    return traffic_with_zones

# 5. AGGREGATE TO STATION LEVEL
def aggregate_to_stations(weighted_traffic):
    """Aggregate weighted traffic to air quality station level"""
    
    # Group by station and time
    station_traffic = weighted_traffic.groupby(['datetime', 'station_name']).agg({
        # Raw counts by zone
        'Light_Count': 'sum',
        'Medium_Count': 'sum',
        'Heavy_Count': 'sum',
        'total_vehicles': 'sum',
        
        # Weighted metrics
        'weighted_light': 'sum',
        'weighted_medium': 'sum',
        'weighted_heavy': 'sum',
        'weighted_total': 'sum',
        'final_weight': 'sum',
        
        # Zone distribution
        'zone_id': lambda x: list(x.value_counts().to_dict().items()),
        'data_source': lambda x: list(x.value_counts().to_dict().items())
    }).reset_index()
    
    # Calculate additional metrics
    station_traffic['avg_vehicle_weight'] = (
        station_traffic['final_weight'] / station_traffic['total_vehicles']
    ).fillna(0)
    
    station_traffic['heavy_vehicle_impact_ratio'] = (
        station_traffic['weighted_heavy'] / station_traffic['weighted_total']
    ).fillna(0)
    
    # Create hourly timestamp
    station_traffic['hour'] = pd.to_datetime(station_traffic['datetime']).dt.floor('H')
    
    return station_traffic
# 6. FIXED MERGE WITH AIR QUALITY DATA
def merge_with_air_quality(station_traffic, df_no2, df_pm10, df_pm25):
    """Merge traffic data with air quality measurements handling different temporal resolutions"""
    
    print("\n--- Debugging merge process ---")
    
    # First, let's check what station names we have
    print(f"Traffic stations: {sorted(station_traffic['station_name'].unique())}")
    
    # Process NO2 data (HOURLY)
    def process_no2_data(df):
        # Get station columns (exclude 'data' and validation columns)
        station_cols = [col for col in df.columns 
                       if col != 'data' and not col.startswith('v_')]
        
        print(f"NO2 stations available: {station_cols}")
        
        # Melt to long format
        melted = df.melt(
            id_vars=['data'],
            value_vars=station_cols,
            var_name='station_name',
            value_name='NO2'
        )
        
        melted['datetime'] = pd.to_datetime(melted['data'])
        melted['hour'] = melted['datetime'].dt.floor('H')
        
        
        return melted[['hour', 'station_name', 'NO2']]
    
    # Process each pollutant
    no2_hourly = process_no2_data(df_no2)
    
    print(f"\nProcessed data shapes:")
    print(f"NO2: {no2_hourly.shape}")
    
    # Check for station name mismatches
    traffic_stations = set(station_traffic['station_name'].unique())
    no2_stations = set(no2_hourly['station_name'].unique())
    
    print(f"\nStation overlap:")
    print(f"Traffic ∩ NO2: {traffic_stations.intersection(no2_stations)}")

    # Start with traffic data
    final_dataset = station_traffic.copy()
    
    # Merge with NO2 (hourly data)
    if len(traffic_stations.intersection(no2_stations)) > 0:
        final_dataset = final_dataset.merge(
            no2_hourly,
            on=['hour', 'station_name'],
            how='left'
        )
        print(f"\nAfter NO2 merge: {final_dataset.shape}")
    else:
        print("\nWARNING: No matching stations between traffic and NO2 data!")
        final_dataset['NO2'] = np.nan
    
    
    # If no air quality data was matched, keep traffic data anyway
    if final_dataset.empty:
        print("\nERROR: Dataset became empty after merging!")
        print("Keeping traffic data without air quality measurements...")
        final_dataset = station_traffic.copy()
        final_dataset['NO2'] = np.nan
    
    # Report data availability
    print(f"\nFinal dataset shape: {final_dataset.shape}")
    print(f"Records with NO2 data: {final_dataset['NO2'].notna().sum() if 'NO2' in final_dataset else 0}")
    
    
    return final_dataset

# 7. FIXED TEMPORAL FEATURES AND LAGS
def add_temporal_features(df):
    """Add temporal features and lagged variables for time series analysis"""
    
    if df.empty:
        print("WARNING: Empty dataframe received in add_temporal_features")
        return df
    
    df = df.sort_values(['station_name', 'hour']).copy()
    
    # Temporal features
    df['hour_of_day'] = df['hour'].dt.hour
    df['day_of_week'] = df['hour'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19])
    df['month'] = df['hour'].dt.month
    df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'])
    
    # Only add lags if we have enough data
    if len(df) > 24:  # At least one day of data
        # Lagged features for hourly data
        lag_hours = [1, 2, 3, 6, 12, 24]
        
        for lag in lag_hours:
            # Traffic lags
            df[f'traffic_lag_{lag}h'] = df.groupby('station_name')['weighted_total'].shift(lag)
            
            # Only lag NO2 since it's hourly
            if 'NO2' in df.columns:
                df[f'NO2_lag_{lag}h'] = df.groupby('station_name')['NO2'].shift(lag)
        
        # Rolling averages
        window_sizes = [3, 6, 12, 24]  # hours
        
        for window in window_sizes:
            # Traffic rolling means
            df[f'traffic_rolling_mean_{window}h'] = (
                df.groupby('station_name')['weighted_total']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Only for NO2 (hourly data)
            if 'NO2' in df.columns:
                df[f'NO2_rolling_mean_{window}h'] = (
                    df.groupby('station_name')['NO2']
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
    
    return df


# 8. DATA QUALITY AND FILTERING
def apply_data_quality_filters(df):
    """Apply data quality filters and handle outliers"""
    
    # Remove negative values
    for col in ['NO2', 'PM10', 'PM2.5', 'total_vehicles', 'weighted_total']:
        if col in df.columns:
            df = df[df[col] >= 0]
    
    # Remove extreme outliers using IQR method
    def remove_outliers_iqr(df, column, multiplier=3):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    # Apply outlier removal for key columns
    for col in ['NO2', 'PM10', 'PM2.5']:
        if col in df.columns:
            df = remove_outliers_iqr(df, col)
    
    return df

# 9. CREATE SUMMARY STATISTICS
def create_zone_summary(traffic_with_zones, buffer_zones):
    """Create summary statistics for each zone"""
    
    zone_summary = traffic_with_zones.groupby(['station_name', 'zone_id', 'zone_name']).agg({
        'total_vehicles': ['sum', 'mean', 'std'],
        'weighted_total': ['sum', 'mean'],
        'data_source': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Flatten column names
    zone_summary.columns = ['_'.join(col).strip('_') for col in zone_summary.columns.values]
    
    # Add zone information
    zone_summary = zone_summary.merge(
        buffer_zones[['station_name', 'zone_id', 'inner_radius_km', 'outer_radius_km']],
        on=['station_name', 'zone_id'],
        how='left'
    )
    
    return zone_summary

# 10. MAIN EXECUTION FUNCTION
def process_traffic_air_quality_data(
    df_bol, df_regional, bologna_meta, regional_meta,
    stations_df, df_no2, df_pm10, df_pm25,
    zone_distances_km=[0.5, 1.5, 3.0], bologna_gdf=None
):
    """Main function to process all data using buffer zone approach"""
    
    print("Step 1: Preparing traffic data...")
    combined_traffic = prepare_traffic_data(df_bol, df_regional, bologna_meta, regional_meta)
    print(f"  - Combined traffic records: {len(combined_traffic)}")
    
    print("\nStep 2: Creating buffer zones...")
    buffer_zones = create_buffer_zones(stations_df, zone_distances_km)
    print(f"  - Total zones created: {len(buffer_zones)}")
    print(f"  - Zones per station: {len(zone_distances_km)}")
    
    print("\nStep 3: Assigning traffic to zones...")
    traffic_with_zones = assign_traffic_to_zones(combined_traffic, buffer_zones)
    
    # Report unassigned traffic points
    unassigned = traffic_with_zones[traffic_with_zones['station_name'].isna()]
    if len(unassigned) > 0:
        traffic_with_zones = traffic_with_zones.dropna(subset=['station_name'])
    
    print(f"  - Assigned traffic records: {len(traffic_with_zones)}")
    
    print("\nStep 4: Calculating weighted traffic...")
    weighted_traffic = calculate_weighted_traffic(traffic_with_zones)
    
    print("\nStep 5: Aggregating to station level...")
    station_traffic = aggregate_to_stations(weighted_traffic)
    print(f"  - Station-hour records: {len(station_traffic)}")
    

    print("\nStep 6: Merging with air quality data...")
    final_dataset = merge_with_air_quality(station_traffic, df_no2, df_pm10, df_pm25)
    print(f"  - Merged records: {len(final_dataset)}")
    
    print(final_dataset.shape)
    print(final_dataset.head())
    
    print("\nStep 7: Adding temporal features...")
    final_dataset = add_temporal_features(final_dataset)
    
    print("\nStep 8: Applying data quality filters...")
    initial_len = len(final_dataset)
    final_dataset = apply_data_quality_filters(final_dataset)
    print(f"  - Records removed: {initial_len - len(final_dataset)}")
    
    print("\nStep 9: Creating zone summary...")
    zone_summary = create_zone_summary(traffic_with_zones, buffer_zones)

    print("\nStep 10: Finalizing results...")
    plot_buffer_zones(buffer_zones, stations_df, bologna_meta, regional_meta, bologna_gdf)

    print("\n✓ Processing complete!")
    
    return {
        'final_dataset': final_dataset,
        'zone_summary': zone_summary,
        'buffer_zones': buffer_zones,
        'traffic_with_zones': traffic_with_zones
    }


# 11. ENHANCED VISUALIZATION HELPER FUNCTIONS
def plot_buffer_zones(buffer_zones, stations_df, bologna_meta, regional_meta, 
                     boundary_gdf=None, save_path=None):
    """Visualize buffer zones, air quality stations, traffic sensors, and regional boundaries"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 25))
    
    # 1. Plot regional boundary if provided
    if boundary_gdf is not None:
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=2, 
                                  linestyle='--', label='Regional Boundary')
    
    # 2. Plot buffer zones with different colors
    colors = ['#ffcccc', '#ffd9b3', '#ffff99']  # Light red, orange, yellow
    
    for i, zone_id in enumerate(sorted(buffer_zones['zone_id'].unique())):
        zone_data = buffer_zones[buffer_zones['zone_id'] == zone_id]
        zone_data.plot(ax=ax, color=colors[i], alpha=0.4, edgecolor='gray', linewidth=0.5)
    
    # 3. Plot traffic sensors - Bologna
    if not bologna_meta.empty:
        bologna_gdf = gpd.GeoDataFrame(bologna_meta, geometry='geometry', crs='EPSG:4326')
        bologna_gdf.plot(ax=ax, color='blue', markersize=30, marker='o', 
                         alpha=0.7, edgecolor='darkblue', linewidth=1)
    
    # 4. Plot traffic sensors - Regional
    if not regional_meta.empty:
        regional_gdf = gpd.GeoDataFrame(regional_meta, geometry='geometry', crs='EPSG:4326')
        regional_gdf.plot(ax=ax, color='green', markersize=30, marker='^', 
                         alpha=0.7, edgecolor='darkgreen', linewidth=1)
    
    # 5. Plot air quality stations (on top)
    stations_gdf = gpd.GeoDataFrame(stations_df, geometry='geometry', crs='EPSG:4326')
    stations_gdf.plot(ax=ax, color='red', markersize=100, marker='*', 
                     edgecolor='darkred', linewidth=1.5)
    
    # 6. Add station labels
    for idx, row in stations_gdf.iterrows():
        ax.annotate(row['station'], 
                   xy=(row.geometry.x, row.geometry.y),
                   xytext=(5, 5), 
                   textcoords='offset points', 
                   fontsize=9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 7. Create custom legend
    legend_elements = [
        # Air quality station
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=15, label='Air Quality Stations', markeredgecolor='darkred'),
        
        # Traffic sensors
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='Bologna Traffic Sensors', markeredgecolor='darkblue'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', 
               markersize=10, label='Regional Traffic Sensors', markeredgecolor='darkgreen'),
        
        # Buffer zones
        mpatches.Patch(color='#ffcccc', label='Zone 1 (0-0.5 km)', alpha=0.4),
        mpatches.Patch(color='#ffd9b3', label='Zone 2 (0.5-1.5 km)', alpha=0.4),
        mpatches.Patch(color='#ffff99', label='Zone 3 (1.5-3.0 km)', alpha=0.4),
    ]
    
    # Add boundary to legend if present
    if boundary_gdf is not None:
        legend_elements.append(
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Regional Boundary')
        )
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
              frameon=True, fancybox=True, shadow=True)
    
    # 8. Set plot properties
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Air Quality Monitoring Stations and Traffic Sensors\nwith Buffer Zones', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Get bounds for better view
    minx, miny, maxx, maxy = buffer_zones.total_bounds
    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Additional function to create a summary map with sensor density
def plot_sensor_density_map(bologna_meta, regional_meta, stations_df, 
                          grid_size=0.01, save_path=None):
    """Create a heatmap showing sensor density"""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde
    import numpy as np
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Combine all sensor locations
    all_points = []
    
    if not bologna_meta.empty:
        bologna_points = [(p.x, p.y) for p in bologna_meta.geometry]
        all_points.extend(bologna_points)
    
    if not regional_meta.empty:
        regional_points = [(p.x, p.y) for p in regional_meta.geometry]
        all_points.extend(regional_points)
    
    if len(all_points) > 0:
        # Create density map
        x_coords = [p[0] for p in all_points]
        y_coords = [p[1] for p in all_points]
        
        # Create grid
        x_min, x_max = min(x_coords) - 0.05, max(x_coords) + 0.05
        y_min, y_max = min(y_coords) - 0.05, max(y_coords) + 0.05
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size),
                            np.arange(y_min, y_max, grid_size))
        
        # Calculate density
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([x_coords, y_coords])
        kernel = gaussian_kde(values)
        density = np.reshape(kernel(positions).T, xx.shape)
        
        # Plot density
        im = ax.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.6)
        plt.colorbar(im, ax=ax, label='Sensor Density')
    
    # Plot individual sensors
    if not bologna_meta.empty:
        bologna_gdf = gpd.GeoDataFrame(bologna_meta, geometry='geometry', crs='EPSG:4326')
        bologna_gdf.plot(ax=ax, color='blue', markersize=20, marker='o', 
                         alpha=0.8, label='Bologna Sensors')
    
    if not regional_meta.empty:
        regional_gdf = gpd.GeoDataFrame(regional_meta, geometry='geometry', crs='EPSG:4326')
        regional_gdf.plot(ax=ax, color='green', markersize=20, marker='^', 
                         alpha=0.8, label='Regional Sensors')
    
    # Plot air quality stations
    stations_gdf = gpd.GeoDataFrame(stations_df, geometry='geometry', crs='EPSG:4326')
    stations_gdf.plot(ax=ax, color='red', markersize=100, marker='*', 
                     label='Air Quality Stations', edgecolor='darkred', linewidth=1.5)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Traffic Sensor Density Map with Air Quality Stations', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
