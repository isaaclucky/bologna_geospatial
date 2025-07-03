import numpy as np
import contextily as ctx
from shapely.geometry import Point, Polygon, LineString
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import warnings
import osmnx as ox
import pandas as pd
import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from folium.plugins import HeatMap, HeatMapWithTime
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

warnings.filterwarnings('ignore')

def analyze_traffic_by_zones(df_bol, bol_gdf, bologna_metadata, bol_zones, stations_df, df_no2):
    """
    Comprehensive zone-based analysis of traffic and air quality
    """
    
    # 1. ASSIGN TRAFFIC SENSORS TO ZONES
    print("1. Assigning traffic sensors to zones...")
    
    # Ensure CRS match
    if bologna_metadata.crs != bol_zones.crs:
        bologna_metadata = bologna_metadata.to_crs(bol_zones.crs)
    
    # Spatial join to assign sensors to zones
    sensors_in_zones = gpd.sjoin(
        bologna_metadata, 
        bol_zones[['nomezona', 'codzona', 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # 2. AGGREGATE TRAFFIC DATA BY ZONE
    print("\n2. Aggregating traffic data by zone...")
    
    traffic_with_zones = df_bol.merge(
        sensors_in_zones[['codice', 'nomezona', 'codzona']], 
        left_on='id_uni', 
        right_on='codice', 
        how='left'
    )
    
    # Remove rows without zone assignment
    traffic_with_zones = traffic_with_zones.dropna(subset=['nomezona'])
    
    # Aggregate by zone
    zone_traffic = traffic_with_zones.groupby(['nomezona', 'codzona']).agg({
        'total_vehicles': ['sum', 'mean', 'std'],
        'Heavy_Count': ['sum', 'mean'],
        'Light_Count': ['sum', 'mean'],
        'Medium_Count': ['sum', 'mean'],
        'id_uni': 'nunique'  # Number of sensors per zone
    }).reset_index()
    
    # Flatten column names properly
    zone_traffic.columns = ['nomezona', 'codzona', 'total_vehicles_sum', 'total_vehicles_mean', 
                       'total_vehicles_std', 'Heavy_Count_sum', 'Heavy_Count_mean',
                       'Light_Count_sum', 'Light_Count_mean', 'Medium_Count_sum',
                       'Medium_Count_mean', 'sensor_count']
      
    
    # Rename columns for clarity
    zone_traffic = zone_traffic.rename(columns={
        'nomezona': 'zone_name',
        'codzona': 'zone_code',
        'id_uni_nunique': 'sensor_count'
    })
    
    print(f"Zone traffic columns: {zone_traffic.columns.tolist()}")
    
    # 3. CALCULATE TRAFFIC DENSITY AND PATTERNS
    print("\n3. Calculating traffic density and patterns...")
    
    # Add zone area to calculate density
    zone_traffic = zone_traffic.merge(
        bol_zones[['codzona', 'area']], 
        left_on='zone_code', 
        right_on='codzona',
        how='left'
    ).drop('codzona', axis=1)
    
    print(zone_traffic.head())
    # Calculate traffic density (vehicles per km²)
    zone_traffic['traffic_density'] = zone_traffic['total_vehicles_sum'] / (zone_traffic['area'] / 1_000_000)
    zone_traffic['heavy_vehicle_ratio'] = zone_traffic['Heavy_Count_sum'] / zone_traffic['total_vehicles_sum']
    zone_traffic['heavy_vehicle_ratio'] = zone_traffic['heavy_vehicle_ratio'].fillna(0)
    
    # 4. ASSIGN AIR QUALITY STATIONS TO ZONES
    print("\n4. Linking air quality data to zones...")
    
    # Convert stations to GeoDataFrame if needed
    if stations_df.crs != bol_zones.crs:
        stations_df = stations_df.to_crs(bol_zones.crs)
    
    # Assign air quality stations to zones
    stations_in_zones = gpd.sjoin(
        stations_df, 
        bol_zones[['nomezona', 'codzona', 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    # 5. AGGREGATE AIR QUALITY BY ZONE
    # Get valid station columns from df_no2
    valid_stations = [col for col in df_no2.columns if col in stations_df['station'].values]
    
    if valid_stations:
        # Melt NO2 data for easier processing
        no2_melted = df_no2.melt(
            id_vars=['data'], 
            value_vars=valid_stations,
            var_name='station', 
            value_name='NO2'
        )
        
        # Remove NaN values
        no2_melted = no2_melted.dropna(subset=['NO2'])
        
        # Add zone info to NO2 data
        no2_with_zones = no2_melted.merge(
            stations_in_zones[['station', 'nomezona', 'codzona']], 
            on='station', 
            how='left'
        )
        
        # Aggregate NO2 by zone
        zone_air_quality = no2_with_zones.groupby(['nomezona', 'codzona']).agg({
            'NO2': ['mean', 'std', 'max', 'min'],
            'station': 'nunique'
        }).reset_index()
        
        zone_air_quality.rename(columns={
        """(_'_N_O_2_'_,_ _'_m_e_a_n_'_)""": 'NO2_mean',
        """(_'_N_O_2_'_,_ _'_s_t_d_'_)""": 'NO2_std',
        """(_'_N_O_2_'_,_ _'_m_a_x_'_)""": 'NO2_max',
        """(_'_N_O_2_'_,_ _'_m_i_n_'_)""": 'NO2_min',
        """(_'_s_t_a_t_i_o_n_'_,_ _'_n_u_n_i_q_u_e_'_)""": 'station_nunique'
        }, inplace=True)  
        zone_air_quality.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0]
                                    for col in zone_air_quality.columns
                                    ]
        zone_air_quality = zone_air_quality.rename(columns={
            'nomezona': 'zone_name',
            'codzona': 'zone_code',
            'station_nunique': 'aq_station_count'
        })
    else:
        print("Warning: No matching air quality stations found in NO2 data")
        zone_air_quality = pd.DataFrame()
    
    # 6. MERGE TRAFFIC AND AIR QUALITY DATA
    print("\n5. Merging traffic and air quality data...")
    
    if not zone_air_quality.empty:
        zone_analysis = zone_traffic.merge(
            zone_air_quality, 
            on=['zone_name', 'zone_code'],
            how='outer'
        )
    else:
        zone_analysis = zone_traffic.copy()
    
    # 7. CREATE VISUALIZATION MAP
    print("\n6. Creating zone analysis map...")
    
    # Prepare data for mapping
    zones_for_map = bol_zones.merge(
        zone_analysis, 
        left_on='codzona', 
        right_on='zone_code',
        how='left'
    )
    
    # Create interactive map
    m = zones_for_map.explore(
        column='traffic_density',
        tooltip=['nomezona', 'traffic_density', 'heavy_vehicle_ratio'],
        popup=True,
        cmap='YlOrRd',
        scheme='quantiles',
        k=5,
        legend=True,
        legend_kwds={'title': 'Traffic Density (vehicles/km²)'},
        style_kwds={'fillOpacity': 0.7, 'weight': 2},
        tiles='CartoDB positron',
        name='Traffic Density by Zone'
    )
    
    # Add sensor points
    bologna_metadata.explore(
        m=m,
        color='blue',
        marker_kwds={'radius': 4},
        tooltip=['codice'],
        name='Traffic Sensors'
    )
    
    # Add air quality stations
    stations_df.explore(
        m=m,
        color='green',
        marker_kwds={'radius': 6},
        tooltip=['station'],
        name='Air Quality Stations'
    )
    
    return zone_analysis, zones_for_map, m



def create_simple_traffic_flow_map(df_bol, bol_gdf, bologna_metadata):
    """
    Simpler version focusing on clarity
    """
    
    # Get unique flows (you might want to aggregate by time period)
    # For now, let's just take a sample or specific time
    sample_flows = df_bol.groupby(['id_uni', 'Node_da', 'Node_a', 'Forward']).agg({
        'datetime': 'count',
        'total_vehicles': 'sum',
        'Heavy_Count': 'sum',
        'direction': 'first',
        'angle': 'first'
    }).reset_index()
    
    # Create flow lines
    flows = []
    for idx, row in sample_flows.iterrows():
        start_node = bol_gdf[bol_gdf['codice'] == row['Node_da']]
        end_node = bol_gdf[bol_gdf['codice'] == row['Node_a']]
        
        if len(start_node) > 0 and len(end_node) > 0:
            if row['Forward'] == 1:
                line = LineString([start_node.geometry.iloc[0], end_node.geometry.iloc[0]])
            else:
                line = LineString([end_node.geometry.iloc[0], start_node.geometry.iloc[0]])
            
            flows.append({
                'geometry': line,
                'sensor': row['id_uni'],
                'vehicles': row['total_vehicles'],
                'datetime': row['datetime'],
                'from': row['Node_da'] if row['Forward'] == 1 else row['Node_a'],
                'to': row['Node_a'] if row['Forward'] == 1 else row['Node_da']
            })
    flow_gdf = gpd.GeoDataFrame(flows, crs=bol_gdf.crs, geometry='geometry')
    
    # Create map
    m = bologna_metadata.explore(
        tooltip=['codice'],
        marker_kwds={'radius': 8, 'color': 'red', 'fill': True},
        style_kwds={'fillOpacity': 0.7},
        tiles='OpenStreetMap',
        name='Traffic Sensors'
    )
    
    # Add flows with thickness based on volume
    flow_gdf['line_width'] = 1 + (flow_gdf['vehicles'] / flow_gdf['vehicles'].max() * 5)
    
    flow_gdf.explore(
        m=m,
        color='blue',
        style_kwds={'opacity': 0.6},
        tooltip=['sensor', 'from', 'to', 'vehicles','datetime'],
        name='Traffic Flows'
    )
    
    return m

def create_zone_comparison_maps(zone_analysis, bol_zones, bologna_boundary, stations_df=None):
    """
    Create map visualizations for zone comparison analysis
    """
    # Create figure with multiple subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # Merge zone analysis with geographic data
    # Assuming zone_analysis has a 'zone_name' column that matches with bol_zones
    zones_with_data = bol_zones.merge(
        zone_analysis, 
        left_on='nomezona',  # adjust this to match your zone identifier column
        right_on='zone_name', 
        how='left'
    )
    
    # 1. Traffic Density Map
    ax1 = axes[0]
    bologna_boundary.plot(ax=ax1, color='lightgray', edgecolor='black', linewidth=2)
    
    # Plot traffic density
    zones_with_data.plot(
        column='traffic_density',
        ax=ax1,
        legend=True,
        cmap='YlOrRd',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        legend_kwds={'label': 'Traffic Density (vehicles/km²)', 'orientation': 'horizontal', 'pad': 0.05}
    )
    
    # Add zone labels with traffic density values
    for idx, row in zones_with_data.iterrows():
        if pd.notna(row.get('traffic_density', None)) and row.geometry:
            centroid = row.geometry.centroid
            ax1.annotate(
                f"{row.get('zone_name', 'Zone')[:10]}\n{row.get('traffic_density', 0):.0f}",
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    ax1.set_title('Traffic Density by Zone', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. Heavy Vehicle Ratio Map
    ax2 = axes[1]
    bologna_boundary.plot(ax=ax2, color='lightgray', edgecolor='black', linewidth=2)
    
    zones_with_data.plot(
        column='heavy_vehicle_ratio',
        ax=ax2,
        legend=True,
        cmap='RdPu',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        legend_kwds={'label': 'Heavy Vehicle Ratio', 'orientation': 'horizontal', 'pad': 0.05}
    )
    
    # Add labels
    for idx, row in zones_with_data.iterrows():
        if pd.notna(row.get('heavy_vehicle_ratio', None)) and row.geometry:
            centroid = row.geometry.centroid
            ax2.annotate(
                f"{row.get('zone_name', 'Zone')[:10]}\n{row.get('heavy_vehicle_ratio', 0):.2%}",
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    ax2.set_title('Heavy Vehicle Ratio by Zone', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Sensor Distribution Map
    ax3 = axes[2]
    bologna_boundary.plot(ax=ax3, color='lightgray', edgecolor='black', linewidth=2)
    
    # Create a custom colormap for sensor count
    zones_with_data['sensor_count_normalized'] = zones_with_data['sensor_count'].fillna(0)
    
    zones_with_data.plot(
        column='sensor_count_normalized',
        ax=ax3,
        legend=True,
        cmap='Blues',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        legend_kwds={'label': 'Number of Sensors', 'orientation': 'horizontal', 'pad': 0.05}
    )
    
    # Add sensor count labels
    for idx, row in zones_with_data.iterrows():
        if pd.notna(row.get('sensor_count', None)) and row.geometry:
            centroid = row.geometry.centroid
            ax3.annotate(
                f"{row.get('zone_name', 'Zone')[:10]}\n{int(row.get('sensor_count', 0))} sensors",
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    ax3.set_title('Traffic Sensor Distribution by Zone', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Combined Metric Map (Traffic Density vs Air Quality if available)
    ax4 = axes[3]
    bologna_boundary.plot(ax=ax4, color='lightgray', edgecolor='black', linewidth=2)
    
    # Check if NO2 data exists
    if 'NO2_mean' in zones_with_data.columns and zones_with_data['NO2_mean'].notna().any():
        # Create a composite score
        zones_with_data['composite_score'] = (
            zones_with_data['traffic_density'].fillna(0) * 0.5 + 
            zones_with_data['NO2_mean'].fillna(0) * 0.5
        )
        
        zones_with_data.plot(
            column='composite_score',
            ax=ax4,
            legend=True,
            cmap='Reds',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8,
            legend_kwds={'label': 'Traffic-Air Quality Index', 'orientation': 'horizontal', 'pad': 0.05}
        )
        
        # Add air quality stations if provided
        if stations_df is not None:
            stations_df.plot(
                ax=ax4,
                color='blue',
                markersize=100,
                marker='*',
                edgecolor='white',
                linewidth=2,
                alpha=0.9,
                label='Air Quality Stations'
            )
        
        ax4.set_title('Combined Traffic-Air Quality Impact', fontsize=14, fontweight='bold')
    else:
        # If no air quality data, show top traffic zones
        zones_with_data['is_top_traffic'] = zones_with_data['traffic_density'] >= zones_with_data['traffic_density'].quantile(0.75)
        
        # Plot all zones in light color
        zones_with_data.plot(
            ax=ax4,
            color='lightblue',
            edgecolor='black',
            linewidth=0.5,
            alpha=0.5
        )
        
        # Highlight top traffic zones
        top_zones = zones_with_data[zones_with_data['is_top_traffic']]
        top_zones.plot(
            ax=ax4,
            color='darkred',
            edgecolor='black',
            linewidth=1,
            alpha=0.8
        )
        
        # Add labels for top zones
        for idx, row in top_zones.iterrows():
            if row.geometry:
                centroid = row.geometry.centroid
                ax4.annotate(
                    f"{row.get('zone_name', 'Zone')[:10]}",
                    xy=(centroid.x, centroid.y),
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8)
                )
        
        # Add legend
        red_patch = mpatches.Patch(color='darkred', label='Top 25% Traffic Zones')
        blue_patch = mpatches.Patch(color='lightblue', label='Other Zones')
        ax4.legend(handles=[red_patch, blue_patch], loc='upper right')
        
        ax4.set_title('Top Traffic Density Zones', fontsize=14, fontweight='bold')
    
    ax4.axis('off')
    
    plt.tight_layout()
    return fig

# Additional function to create an interactive hover map
def create_interactive_zone_map(zone_analysis, bol_zones, bologna_boundary):
    """
    Create a single comprehensive map with all zone information
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    
    # Merge data
    zones_with_data = bol_zones.merge(
        zone_analysis, 
        left_on='nomezona',  # adjust based on your column names
        right_on='zone_name', 
        how='left'
    )
    
    # Plot boundary
    bologna_boundary.plot(ax=ax, color='white', edgecolor='black', linewidth=3)
    
    # Create size based on sensor count
    zones_with_data['marker_size'] = zones_with_data['sensor_count'].fillna(1) * 10
    
    # Plot zones with traffic density coloring
    zones_with_data.plot(
        column='traffic_density',
        ax=ax,
        legend=True,
        cmap='YlOrRd',
        edgecolor='black',
        linewidth=1,
        alpha=0.7,
        legend_kwds={
            'label': 'Traffic Density (vehicles/km²)', 
            'orientation': 'vertical',
            'shrink': 0.8
        }
    )
    
    # Add comprehensive labels for each zone
    for idx, row in zones_with_data.iterrows():
        if row.geometry and pd.notna(row.get('traffic_density', None)):
            centroid = row.geometry.centroid
            
            # Create info text
            info_text = f"{row.get('zone_name', 'Zone')[:15]}\n"
            info_text += f"Traffic: {row.get('traffic_density', 0):.0f} v/km²\n"
            info_text += f"Heavy: {row.get('heavy_vehicle_ratio', 0):.1%}\n"
            info_text += f"Sensors: {int(row.get('sensor_count', 0))}"
            
            if 'NO2_mean' in row and pd.notna(row['NO2_mean']):
                info_text += f"\nNO2: {row['NO2_mean']:.1f} μg/m³"
            
            # Determine text color based on traffic density
            text_color = 'white' if row.get('traffic_density', 0) > zones_with_data['traffic_density'].median() else 'black'
            
            ax.annotate(
                info_text,
                xy=(centroid.x, centroid.y),
                ha='center',
                va='center',
                fontsize=7,
                color=text_color,
                bbox=dict(
                    boxstyle='round,pad=0.4', 
                    facecolor='black' if text_color == 'white' else 'white', 
                    alpha=0.7,
                    edgecolor='gray'
                )
            )
    
    ax.set_title('Bologna Zone Analysis: Traffic and Environmental Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add info box
    info_text = "Zone metrics shown:\n• Traffic density (color)\n• Heavy vehicle ratio\n• Sensor count"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig


def create_base_map_with_stations(bologna_metadata, regional_metadata, stations_df, buffer_zones, bologna_boundary):
    """
    Create interactive base map showing all monitoring stations and buffer zones
    """
    import folium
    
    # Ensure all data is in the same CRS (WGS84)
    crs_wgs84 = 'EPSG:4326'
    
    # Convert to GeoDataFrame if needed and ensure correct CRS
    if not isinstance(stations_df, gpd.GeoDataFrame):
        if 'geometry' not in stations_df.columns:
            stations_df['geometry'] = [Point(lon, lat) for lon, lat in 
                                      zip(stations_df['longitude'], stations_df['latitude'])]
        stations_df = gpd.GeoDataFrame(stations_df, crs=crs_wgs84)
    else:
        stations_df = stations_df.to_crs(crs_wgs84)
    
    # Start with buffer zones as base map (if available)
    m = bologna_boundary.explore(
        color='lightgray',
        fill=True,
        fill_color='lightgray',
        fill_opacity=0,
        name='Bologna Boundary',
        tiles='CartoDB positron'
    )
    
    # Plot buffer zones first (so they appear under the points)
    if buffer_zones is not None and len(buffer_zones) > 0:
        # Ensure buffer_zones is in correct CRS
        if hasattr(buffer_zones, 'crs') and buffer_zones.crs is not None:
            buffer_zones = buffer_zones.to_crs(crs_wgs84)
        
        # Add zone labels to buffer zones
        buffer_zones = buffer_zones.copy()
        buffer_zones['zone_label'] = buffer_zones.apply(
            lambda x: f'Zone {x["zone_id"]} ({x["inner_radius_km"]}-{x["outer_radius_km"]} km)', 
            axis=1
        )
        
        # Define colors for each zone
        zone_colors = ['#ffcccc', '#ffeecc', '#ffffcc']  # Light red, orange, yellow
        
        # Create base map with buffer zones
        m = buffer_zones.explore(
            m=m,
            column='zone_id',
            categorical=True,
            cmap=zone_colors,
            style_kwds={
                'fillOpacity': 0.3,
                'weight': 1,
                'color': 'black'
            },
            tooltip=['station_name', 'zone_label'],
            legend_kwds={'caption': 'Buffer Zones'},
            tiles='CartoDB positron'
        )
    
    # If no buffer zones, create base map centered on stations
    if m is None:
        # Calculate center from stations_df
        center_lat = stations_df.geometry.y.mean()
        center_lon = stations_df.geometry.x.mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='CartoDB positron'
        )
    
    # Convert map to folium if it's a folium object from explore
    if hasattr(m, '_parent'):
        folium_map = m
    else:
        folium_map = m
    
    # Add Bologna traffic monitoring stations
    if bologna_metadata is not None and len(bologna_metadata) > 0:
        bologna_metadata = bologna_metadata.to_crs(crs_wgs84).copy()
        bologna_metadata['station_type'] = 'Bologna Traffic Station'
        bologna_metadata['label'] = bologna_metadata.apply(
            lambda x: f"Bologna Traffic - ID: {x.get('codice', 'N/A')}", axis=1
        )
        
        # Create a feature group for Bologna stations
        bologna_group = folium.FeatureGroup(name=f'Bologna Traffic Stations (n={len(bologna_metadata)})')
        
        for idx, row in bologna_metadata.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=6,
                popup=row['label'],
                tooltip=row['label'],
                color='darkblue',
                fill=True,
                fillColor='blue',
                fillOpacity=0.8,
                weight=1
            ).add_to(bologna_group)
        
        bologna_group.add_to(folium_map)
    
    # Add Regional traffic monitoring stations
    if regional_metadata is not None and len(regional_metadata) > 0:
        regional_metadata = regional_metadata.to_crs(crs_wgs84).copy()
        regional_metadata['station_type'] = 'Regional Traffic Station'
        regional_metadata['label'] = regional_metadata.apply(
            lambda x: f"Regional Traffic - ID: {x.get('NAME', 'N/A')}", axis=1
        )
        
        # Create a feature group for Regional stations
        regional_group = folium.FeatureGroup(name=f'Regional Traffic Stations (n={len(regional_metadata)})')
        
        for idx, row in regional_metadata.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                popup=row['label'],
                tooltip=row['label'],
                color='darkgreen',
                fill=True,
                fillColor='green',
                fillOpacity=0.8,
                weight=1
            ).add_to(regional_group)
        
        regional_group.add_to(folium_map)
    
    # Add air quality monitoring stations
    stations_group = folium.FeatureGroup(name=f'Air Quality Stations (n={len(stations_df)})')
    
    for idx, row in stations_df.iterrows():
        # Create popup text with all available information
        popup_text = f"<b>{row['station']}</b><br>"
        if 'address' in row:
            popup_text += f"Address: {row['address']}<br>"
        popup_text += f"Lat: {row['latitude']:.6f}<br>"
        popup_text += f"Lon: {row['longitude']:.6f}"
        
        # Add marker with custom icon
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=row['station'],
            color='darkred',
            fill=True,
            fillColor='red',
            fillOpacity=0.9,
            weight=2
        ).add_to(stations_group)
        
        # Add station label
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(
                html=f'<div style="font-size: 10pt; font-weight: bold; '
                     f'background-color: white; padding: 2px; '
                     f'border: 1px solid black; border-radius: 3px;">'
                     f'{row["station"]}</div>'
            )
        ).add_to(stations_group)
    
    stations_group.add_to(folium_map)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(folium_map)
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; 
                left: 50%; 
                transform: translateX(-50%);
                width: 90%; 
                max-width: 800px;
                z-index: 9999;
                background-color: white;
                padding: 10px;
                border: 2px solid black;
                border-radius: 5px;
                text-align: center;">
        <h3 style="margin: 0; font-family: Arial; font-size: 20px;">
            <b>Monitoring Station Network: Traffic and Air Quality</b>
        </h3>
    </div>
    '''
    folium_map.get_root().html.add_child(folium.Element(title_html))
    
    # Fit bounds to show all data
    try:
        all_bounds = []
        if buffer_zones is not None and len(buffer_zones) > 0:
            all_bounds.extend(buffer_zones.total_bounds)
        if bologna_metadata is not None and len(bologna_metadata) > 0:
            all_bounds.extend(bologna_metadata.total_bounds)
        if regional_metadata is not None and len(regional_metadata) > 0:
            all_bounds.extend(regional_metadata.total_bounds)
        all_bounds.extend(stations_df.total_bounds)
        
        sw = [min(all_bounds[1::2]), min(all_bounds[0::2])]
        ne = [max(all_bounds[3::2]), max(all_bounds[2::2])]
        folium_map.fit_bounds([sw, ne])
    except:
        pass
    
    return folium_map

def visualize_traffic_flow_directions(df_bol, bologna_metadata):
    """
    Visualize traffic flow directions using arrows
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    
    # Ensure CRS consistency
    crs_wgs84 = 'EPSG:4326'
    if bologna_metadata is not None:
        bologna_metadata = bologna_metadata.to_crs(crs_wgs84)
    
    # Merge traffic data with metadata to get locations
    if 'id_uni' in df_bol.columns and 'codice' in bologna_metadata.columns:
        # Aggregate traffic by station and direction
        traffic_summary = df_bol.groupby(['id_uni', 'direction', 'angle']).agg({
            'total_vehicles': 'sum',
            'Heavy_Count': 'sum'
        }).reset_index()
        
        # Merge with geometry
        traffic_with_geo = traffic_summary.merge(
            bologna_metadata[['codice', 'geometry']], 
            left_on='id_uni', 
            right_on='codice'
        )
        
        # Convert to GeoDataFrame
        traffic_gdf = gpd.GeoDataFrame(traffic_with_geo, crs=crs_wgs84)
        
        # Plot base stations
        bologna_metadata.plot(ax=ax, 
                             color='lightgray', 
                             marker='o', 
                             markersize=30, 
                             alpha=0.5,
                             edgecolor='gray',
                             linewidth=1)
        
        # Normalize traffic volume for arrow sizing
        max_traffic = traffic_gdf['total_vehicles'].max()
        min_traffic = traffic_gdf['total_vehicles'].min()
        
        # Create colormap for traffic volume
        cmap = plt.cm.YlOrRd
        norm = plt.Normalize(vmin=min_traffic, vmax=max_traffic)
        
        # Plot arrows for each direction
        for idx, row in traffic_gdf.iterrows():
            # Get coordinates
            x, y = row.geometry.x, row.geometry.y
            
            # Convert angle to radians and calculate arrow direction
            angle_rad = np.radians(row['angle'])
            
            # Arrow length proportional to traffic volume
            arrow_length = 0.005 * (row['total_vehicles'] / max_traffic) + 0.002
            dx = arrow_length * np.cos(angle_rad)
            dy = arrow_length * np.sin(angle_rad)
            
            # Arrow width proportional to heavy vehicle count
            arrow_width = 0.0002 + 0.0003 * (row['Heavy_Count'] / traffic_gdf['Heavy_Count'].max())
            
            # Color based on total traffic
            color = cmap(norm(row['total_vehicles']))
            
            # Create arrow
            arrow = FancyArrowPatch((x, y), (x + dx, y + dy),
                                   arrowstyle='->', 
                                   connectionstyle='arc3,rad=0',
                                   color=color,
                                   linewidth=2,
                                   mutation_scale=20,
                                   alpha=0.8)
            ax.add_patch(arrow)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Total Vehicle Count', fontsize=12)
        
        # Add direction legend
        legend_elements = []
        for direction in df_bol['direction'].unique()[:5]:  # Show first 5 directions
            legend_elements.append(plt.Line2D([0], [0], marker='>', color='gray', 
                                            markersize=10, label=f'Direction: {direction}'))
        
        # Add legend for arrow size
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=4, 
                                        label='Width ∝ Heavy vehicles'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.15, 1))
    
    # Set bounds
    if bologna_metadata is not None and len(bologna_metadata) > 0:
        bounds = bologna_metadata.total_bounds
        dx = (bounds[2] - bounds[0]) * 0.1
        dy = (bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - dx, bounds[2] + dx)
        ax.set_ylim(bounds[1] - dy, bounds[3] + dy)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs=crs_wgs84, source=ctx.providers.CartoDB.Positron)
    except:
        print("Warning: Could not add basemap")
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Traffic Flow Directions and Intensity', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()



def create_simple_traffic_flow_map(df_bol, bol_gdf, bologna_metadata):
    """
    Simpler version focusing on clarity
    """
    
    # Get unique flows (you might want to aggregate by time period)
    # For now, let's just take a sample or specific time
    sample_flows = df_bol.groupby(['id_uni', 'Node_da', 'Node_a', 'Forward']).agg({
        'total_vehicles': 'sum',
        'Heavy_Count': 'sum',
        'direction': 'first',
        'angle': 'first'
    }).reset_index()
    
    # Create flow lines
    flows = []
    for idx, row in sample_flows.iterrows():
        start_node = bol_gdf[bol_gdf['codice'] == row['Node_da']]
        end_node = bol_gdf[bol_gdf['codice'] == row['Node_a']]
        
        if len(start_node) > 0 and len(end_node) > 0:
            if row['Forward'] == 1:
                line = LineString([start_node.geometry.iloc[0], end_node.geometry.iloc[0]])
            else:
                line = LineString([end_node.geometry.iloc[0], start_node.geometry.iloc[0]])
            
            flows.append({
                'geometry': line,
                'sensor': row['id_uni'],
                'vehicles': row['total_vehicles'],
                'from': row['Node_da'] if row['Forward'] == 1 else row['Node_a'],
                'to': row['Node_a'] if row['Forward'] == 1 else row['Node_da']
            })
    flow_gdf = gpd.GeoDataFrame(flows, crs=bol_gdf.crs, geometry='geometry')
    
    # Create map
    m = bologna_metadata.explore(
        tooltip=['codice'],
        marker_kwds={'radius': 8, 'color': 'red', 'fill': True},
        style_kwds={'fillOpacity': 0.7},
        tiles='OpenStreetMap',
        name='Traffic Sensors'
    )
    
    # Add flows with thickness based on volume
    flow_gdf['line_width'] = 1 + (flow_gdf['vehicles'] / flow_gdf['vehicles'].max() * 5)
    
    flow_gdf.explore(
        m=m,
        color='blue',
        style_kwds={'opacity': 0.6},
        tooltip=['sensor', 'from', 'to', 'vehicles'],
        name='Traffic Flows'
    )
    
    return m

# Alternative simplified approach using just sensor locations
def create_simple_road_heatmap(df_bol, bologna_metadata, bol_gdf):
    """
    Simplified version that maps traffic directly to nearest roads
    """
    print("Creating simplified road traffic heatmap...")
    
    # Get road network
    bounds = bologna_metadata.total_bounds
    G = ox.graph_from_bbox(
        bounds[3] + 0.01, bounds[1] - 0.01, 
        bounds[2] + 0.01, bounds[0] - 0.01, 
        network_type='drive'
    )
    nodes, edges = ox.graph_to_gdfs(G)
    
    # Ensure same CRS
    if edges.crs != bologna_metadata.crs:
        edges = edges.to_crs(bologna_metadata.crs)
    
    # Aggregate traffic by sensor
    df_bol['total_vehicles'] = df_bol['Light_Count'] + df_bol['Medium_Count'] + df_bol['Heavy_Count']
    sensor_traffic = df_bol.groupby('id_uni').agg({
        'total_vehicles': 'sum',
        'Heavy_Count': 'sum'
    }).reset_index()
    
    # Merge with sensor locations
    sensor_geo = bologna_metadata.merge(
        sensor_traffic,
        left_on='codice',
        right_on='id_uni',
        how='inner'
    )
    
    # Map each sensor to nearest road
    road_traffic_dict = {}
    
    for idx, sensor in sensor_geo.iterrows():
        # Find nearest road
        distances = edges.geometry.distance(sensor.geometry)
        nearest_idx = distances.idxmin()
        
        # Add traffic to road
        if nearest_idx not in road_traffic_dict:
            road_traffic_dict[nearest_idx] = {
                'traffic_volume': 0,
                'heavy_vehicles': 0,
                'sensor_count': 0
            }
        
        road_traffic_dict[nearest_idx]['traffic_volume'] += sensor['total_vehicles']
        road_traffic_dict[nearest_idx]['heavy_vehicles'] += sensor['Heavy_Count']
        road_traffic_dict[nearest_idx]['sensor_count'] += 1
    
    # Add traffic data to roads
    edges['traffic_volume'] = 0
    edges['heavy_vehicles'] = 0
    edges['sensor_count'] = 0
    
    for road_idx, traffic_data in road_traffic_dict.items():
        edges.loc[road_idx, 'traffic_volume'] = traffic_data['traffic_volume']
        edges.loc[road_idx, 'heavy_vehicles'] = traffic_data['heavy_vehicles']
        edges.loc[road_idx, 'sensor_count'] = traffic_data['sensor_count']
    
    # Calculate density
    edges['traffic_density'] = edges['traffic_volume'] / edges.geometry.length
    
    # Create interactive map
    center_lat = bologna_metadata.geometry.y.mean()
    center_lon = bologna_metadata.geometry.x.mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')
    
    # Add heatmap layer using sensor points
    heat_data = [[row.geometry.y, row.geometry.x, row['total_vehicles']] 
                 for idx, row in sensor_geo.iterrows()]
    
    plugins.HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=25,
        blur=15,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1: 'red'}
    ).add_to(m)
    
    # Add road colors for roads with traffic
    traffic_roads = edges[edges['traffic_volume'] > 0]
    
    if len(traffic_roads) > 0:
        max_volume = traffic_roads['traffic_volume'].quantile(0.95)
        
        for idx, road in traffic_roads.iterrows():
            # Normalize volume for color
            norm_volume = min(road['traffic_volume'] / max_volume, 1.0)
            
            # Create color
            color = f'#{int(255 * norm_volume):02x}0000'  # Red gradient
            
            # Get coordinates
            coords = [[lat, lon] for lon, lat in road.geometry.coords]
            
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                opacity=0.7,
                popup=f"Traffic: {int(road['traffic_volume'])} vehicles<br>"
                      f"Sensors: {int(road['sensor_count'])}",
                tooltip=f"{int(road['traffic_volume'])} vehicles"
            ).add_to(m)
    
    # Add sensors
    for idx, sensor in sensor_geo.iterrows():
        folium.CircleMarker(
            location=[sensor.geometry.y, sensor.geometry.x],
            radius=6,
            color='darkblue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.8,
            popup=f"Sensor: {sensor['codice']}<br>Traffic: {int(sensor['total_vehicles'])}",
            tooltip=f"ID: {sensor['codice']}"
        ).add_to(m)
    
    return m, edges



##################################
# Traffic Heatmap Visualizations #
##################################


# 1. INTERACTIVE FOLIUM HEATMAP
def create_traffic_heatmap(df_bol, bologna_metadata):
    """
    Creates an interactive heatmap of traffic volume using Folium
    """
    # Merge traffic data with locations
    traffic_with_loc = df_bol.groupby('id_uni')['total_vehicles'].mean().reset_index()
    traffic_with_loc = traffic_with_loc.merge(bologna_metadata[['codice', 'geometry']], 
                                              left_on='id_uni', 
                                              right_on='codice')
    traffic_with_loc = gpd.GeoDataFrame(traffic_with_loc, geometry='geometry', crs='EPSG:4326')
    
    # Extract coordinates
    traffic_with_loc['lat'] = traffic_with_loc['geometry'].y
    traffic_with_loc['lon'] = traffic_with_loc['geometry'].x
    
    # Create base map centered on Bologna
    center_lat = traffic_with_loc['lat'].mean()
    center_lon = traffic_with_loc['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Prepare data for heatmap
    heat_data = [[row['lat'], row['lon'], row['total_vehicles']] 
                 for idx, row in traffic_with_loc.iterrows()]
    
    # Add heatmap
    HeatMap(heat_data, 
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'orange', 1.0: 'red'}).add_to(m)
    
    # Add markers for high traffic areas
    for idx, row in traffic_with_loc.iterrows():
        if row['total_vehicles'] > traffic_with_loc['total_vehicles'].quantile(0.9):
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                popup=f"Sensor {row['id_uni']}<br>Avg Traffic: {row['total_vehicles']:.0f}",
                color='red',
                fill=True,
                fillColor='red'
            ).add_to(m)
    
    return m

# 2. TIME-ANIMATED HEATMAP
def create_animated_traffic_heatmap(df_bol, bologna_metadata):
    """
    Creates an animated heatmap showing traffic changes over time
    """
    # Merge with locations
    df_with_loc = df_bol.merge(bologna_metadata[['codice', 'geometry']], 
                               left_on='id_uni', 
                               right_on='codice')
    
    df_with_loc = gpd.GeoDataFrame(df_with_loc, geometry='geometry', crs='EPSG:4326')
    
    # Extract coordinates
    df_with_loc['lat'] = df_with_loc['geometry'].y
    df_with_loc['lon'] = df_with_loc['geometry'].x
    
    # Group by hour for animation
    df_with_loc['hour'] = df_with_loc['datetime'].dt.hour
    
    # Prepare data for each hour
    heat_data_time = []
    for hour in range(24):
        hour_data = df_with_loc[df_with_loc['hour'] == hour]
        hour_avg = hour_data.groupby(['lat', 'lon'])['total_vehicles'].mean().reset_index()
        heat_data_time.append([[row['lat'], row['lon'], row['total_vehicles']] 
                              for idx, row in hour_avg.iterrows()])
    
    # Create base map
    center_lat = df_with_loc['lat'].mean()
    center_lon = df_with_loc['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add time-based heatmap
    HeatMapWithTime(heat_data_time,
                    index=[f"{h:02d}:00" for h in range(24)],
                    auto_play=True,
                    max_opacity=0.8,
                    radius=25,
                    gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 
                             0.8: 'yellow', 0.9: 'orange', 1.0: 'red'}).add_to(m)
    
    return m

# 3. HEXBIN GEOGRAPHICAL HEATMAP
def create_hexbin_heatmap(df_bol, bologna_metadata):
    """
    Creates a hexagonal bin heatmap for traffic density
    """
    # Merge and prepare data
    traffic_avg = df_bol.groupby('id_uni')['total_vehicles'].mean().reset_index()
    traffic_with_loc = traffic_avg.merge(bologna_metadata[['codice', 'geometry']], 
                                        left_on='id_uni', 
                                        right_on='codice')
    traffic_with_loc = gpd.GeoDataFrame(traffic_with_loc, geometry='geometry', crs='EPSG:4326')
    traffic_with_loc['lon'] = traffic_with_loc['geometry'].x
    traffic_with_loc['lat'] = traffic_with_loc['geometry'].y
    
    # Expand points based on traffic volume for density effect
    expanded_points = []
    for _, row in traffic_with_loc.iterrows():
        # Add multiple points based on traffic volume
        n_points = int(row['total_vehicles'] )  # Scale factor
        for _ in range(max(1, n_points)):
            # Add small random offset
            lon_offset = np.random.normal(0, 0.001)
            lat_offset = np.random.normal(0, 0.001)
            expanded_points.append([row['lon'] + lon_offset, 
                                  row['lat'] + lat_offset])
    
    expanded_df = pd.DataFrame(expanded_points, columns=['lon', 'lat'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create hexbin plot
    hb = ax.hexbin(expanded_df['lon'], expanded_df['lat'], 
                   gridsize=30, cmap='YlOrRd', alpha=0.8, 
                   mincnt=1, edgecolors='black', linewidths=0.2)
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Traffic Density', fontsize=12)
    
    # Plot actual sensor locations
    ax.scatter(traffic_with_loc['lon'], traffic_with_loc['lat'], 
              c='blue', s=30, edgecolors='white', linewidth=1, 
              zorder=5, label='Traffic Sensors')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Traffic Density Hexbin Heatmap - Bologna', fontsize=16, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

# 4. PEAK HOUR COMPARISON HEATMAP
def create_peak_hour_comparison(df_bol, bologna_metadata):
    """
    Creates side-by-side heatmaps comparing rush hour vs non-rush hour traffic
    """
    # Define rush hours (7-9 AM and 5-7 PM)
    df_bol['hour'] = df_bol['datetime'].dt.hour
    df_bol['is_rush_hour'] = df_bol['hour'].isin([7, 8, 17, 18])
    
    # Calculate averages for rush and non-rush hours
    rush_traffic = df_bol[df_bol['is_rush_hour']].groupby('id_uni')['total_vehicles'].mean().reset_index()
    non_rush_traffic = df_bol[~df_bol['is_rush_hour']].groupby('id_uni')['total_vehicles'].mean().reset_index()
    
    # Merge with locations
    rush_with_loc = rush_traffic.merge(bologna_metadata[['codice', 'geometry']], 
                                      left_on='id_uni', right_on='codice')
    non_rush_with_loc = non_rush_traffic.merge(bologna_metadata[['codice', 'geometry']], 
                                              left_on='id_uni', right_on='codice')
    rush_with_loc = gpd.GeoDataFrame(rush_with_loc, geometry='geometry', crs='EPSG:4326')
    non_rush_with_loc = gpd.GeoDataFrame(non_rush_with_loc, geometry='geometry', crs='EPSG:4326')
    # Extract coordinates
    for df in [rush_with_loc, non_rush_with_loc]:
        df['lon'] = df['geometry'].x
        df['lat'] = df['geometry'].y
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Common colormap normalization
    vmin = min(rush_with_loc['total_vehicles'].min(), non_rush_with_loc['total_vehicles'].min())
    vmax = max(rush_with_loc['total_vehicles'].max(), non_rush_with_loc['total_vehicles'].max())
    
    # Plot rush hour
    scatter1 = ax1.scatter(rush_with_loc['lon'], rush_with_loc['lat'], 
                          c=rush_with_loc['total_vehicles'], 
                          s=200, cmap='Reds', 
                          edgecolors='black', linewidth=0.5,
                          vmin=vmin, vmax=vmax, alpha=0.8)
    ax1.set_title('Rush Hour Traffic (7-9 AM, 5-7 PM)', fontsize=14)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot non-rush hour
    scatter2 = ax2.scatter(non_rush_with_loc['lon'], non_rush_with_loc['lat'], 
                          c=non_rush_with_loc['total_vehicles'], 
                          s=200, cmap='Reds', 
                          edgecolors='black', linewidth=0.5,
                          vmin=vmin, vmax=vmax, alpha=0.8)
    ax2.set_title('Non-Rush Hour Traffic', fontsize=14)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1, label='Average Vehicle Count')
    cbar2 = plt.colorbar(scatter2, ax=ax2, label='Average Vehicle Count')
    
    # Add basemaps
    for ax in [ax1, ax2]:
        try:
            ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron, alpha=0.5)
        except:
            pass
    
    plt.suptitle('Traffic Volume Comparison: Rush Hour vs Non-Rush Hour', fontsize=16, y=0.98)
    plt.tight_layout()
    return fig



def create_traffic_heatmap_png(df_bol, bologna_metadata, stations_df=None, 
                              bologna_boundary=None, save_path='bologna_traffic_heatmap.png', 
                              dpi=300, figsize=(16, 12)):
    """
    Creates a static PNG heatmap of traffic volume with air quality stations and full city boundary
    """
    # Merge traffic data with locations
    traffic_with_loc = df_bol.groupby('id_uni')['total_vehicles'].mean().reset_index()
    traffic_with_loc = traffic_with_loc.merge(bologna_metadata[['codice', 'geometry']], 
                                              left_on='id_uni', 
                                              right_on='codice')
    traffic_with_loc = gpd.GeoDataFrame(traffic_with_loc, geometry='geometry', crs='EPSG:4326')
    
    # Extract coordinates
    traffic_with_loc['lon'] = traffic_with_loc['geometry'].x
    traffic_with_loc['lat'] = traffic_with_loc['geometry'].y
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine map extent based on Bologna boundary (if provided) or traffic points
    if bologna_boundary is not None:
        if not isinstance(bologna_boundary, gpd.GeoDataFrame):
            bologna_boundary = gpd.GeoDataFrame(bologna_boundary, geometry='geometry', crs='EPSG:4326')
        else:
            bologna_boundary = bologna_boundary.to_crs('EPSG:4326')
        
        # Get bounds from Bologna boundary
        bounds = bologna_boundary.total_bounds  # [minx, miny, maxx, maxy]
        lon_min, lat_min, lon_max, lat_max = bounds
        
        # Add padding to ensure full boundary is visible
        lon_pad = (lon_max - lon_min) * 0.05  # 5% padding
        lat_pad = (lat_max - lat_min) * 0.05
    else:
        # Use traffic points bounds if no boundary provided
        lon_min, lon_max = traffic_with_loc['lon'].min(), traffic_with_loc['lon'].max()
        lat_min, lat_max = traffic_with_loc['lat'].min(), traffic_with_loc['lat'].max()
        lon_pad = (lon_max - lon_min) * 0.1
        lat_pad = (lat_max - lat_min) * 0.1
    
    # Set the axis limits to show full Bologna
    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)
    
    # Add basemap first (so it's in the background)
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', 
                        source=ctx.providers.CartoDB.Positron, 
                        alpha=0.6, zoom='auto')
    except Exception as e:
        print(f"Warning: Could not add basemap. Error: {e}")
        ax.set_facecolor('#f0f0f0')
    
    # Create heatmap using KDE
    # Prepare weighted points based on traffic volume
    lon_points = []
    lat_points = []
    
    for _, row in traffic_with_loc.iterrows():
        # Repeat points based on traffic volume (normalized)
        repeat_count = int(row['total_vehicles'] / 100)
        repeat_count = max(1, min(repeat_count, 100))
        
        # Add some random jitter to avoid exact overlaps
        for _ in range(repeat_count):
            lon_points.append(row['lon'] + np.random.normal(0, 0.0001))
            lat_points.append(row['lat'] + np.random.normal(0, 0.0001))
    
    # Create grid for density calculation (covering full extent)
    xx, yy = np.mgrid[lon_min-lon_pad:lon_max+lon_pad:300j, 
                      lat_min-lat_pad:lat_max+lat_pad:300j]
    
    # Calculate kernel density
    if len(lon_points) > 0:
        positions = np.vstack([lon_points, lat_points])
        kernel = gaussian_kde(positions)
        f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
        
        # Plot heatmap
        heatmap = ax.contourf(xx, yy, f.T, levels=20, cmap='YlOrRd', alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', 
                           pad=0.02, shrink=0.8)
        cbar.set_label('Traffic Density', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
    
    # Add Bologna boundary
    if bologna_boundary is not None:
        bologna_boundary.boundary.plot(ax=ax, color='black', linewidth=3, 
                                     linestyle='-', alpha=0.8, label='Bologna City Boundary')
        
        # Optional: add a subtle fill to show city area
        bologna_boundary.plot(ax=ax, color='none', edgecolor='black', 
                            linewidth=3, alpha=0.8)
    
    # Plot traffic sensors
    # High traffic sensors
    high_traffic_threshold = traffic_with_loc['total_vehicles'].quantile(0.9)
    high_traffic = traffic_with_loc[traffic_with_loc['total_vehicles'] > high_traffic_threshold]
    regular_traffic = traffic_with_loc[traffic_with_loc['total_vehicles'] <= high_traffic_threshold]
    
    # Plot regular sensors
    regular_traffic.plot(ax=ax, color='darkred', markersize=30, 
                        alpha=0.6, edgecolor='white', linewidth=0.5)
    
    # Plot high traffic sensors
    high_traffic.plot(ax=ax, color='red', markersize=60, 
                     marker='o', edgecolor='darkred', linewidth=2,
                     label=f'High Traffic Sensors (>{high_traffic_threshold:.0f} vehicles/hr)')
    
    # Add air quality stations if provided
    if stations_df is not None:
        # Create GeoDataFrame for stations
        stations_gdf = gpd.GeoDataFrame(
            stations_df, 
            geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude),
            crs='EPSG:4326'
        )
        
        # Plot station areas (circles)
        for idx, station in stations_gdf.iterrows():
            circle = plt.Circle((station['longitude'], station['latitude']), 
                              0.005,  # Radius in degrees (approximately 500m)
                              color='#2E86AB', fill=True, alpha=0.2, 
                              edgecolor='#2E86AB', linewidth=2)
            ax.add_patch(circle)
        
        # Plot station markers
        stations_gdf.plot(ax=ax, color='#2E86AB', markersize=250, 
                         marker='*', edgecolor='white', linewidth=2, 
                         label='Air Quality Stations', zorder=10)
        
        # Add station labels
        for idx, station in stations_gdf.iterrows():
            ax.annotate(station['station'], 
                       xy=(station['longitude'], station['latitude']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', edgecolor='#2E86AB', 
                               alpha=0.9),
                       ha='left')
    
    # Create custom legend
    legend_elements = []
    
    if stations_df is not None:
        legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                    markerfacecolor='#2E86AB', markersize=15, 
                                    label='Air Quality Stations', markeredgecolor='white'))
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='red', markersize=10, 
                                label='High Traffic Sensors (>90th percentile)', 
                                markeredgecolor='darkred'))
    
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='darkred', markersize=8, 
                                label='Traffic Sensors', markeredgecolor='white'))
    
    if bologna_boundary is not None:
        legend_elements.append(Line2D([0], [0], color='black', linewidth=3, 
                                    linestyle='-', label='Bologna City Boundary'))
    
    # Add legend
    legend = ax.legend(handles=legend_elements, loc='upper right', 
                      fontsize=10, frameon=True, fancybox=True, 
                      shadow=True, framealpha=0.9)
    
    # Title and labels
    ax.set_title('Bologna Traffic Volume Heatmap with Air Quality Monitoring Stations', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add data source and timestamp
    fig.text(0.99, 0.01, f'Traffic sensors: {len(traffic_with_loc)} | ' + 
             f'Air quality stations: {len(stations_df) if stations_df is not None else 0}',
             ha='right', va='bottom', fontsize=9, alpha=0.7)
    
    # Ensure aspect ratio is equal to prevent distortion
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save as PNG
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Heatmap saved as: {save_path}")
    
    return fig, ax


# Simple version without adjustText
def plot_heavy_vehicle_ratio(zones_map, figsize=(16, 12), save_path=None, dpi=300):
    """
    Simpler version without text adjustment
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure CRS is WGS84
    if zones_map.crs != 'EPSG:4326':
        zones_map_plot = zones_map.to_crs('EPSG:4326')
    else:
        zones_map_plot = zones_map.copy()
    
    # Plot zones
    zones_map_plot.plot(column='heavy_vehicle_ratio', 
                        ax=ax, 
                        cmap='Blues',
                        alpha=0.5,
                        edgecolor='darkblue',
                        linewidth=1,
                        legend=True,
                        legend_kwds={'label': 'Heavy Vehicle Ratio',
                                    'orientation': 'vertical',
                                    'shrink': 0.8})
    
    # Add basemap
    try:
        ctx.add_basemap(ax, crs='EPSG:4326', 
                        source=ctx.providers.CartoDB.Positron, 
                        alpha=0.8)
    except:
        ax.set_facecolor('#f0f0f0')
    
    # Add zone names
    for idx, row in zones_map_plot.iterrows():
        if pd.notna(row.get('zone_name', None)) and pd.notna(row.get('heavy_vehicle_ratio', None)):
            centroid = row['geometry'].centroid
            ax.annotate(f"{row['zone_name']}\n{row['heavy_vehicle_ratio']:.1%}", 
                       xy=(centroid.x, centroid.y),
                       ha='center', 
                       va='center',
                       fontsize=8,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', 
                               facecolor='white', 
                               edgecolor='darkblue',
                               alpha=0.8))
    
    ax.set_title('Heavy Vehicle Ratio by Zone in Bologna', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return fig, ax
