import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
import folium
from folium.plugins import HeatMap
import plotly.graph_objects as go
import plotly.express as px
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TrafficAirQualityVisualizer:
    def __init__(self, final_dataset, buffer_zones, traffic_with_zones, 
                 zone_summary, stations_df):
        self.final_dataset = final_dataset
        self.buffer_zones = buffer_zones
        self.traffic_with_zones = traffic_with_zones
        self.zone_summary = zone_summary
        self.stations_df = stations_df
        
    def create_traffic_density_heatmap(self, datetime_filter=None, bologna_gpd=None):
        """Create interactive traffic density heat map using Folium"""
        
        # Filter data if datetime provided
        if datetime_filter:
            data = self.traffic_with_zones[
                self.traffic_with_zones['datetime'] == datetime_filter
            ].copy()
        else:
            # Use aggregated data
            data = self.traffic_with_zones.groupby('geometry').agg({
                'total_vehicles': 'mean',
                'weighted_total': 'mean'
            }).reset_index()
        
        # Create base map centered on Bologna
        center_lat = self.stations_df.latitude.mean()
        center_lon = self.stations_df.longitude.mean()
        traffic_map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='CartoDB positron'
        )

        # Add Bologna boundary if provided
        if bologna_gpd is not None:
            folium.GeoJson(
                bologna_gpd.geometry.iloc[0].__geo_interface__,
                name="Bologna Boundary",
                style_function=lambda x: {
                    'fillColor': 'none',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.1
                }
            ).add_to(traffic_map)
        
        # Prepare data for heatmap
        heat_data = []
        for idx, row in data.iterrows():
            if hasattr(row.geometry, 'x'):
                heat_data.append([
                    row.geometry.y,  # latitude
                    row.geometry.x,  # longitude
                    row['weighted_total']  # weight
                ])
        
        # Add heatmap layer
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.0: 'blue',
                0.25: 'cyan',
                0.5: 'yellow',
                0.75: 'orange',
                1.0: 'red'
            }
        ).add_to(traffic_map)
        
        # Add air quality stations
        for idx, station in self.stations_df.iterrows():
            folium.CircleMarker(
                location=[station.latitude, station.longitude],
                radius=8,
                popup=f"Station: {station.station}",
                color='black',
                fill=True,
                fillColor='white',
                fillOpacity=0.8,
                weight=2
            ).add_to(traffic_map)
        
        # Add zone boundaries
        for idx, zone in self.buffer_zones.iterrows():
            folium.GeoJson(
                zone.geometry.__geo_interface__,
                style_function=lambda x, zone_id=zone.zone_id: {
                    'fillColor': ['green', 'yellow', 'red'][zone_id-1],
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.1
                }
            ).add_to(traffic_map)
        
        return traffic_map
    
    def create_weighted_impact_by_zone(self):
        """Visualize weighted traffic impact by zone"""
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Heatmap of weighted impact by station and zone
        ax1 = plt.subplot(2, 2, 1)
        impact_pivot = self.zone_summary.pivot(
            index='station_name', 
            columns='zone_name', 
            values='weighted_total_mean'
        )
        
        sns.heatmap(impact_pivot, 
                   annot=True, 
                   fmt='.0f',
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Weighted Traffic Impact'},
                   ax=ax1)
        ax1.set_title('Average Weighted Traffic Impact by Zone and Station', fontsize=14)
        ax1.set_xlabel('Zone')
        ax1.set_ylabel('Air Quality Station')
        
        # 2. 3D surface plot of impact distribution
        ax2 = plt.subplot(2, 2, 2, projection='3d')
        
        # Create mesh for 3D plot
        stations = self.zone_summary['station_name'].unique()
        zones = self.zone_summary['zone_id'].unique()
        
        X, Y = np.meshgrid(range(len(zones)), range(len(stations)))
        Z = np.zeros_like(X, dtype=float)
        
        for i, station in enumerate(stations):
            for j, zone in enumerate(zones):
                value = self.zone_summary[
                    (self.zone_summary['station_name'] == station) & 
                    (self.zone_summary['zone_id'] == zone)
                ]['weighted_total_mean'].values
                if len(value) > 0:
                    Z[i, j] = value[0]
        
        surf = ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('Zone')
        ax2.set_ylabel('Station')
        ax2.set_zlabel('Weighted Impact')
        ax2.set_title('3D Surface of Weighted Traffic Impact')
        ax2.set_xticks(range(len(zones)))
        ax2.set_xticklabels([f'Zone {z}' for z in zones])
        ax2.set_yticks(range(len(stations)))
        ax2.set_yticklabels(stations, rotation=45)
        
        # 3. Stacked bar chart of traffic volume by zone
        ax3 = plt.subplot(2, 2, 3)
        
        zone_traffic = self.zone_summary.pivot(
            index='station_name',
            columns='zone_name',
            values='total_vehicles_sum'
        )
        
        zone_traffic.plot(kind='bar', 
                         stacked=True, 
                         ax=ax3,
                         color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax3.set_title('Total Traffic Volume Distribution by Zone', fontsize=14)
        ax3.set_xlabel('Air Quality Station')
        ax3.set_ylabel('Total Vehicle Count')
        ax3.legend(title='Zone', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        
        # 4. Radial plot of zone contributions
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        # Select one station for radial plot
        station_example = stations[0]
        station_data = self.zone_summary[
            self.zone_summary['station_name'] == station_example
        ]
        
        theta = np.linspace(0, 2 * np.pi, len(station_data) + 1)
        radii = station_data['weighted_total_mean'].values
        radii = np.append(radii, radii[0])  # Close the plot
        
        ax4.plot(theta, radii, 'o-', linewidth=2, markersize=8)
        ax4.fill(theta, radii, alpha=0.3)
        ax4.set_xticks(theta[:-1])
        ax4.set_xticklabels([f'Zone {z}' for z in station_data['zone_id']])
        ax4.set_title(f'Weighted Impact by Zone\n{station_example}', fontsize=14, pad=20)
        
        plt.tight_layout()
        return fig
    
    def create_distance_decay_analysis(self):
        """Analyze and visualize distance-decay relationships"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Scatter plot with distance vs weighted impact
        ax1 = axes[0, 0]
        
        # Calculate distance for each traffic point
        traffic_distances = []
        for idx, row in self.traffic_with_zones.iterrows():
            station_point = self.stations_df[
                self.stations_df['station'] == row['station_name']
            ].geometry.iloc[0]
            distance = row.geometry.distance(station_point) * 111  # Convert to km
            traffic_distances.append({
                'distance_km': distance,
                'weighted_total': row['weighted_total'],
                'zone_id': row['zone_id'],
                'station_name': row['station_name']
            })
        
        distance_df = pd.DataFrame(traffic_distances)
        
        # Plot by zone
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for i, zone in enumerate([1, 2, 3]):
            zone_data = distance_df[distance_df['zone_id'] == zone]
            ax1.scatter(zone_data['distance_km'], 
                       zone_data['weighted_total'],
                       alpha=0.5, 
                       label=f'Zone {zone}',
                       color=colors[i-1],
                       s=30)
        
        # Add exponential decay fit
        from scipy.optimize import curve_fit
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        
        x_data = distance_df['distance_km'].values
        y_data = distance_df['weighted_total'].values
        
        # Fit only on positive values
        mask = y_data > 0
        popt, _ = curve_fit(exp_decay, x_data[mask], y_data[mask], p0=[100, 0.5])
        
        x_fit = np.linspace(0, distance_df['distance_km'].max(), 100)
        y_fit = exp_decay(x_fit, *popt)
        
        ax1.plot(x_fit, y_fit, 'k--', linewidth=2, 
                label=f'Exponential fit: {popt[0]:.1f} * exp(-{popt[1]:.2f}x)')
        
        ax1.set_xlabel('Distance from Station (km)')
        ax1.set_ylabel('Weighted Traffic Impact')
        ax1.set_title('Distance Decay of Traffic Impact')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot of impact by zone
        ax2 = axes[0, 1]
        
        zone_names = {1: 'Zone 1\n(0-0.5 km)', 
                     2: 'Zone 2\n(0.5-1.5 km)', 
                     3: 'Zone 3\n(1.5-3.0 km)'}
        distance_df['zone_label'] = distance_df['zone_id'].map(zone_names)
        
        bp = ax2.boxplot([distance_df[distance_df['zone_id'] == z]['weighted_total'].values 
                         for z in [1, 2, 3]],
                        labels=[zone_names[z] for z in [1, 2, 3]],
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax2.set_ylabel('Weighted Traffic Impact')
        ax2.set_title('Traffic Impact Distribution by Zone')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # 3. Average impact vs distance with confidence intervals
        ax3 = axes[1, 0]
        
        # Bin distances and calculate statistics
        distance_bins = np.linspace(0, distance_df['distance_km'].max(), 20)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        
        mean_impacts = []
        std_impacts = []
        
        for i in range(len(distance_bins) - 1):
            mask = (distance_df['distance_km'] >= distance_bins[i]) & \
                   (distance_df['distance_km'] < distance_bins[i+1])
            bin_data = distance_df.loc[mask, 'weighted_total']
            
            if len(bin_data) > 0:
                mean_impacts.append(bin_data.mean())
                std_impacts.append(bin_data.std())
            else:
                mean_impacts.append(np.nan)
                std_impacts.append(np.nan)
        
        mean_impacts = np.array(mean_impacts)
        std_impacts = np.array(std_impacts)
        
        # Plot with error bars
        valid = ~np.isnan(mean_impacts)
        ax3.errorbar(bin_centers[valid], mean_impacts[valid], 
                    yerr=std_impacts[valid], 
                    fmt='o-', capsize=5, capthick=2,
                    markersize=8, linewidth=2)
        
        ax3.fill_between(bin_centers[valid], 
                        mean_impacts[valid] - std_impacts[valid],
                        mean_impacts[valid] + std_impacts[valid],
                        alpha=0.3)
        
        ax3.set_xlabel('Distance from Station (km)')
        ax3.set_ylabel('Mean Weighted Traffic Impact')
        ax3.set_title('Average Impact vs Distance with Standard Deviation')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative impact by distance
        ax4 = axes[1, 1]
        
        # Sort by distance and calculate cumulative impact
        sorted_df = distance_df.sort_values('distance_km')
        cumulative_impact = sorted_df['weighted_total'].cumsum()
        total_impact = cumulative_impact.iloc[-1]
        cumulative_pct = (cumulative_impact / total_impact) * 100
        
        ax4.plot(sorted_df['distance_km'], cumulative_pct, linewidth=2)
        
        # Add zone boundaries
        zone_boundaries = [0.5, 1.5, 3.0]
        zone_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        for i, (boundary, color) in enumerate(zip(zone_boundaries, zone_colors)):
            ax4.axvline(boundary, color=color, linestyle='--', alpha=0.7, linewidth=2)
            # Find percentage at boundary
            idx = np.searchsorted(sorted_df['distance_km'], boundary)
            if idx < len(cumulative_pct):
                pct = cumulative_pct.iloc[idx]
                ax4.text(boundary + 0.1, pct, f'{pct:.1f}%', 
                        color=color, weight='bold', fontsize=10)
        
        ax4.set_xlabel('Distance from Station (km)')
        ax4.set_ylabel('Cumulative Impact (%)')
        ax4.set_title('Cumulative Traffic Impact by Distance')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def create_interactive_3d_visualization(self):
        """Create interactive 3D visualization using Plotly"""
        
        # Prepare data for 3D visualization
        viz_data = []
        
        for idx, row in self.traffic_with_zones.iterrows():
            if hasattr(row.geometry, 'x'):
                viz_data.append({
                    'lon': row.geometry.x,
                    'lat': row.geometry.y,
                    'weighted_impact': row['weighted_total'],
                    'zone_id': row['zone_id'],
                    'station': row['station_name'],
                    'total_vehicles': row.get('total_vehicles', 0)
                })
        
        viz_df = pd.DataFrame(viz_data)
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add traffic points colored by zone
        colors = ['green', 'yellow', 'red']
        for i, zone in enumerate([1, 2, 3]):
            zone_data = viz_df[viz_df['zone_id'] == zone]
            
            fig.add_trace(go.Scatter3d(
                x=zone_data['lon'],
                y=zone_data['lat'],
                z=zone_data['weighted_impact'],
                mode='markers',
                name=f'Zone {zone}',
                marker=dict(
                    size=5,
                    color=colors[i],
                    opacity=0.6,
                    line=dict(width=0.5, color='DarkSlateGray')
                ),
                text=[f"Station: {s}<br>Vehicles: {int(v)}<br>Impact: {w:.1f}" 
                      for s, v, w in zip(zone_data['station'], 
                                         zone_data['total_vehicles'],
                                         zone_data['weighted_impact'])],
                hoverinfo='text'
            ))
        
        # Add air quality stations as larger markers
        station_data = []
        for idx, station in self.stations_df.iterrows():
            # Get average NO2 for this station
            avg_no2 = self.final_dataset[
                self.final_dataset['station_name'] == station['station']
            ]['NO2'].mean()
            
            station_data.append({
                'lon': station.longitude,
                'lat': station.latitude,
                'name': station.station,
                'no2': avg_no2
            })
        
        station_df = pd.DataFrame(station_data)
        
        fig.add_trace(go.Scatter3d(
            x=station_df['lon'],
            y=station_df['lat'],
            z=[0] * len(station_df),  # Place at ground level
            mode='markers+text',
            name='Air Quality Stations',
            marker=dict(
                size=15,
                color=station_df['no2'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Avg NO2<br>(μg/m³)"),
                line=dict(width=2, color='black')
            ),
            text=station_df['name'],
            textposition='top center',
            hovertext=[f"{name}<br>Avg NO2: {no2:.1f} μg/m³" 
                      for name, no2 in zip(station_df['name'], station_df['no2'])]
        ))
        
        # Update layout
        fig.update_layout(
            title='3D Visualization of Traffic Impact and Air Quality Stations',
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Weighted Traffic Impact',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            height=800
        )
        
        return fig
    
    def create_zone_comparison_dashboard(self):
        """Create comprehensive zone comparison dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Traffic composition by zone
        ax1 = plt.subplot(2, 3, 1)
        
        # Get vehicle type proportions by zone
        vehicle_types = []
        for zone in [1, 2, 3]:
            zone_data = self.traffic_with_zones[self.traffic_with_zones['zone_id'] == zone]
            vehicle_types.append({
                'Zone': f'Zone {zone}',
                'Light': zone_data['Light_Count'].sum(),
                'Medium': zone_data['Medium_Count'].sum(),
                'Heavy': zone_data['Heavy_Count'].sum()
            })
        
        vehicle_df = pd.DataFrame(vehicle_types)
        vehicle_df.set_index('Zone', inplace=True)
        vehicle_df = vehicle_df.div(vehicle_df.sum(axis=1), axis=0) * 100
        
        vehicle_df.plot(kind='bar', stacked=True, ax=ax1,
                       color=['#3498db', '#f39c12', '#e74c3c'])
        ax1.set_ylabel('Percentage (%)')
        ax1.set_title('Vehicle Type Distribution by Zone')
        ax1.legend(title='Vehicle Type', bbox_to_anchor=(1.05, 1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
        
        # 2. Average impact per vehicle by zone
        ax2 = plt.subplot(2, 3, 2)
        
        impact_per_vehicle = []
        for zone in [1, 2, 3]:
            zone_data = self.traffic_with_zones[self.traffic_with_zones['zone_id'] == zone]
            avg_impact = zone_data['weighted_total'].sum() / zone_data['total_vehicles'].sum()
            impact_per_vehicle.append({
                'Zone': f'Zone {zone}',
                'Impact per Vehicle': avg_impact
            })
        
        impact_df = pd.DataFrame(impact_per_vehicle)
        bars = ax2.bar(impact_df['Zone'], impact_df['Impact per Vehicle'],
                       color=['#2ecc71', '#f39c12', '#e74c3c'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        ax2.set_ylabel('Average Impact per Vehicle')
        ax2.set_title('Traffic Impact Efficiency by Zone')
        
        
        # 4. Zone coverage map
        ax4 = plt.subplot(2, 3, 4)
        
        # Plot zone boundaries
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for idx, zone in self.buffer_zones.iterrows():
            zone_gdf = gpd.GeoDataFrame([zone], crs='EPSG:4326')
            zone_gdf.plot(ax=ax4, 
                         color=colors[zone.zone_id-1], 
                         alpha=0.3,
                         edgecolor='black',
                         linewidth=1)
        
        # Add stations
        self.stations_df.plot(ax=ax4, 
                             color='black', 
                             markersize=100,
                             marker='*',
                             edgecolor='white',
                             linewidth=2)
        
        # Add labels
        for idx, station in self.stations_df.iterrows():
            ax4.annotate(station.station, 
                        (station.longitude, station.latitude),
                        xytext=(5, 5), 
                        textcoords='offset points',
                        fontsize=8,
                        weight='bold')
        
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Zone Coverage Areas')
        
        # 6. Zone summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary table
        summary_stats = []
        for zone in [1, 2, 3]:
            zone_data = self.zone_summary[self.zone_summary['zone_id'] == zone]
            summary_stats.append([
                f'Zone {zone}',
                f"{zone_data['total_vehicles_sum'].sum():,.0f}",
                f"{zone_data['weighted_total_mean'].mean():.1f}",
                f"{zone_data['inner_radius_km'].iloc[0]}-{zone_data['outer_radius_km'].iloc[0]} km"
            ])
        
        table = ax6.table(cellText=summary_stats,
                         colLabels=['Zone', 'Total Vehicles', 'Avg Impact', 'Distance Range'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2, 0.3, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(summary_stats) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#34495e')
                    cell.set_text_props(weight='bold', color='white')
                else:  # Data rows
                    cell.set_facecolor(colors[i-1])
                    cell.set_alpha(0.3)
        
        ax6.set_title('Zone Summary Statistics', fontsize=14, pad=20)
        
        plt.tight_layout()
        
    
        """Analyze and visualize zone effectiveness in capturing traffic impact"""
        
        fig = plt.figure(figsize=(18, 10))
        
        # 1. Impact capture efficiency by zone
        ax1 = plt.subplot(2, 3, 1)
        
        # Calculate total impact and percentage by zone
        total_impact = self.traffic_with_zones['weighted_total'].sum()
        zone_impacts = []
        
        for zone in [1, 2, 3]:
            zone_data = self.traffic_with_zones[self.traffic_with_zones['zone_id'] == zone]
            zone_impact = zone_data['weighted_total'].sum()
            zone_impacts.append({
                'Zone': f'Zone {zone}',
                'Impact': zone_impact,
                'Percentage': (zone_impact / total_impact) * 100,
                'Count': len(zone_data)
            })
        
        impact_df = pd.DataFrame(zone_impacts)
        
        # Create pie chart
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges, texts, autotexts = ax1.pie(impact_df['Percentage'], 
                                           labels=impact_df['Zone'],
                                           colors=colors,
                                           autopct='%1.1f%%',
                                           startangle=90)
        
        # Add count information
        for i, (wedge, count) in enumerate(zip(wedges, impact_df['Count'])):
            angle = (wedge.theta2 + wedge.theta1) / 2
            x = wedge.center[0] + 0.7 * np.cos(np.radians(angle))
            y = wedge.center[1] + 0.7 * np.sin(np.radians(angle))
            ax1.text(x, y, f'n={count}', ha='center', fontsize=10)
        
        ax1.set_title('Traffic Impact Distribution by Zone')
        
        # 2. Density vs Distance relationship
        ax2 = plt.subplot(2, 3, 2)
        
        # Calculate point density in distance bins
        distance_bins = np.linspace(0, 3, 30)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        densities = []
        avg_impacts = []
        
        for i in range(len(distance_bins) - 1):
            # Calculate area of ring
            inner_radius = distance_bins[i]
            outer_radius = distance_bins[i + 1]
            area = np.pi * (outer_radius**2 - inner_radius**2)
            
            # Count points in this distance range
            mask = (self.traffic_with_zones.geometry.apply(
                lambda g: self._calculate_min_station_distance(g)) >= inner_radius) & \
                   (self.traffic_with_zones.geometry.apply(
                lambda g: self._calculate_min_station_distance(g)) < outer_radius)
            
            count = mask.sum()
            density = count / area if area > 0 else 0
            densities.append(density)
            
            # Average impact in this bin
            if count > 0:
                avg_impacts.append(self.traffic_with_zones.loc[mask, 'weighted_total'].mean())
            else:
                avg_impacts.append(0)
        
        ax2_twin = ax2.twinx()
        
        # Plot density
        line1 = ax2.plot(bin_centers, densities, 'b-', linewidth=2, label='Point Density')
        ax2.set_xlabel('Distance from Station (km)')
        ax2.set_ylabel('Traffic Points per km²', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Plot average impact
        line2 = ax2_twin.plot(bin_centers, avg_impacts, 'r-', linewidth=2, label='Avg Impact')
        ax2_twin.set_ylabel('Average Weighted Impact', color='r')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        
        # Add zone boundaries
        for boundary, color in zip([0.5, 1.5, 3.0], colors):
            ax2.axvline(boundary, color=color, linestyle='--', alpha=0.5)
        
        ax2.set_title('Traffic Density and Impact vs Distance')
        ax2.grid(True, alpha=0.3)
        
        # 3. Zone overlap analysis
        ax3 = plt.subplot(2, 3, 3)
        
        # Calculate overlap between station zones
        overlap_matrix = np.zeros((len(self.stations_df), len(self.stations_df)))
        
        for i, station1 in enumerate(self.stations_df.station):
            for j, station2 in enumerate(self.stations_df.station):
                if i != j:
                    zone1 = self.buffer_zones[
                        (self.buffer_zones['station_name'] == station1) & 
                        (self.buffer_zones['zone_id'] == 3)
                    ].geometry.iloc[0]
                    zone2 = self.buffer_zones[
                        (self.buffer_zones['station_name'] == station2) & 
                        (self.buffer_zones['zone_id'] == 3)
                    ].geometry.iloc[0]
                    
                    intersection = zone1.intersection(zone2)
                    overlap_pct = (intersection.area / zone1.area) * 100
                    overlap_matrix[i, j] = overlap_pct
        
        im = ax3.imshow(overlap_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(len(self.stations_df)))
        ax3.set_yticks(range(len(self.stations_df)))
        ax3.set_xticklabels(self.stations_df.station, rotation=45, ha='right')
        ax3.set_yticklabels(self.stations_df.station)
        ax3.set_title('Zone Overlap Matrix (%)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Overlap Percentage')
        
        # Add text annotations
        for i in range(len(self.stations_df)):
            for j in range(len(self.stations_df)):
                text = ax3.text(j, i, f'{overlap_matrix[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 4. Weighted impact gradient
        ax4 = plt.subplot(2, 3, 4)
        
        # Create distance-based weight visualization
        distances = np.linspace(0, 3, 100)
        weights = []
        
        for d in distances:
            if d <= 0.5:
                w = 1.0
            elif d <= 1.5:
                w = 0.5
            elif d <= 3.0:
                w = 0.333
            else:
                w = 0
            weights.append(w)
        
        ax4.plot(distances, weights, linewidth=3, color='navy')
        ax4.fill_between(distances, weights, alpha=0.3)
        
        # Add zone labels
        ax4.axvspan(0, 0.5, alpha=0.2, color='#2ecc71', label='Zone 1')
        ax4.axvspan(0.5, 1.5, alpha=0.2, color='#f39c12', label='Zone 2')
        ax4.axvspan(1.5, 3.0, alpha=0.2, color='#e74c3c', label='Zone 3')
        
        ax4.set_xlabel('Distance from Station (km)')
        ax4.set_ylabel('Impact Weight')
        ax4.set_title('Distance-Based Weighting Function')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.1, 1.1)
        
        # 5. Station influence comparison
        ax5 = plt.subplot(2, 3, 5)
        
        # Calculate total weighted impact per station
        station_impacts = self.traffic_with_zones.groupby('station_name')[
            'weighted_total'
        ].sum().sort_values(ascending=True)
        
        y_pos = np.arange(len(station_impacts))
        ax5.barh(y_pos, station_impacts.values, color='steelblue')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(station_impacts.index)
        ax5.set_xlabel('Total Weighted Traffic Impact')
        ax5.set_title('Station Influence Comparison')
        
        # Add value labels
        for i, v in enumerate(station_impacts.values):
            ax5.text(v + station_impacts.max() * 0.01, i, f'{v:.0f}', 
                    va='center', fontsize=9)
        
        # 6. Zone efficiency metrics
        ax6 = plt.subplot(2, 3, 6)
        
        # Calculate efficiency metrics
        efficiency_data = []
        for zone in [1, 2, 3]:
            zone_data = self.zone_summary[self.zone_summary['zone_id'] == zone]
            
            # Average area per zone
            zone_areas = self.buffer_zones[self.buffer_zones['zone_id'] == zone]['geometry'].apply(
                lambda g: g.area * 111 * 111  # Convert to km²
            )
            
            efficiency_data.append({
                'Zone': f'Zone {zone}',
                'Avg Area (km²)': zone_areas.mean(),
                'Traffic Points': self.traffic_with_zones[
                      self.traffic_with_zones['zone_id'] == zone

                ].shape[0],
                'Avg Impact': zone_data['weighted_total_mean'].mean(),
                'Impact per km²': zone_data['weighted_total_sum'].sum() / zone_areas.sum()
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        # Create grouped bar chart
        x = np.arange(len(eff_df))
        width = 0.2
        
        # Normalize metrics for comparison
        metrics = ['Traffic Points', 'Avg Impact', 'Impact per km²']
        colors_metrics = ['#3498db', '#e74c3c', '#f39c12']
        
        for i, (metric, color) in enumerate(zip(metrics, colors_metrics)):
            values = eff_df[metric] / eff_df[metric].max()  # Normalize to 0-1
            bars = ax6.bar(x + i * width, values, width, 
                          label=metric, color=color, alpha=0.8)
            
            # Add actual values as text
            for j, (bar, actual) in enumerate(zip(bars, eff_df[metric])):
                height = bar.get_height()
                if metric == 'Avg Area (km²)':
                    text = f'{actual:.1f}'
                elif metric == 'Impact per km²':
                    text = f'{actual:.0f}'
                else:
                    text = f'{int(actual)}'
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02, text, ha='center', va='bottom', fontsize=8)
        
        ax6.set_xlabel('Zone')
        ax6.set_ylabel('Normalized Value')
        ax6.set_title('Zone Efficiency Metrics (Normalized)')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(eff_df['Zone'])
        ax6.legend()
        ax6.set_ylim(0, 1.2)
        
        plt.tight_layout()
        return fig
    
    def _calculate_min_station_distance(self, point):
        """Helper function to calculate minimum distance to any station"""
        min_dist = float('inf')
        for idx, station in self.stations_df.iterrows():
            dist = point.distance(station.geometry) * 111  # Convert to km
            min_dist = min(min_dist, dist)
        return min_dist
    