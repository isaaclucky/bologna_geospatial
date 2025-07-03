import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import Point
import numpy as np
from scipy.spatial import cKDTree
import folium
import matplotlib.pyplot as plt

class TrafficNetworkBuilder:
    """
    Builds unified traffic network from city sensors and regional MTS stations
    """
    
    def __init__(self, place_name="Bologna, Italy"):
        self.place_name = place_name
        self.osm_graph = None
        self.city_graph = nx.DiGraph()
        self.unified_graph = nx.DiGraph()
        self.node_mappings = {}
        
    def load_osm_network(self, network_type='drive', simplify=True):
        """Load OSM road network for Bologna"""
        print("Loading OSM network...")
        self.osm_graph = ox.graph_from_place(
            self.place_name, 
            network_type=network_type,
            simplify=simplify
        )
        # Convert to projected CRS for accurate distance calculations
        self.osm_graph = ox.project_graph(self.osm_graph)
        print(f"Loaded OSM network with {len(self.osm_graph.nodes)} nodes")
        return self.osm_graph
    
    def snap_points_to_osm(self, gdf_points, point_id_col):
        """
        Snap points to nearest OSM nodes
        Returns mapping of point_id -> osm_node_id
        """
        # Ensure projected CRS
        gdf_projected = gdf_points.to_crs(self.osm_graph.graph['crs'])
        
        # Get OSM node coordinates
        osm_nodes = pd.DataFrame(
            [(node, data['x'], data['y']) for node, data in self.osm_graph.nodes(data=True)],
            columns=['osm_node', 'x', 'y']
        )
        
        # Build KDTree for fast nearest neighbor search
        tree = cKDTree(osm_nodes[['x', 'y']].values)
        
        # Find nearest OSM node for each point
        point_coords = np.array([(p.x, p.y) for p in gdf_projected.geometry])
        distances, indices = tree.query(point_coords)
        
        # Create mapping
        mapping = {}
        for idx, (point_id, dist) in enumerate(zip(gdf_points[point_id_col], distances)):
            osm_node = osm_nodes.iloc[indices[idx]]['osm_node']
            mapping[point_id] = {
                'osm_node': osm_node,
                'snap_distance': dist
            }
            
        return mapping
    
    def build_city_network(self, df_directions, bol_gdf):
        """
        Build directed graph from city sensor data
        
        Parameters:
        - df_directions: DataFrame with columns ['id_uni', 'Node_da', 'Node_a', 'direction', 'Forward']
        - bol_gdf: GeoDataFrame with columns ['codice', 'geometry'] where codice matches Node_da/Node_a
        """
        print("Building city sensor network...")
        
        # Ensure bol_gdf is in WGS84
        if bol_gdf.crs != 'EPSG:4326':
            bol_gdf = bol_gdf.to_crs('EPSG:4326')
        
        # Snap city nodes to OSM using 'codice' as identifier
        city_mapping = self.snap_points_to_osm(bol_gdf, 'codice')
        self.node_mappings['city'] = city_mapping
        
        # Build directed graph from city data
        edges_added = 0
        for _, row in df_directions.iterrows():
            node_from = row['Node_da']
            node_to = row['Node_a']
            
            # Map to OSM nodes
            if node_from in city_mapping and node_to in city_mapping:
                osm_from = city_mapping[node_from]['osm_node']
                osm_to = city_mapping[node_to]['osm_node']
                
                # Interpret direction based on your description:
                # direction=1 means node_da -> node_a
                # direction=0 means node_a -> node_da (reverse)
                if row['direction'] == 1:
                    # Normal direction: from Node_da to Node_a
                    self.city_graph.add_edge(osm_from, osm_to, 
                                           city_id=row['id_uni'],
                                           forward=row['Forward'],
                                           direction='forward',
                                           original_from=node_from,
                                           original_to=node_to)
                else:
                    # Reverse direction: from Node_a to Node_da
                    self.city_graph.add_edge(osm_to, osm_from,
                                           city_id=row['id_uni'], 
                                           forward=row['Forward'],
                                           direction='reverse',
                                           original_from=node_to,
                                           original_to=node_from)
                edges_added += 1
        
        print(f"City network: {self.city_graph.number_of_nodes()} nodes, "
              f"{self.city_graph.number_of_edges()} edges")
        return self.city_graph
    
    def connect_regional_stations(self, df_regional, df_regional_gdf):
        """
        Connect regional MTS stations to city network via OSM paths
        
        Parameters:
        - df_regional: DataFrame with columns ['MTSStationID', 'DirectionCode']
        - df_regional_gdf: GeoDataFrame with columns ['NAME', 'LAT_WGS84', 'LONG_WGS84', 'geometry']
                          where NAME is the MTS station ID
        """
        print("Connecting regional stations...")
        
        # Ensure regional GDF is in WGS84
        if df_regional_gdf.crs != 'EPSG:4326':
            df_regional_gdf = df_regional_gdf.to_crs('EPSG:4326')
        
        # Create a copy with MTSStationID column for consistency
        regional_gdf_copy = df_regional_gdf.copy()
        regional_gdf_copy['MTSStationID'] = regional_gdf_copy['NAME']
        
        # Snap regional stations to OSM
        regional_mapping = self.snap_points_to_osm(regional_gdf_copy, 'MTSStationID')
        self.node_mappings['regional'] = regional_mapping
        
        # Find city network entry points (boundary nodes)
        city_nodes = set(self.city_graph.nodes())
        
        # For each regional station, find paths to city network
        connections = []
        for _, row in df_regional.iterrows():
            station_id = row['MTSStationID']
            direction_code = row['DirectionCode']
            
            if station_id not in regional_mapping:
                print(f"Warning: Station {station_id} not found in geodata")
                continue
                
            station_osm = regional_mapping[station_id]['osm_node']
            
            # Find shortest paths to city entry points
            paths_found = 0
            for city_node in list(city_nodes)[:10]:  # Check first 10 for efficiency
                try:
                    if direction_code == 0:  # Inbound to city
                        path = nx.shortest_path(
                            self.osm_graph, 
                            station_osm, 
                            city_node,
                            weight='length'
                        )
                    else:  # Outbound from city
                        path = nx.shortest_path(
                            self.osm_graph,
                            city_node,
                            station_osm, 
                            weight='length'
                        )
                    
                    connections.append({
                        'station_id': station_id,
                        'direction': direction_code,
                        'path': path,
                        'entry_node': city_node
                    })
                    paths_found += 1
                    
                    if paths_found >= 3:  # Find up to 3 connections per station
                        break
                        
                except nx.NetworkXNoPath:
                    continue
        
        print(f"Found {len(connections)} regional-city connections")
        return connections
    
    def build_unified_network(self, df_directions, bol_gdf, df_regional, df_regional_gdf):
        """
        Build complete unified network
        """
        # Load OSM base network
        self.load_osm_network()
        
        # Build city network
        self.build_city_network(df_directions, bol_gdf)
        
        # Connect regional stations
        connections = self.connect_regional_stations(df_regional, df_regional_gdf)
        
        # Create unified graph
        self.unified_graph = self.osm_graph.copy()
        
        # Add city sensor edges with attributes
        for u, v, data in self.city_graph.edges(data=True):
            self.unified_graph[u][v]['sensor_type'] = 'city'
            self.unified_graph[u][v]['sensor_id'] = data.get('city_id')
            self.unified_graph[u][v]['has_sensor'] = True
            self.unified_graph[u][v].update(data)
        
        # Add regional connection paths
        for conn in connections:
            path = conn['path']
            for i in range(len(path)-1):
                if conn['direction'] == 0:  # Inbound
                    if self.unified_graph.has_edge(path[i], path[i+1]):
                        self.unified_graph[path[i]][path[i+1]]['sensor_type'] = 'regional'
                        self.unified_graph[path[i]][path[i+1]]['station_id'] = conn['station_id']
                        self.unified_graph[path[i]][path[i+1]]['flow_direction'] = 'inbound'
                else:  # Outbound
                    if self.unified_graph.has_edge(path[i+1], path[i]):
                        self.unified_graph[path[i+1]][path[i]]['sensor_type'] = 'regional'
                        self.unified_graph[path[i+1]][path[i]]['station_id'] = conn['station_id']
                        self.unified_graph[path[i+1]][path[i]]['flow_direction'] = 'outbound'
        
        # Add node attributes for sensors
        for sensor_id, mapping in self.node_mappings['city'].items():
            osm_node = mapping['osm_node']
            self.unified_graph.nodes[osm_node]['city_sensor_id'] = sensor_id
            self.unified_graph.nodes[osm_node]['node_type'] = 'city_sensor'
            
        for station_id, mapping in self.node_mappings['regional'].items():
            osm_node = mapping['osm_node']
            self.unified_graph.nodes[osm_node]['mts_station_id'] = station_id
            self.unified_graph.nodes[osm_node]['node_type'] = 'regional_station'
        
        print(f"Unified network: {self.unified_graph.number_of_nodes()} nodes, "
              f"{self.unified_graph.number_of_edges()} edges")
        
        return self.unified_graph
    
    def export_network(self, filename="traffic_network.graphml"):
        """Export network for visualization/analysis"""
        ox.save_graphml(self.unified_graph, filename)
        print(f"Network saved to {filename}")
