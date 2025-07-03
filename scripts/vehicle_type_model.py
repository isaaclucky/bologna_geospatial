import psutil
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Optional, Union
import gc

class VehicleTypeModel:
    #########################
    # CONFIGURATION DEFAULTS
    #########################
    VEHICLE_PROFILE = {'VAN': 0.76, 'BOX_TRUCK': 0.16, 'TRUCK': 0.08}
    LTZ_UTILIZATION = { "<25": 0.67, "<50": 0.12, ">50": 0.21 }
    SCVLZ_STATS = { "total": 415, "practical": 154, "coverage_share": 0.214, "illegal_occupancy": 0.58 }
    OPTIMAL_SCVLZ_COVERAGE = 0.496

    HOLIDAYS = [
        '2024-01-01','2024-01-06','2024-03-19','2024-03-31','2024-04-01','2024-04-25',
        '2024-05-01','2024-05-12','2024-06-02','2024-08-15','2024-10-04',
        '2024-11-01','2024-12-08','2024-12-25','2024-12-26'
    ]

    def __init__(self, traffic_csv=None, meta_csv=None, landuse_gpkg=None, ltz_shp=None, nrows=None, validate=False):
        self.traffic_csv = traffic_csv
        self.meta_csv = meta_csv
        self.landuse_gpkg = landuse_gpkg
        self.ltz_shp = ltz_shp
        self.nrows = nrows
        self.validate = validate
        
        # Initialize only essential attributes
        self.metadata = None
        self.sensor_locations = None
        self.chunk_size = 100000  # Process data in chunks

    def load_metadata(self):
        """Load only metadata, not traffic data"""
        if self.meta_csv is None:
            raise ValueError("meta_csv must be provided.")
        
        # Read only necessary columns
        self.metadata = pd.read_csv(
            self.meta_csv, 
            sep=';',
            usecols=['id_uni', 'codice spira', 'codice via', 'longitudine', 'latitudine']
        )

    def process_geodata_efficiently(self):
        """Process geodata with memory optimization"""
        if self.landuse_gpkg is None and not hasattr(self, "landuse_gdf"):
            raise ValueError("Either landuse_gpkg or landuse_gdf must be provided.")
        if self.ltz_shp is None:
            raise ValueError("ltz_shp must be provided.")

        # Create sensors GeoDataFrame
        sensors_gdf = gpd.GeoDataFrame(
            self.metadata[['id_uni']],
            geometry=gpd.points_from_xy(self.metadata['longitudine'], self.metadata['latitudine']),
            crs="EPSG:4326"
        ).to_crs("EPSG:32632")
        
        # Process landuse assignment in batches
        sensor_locations = {}
        batch_size = 1000
        
        for i in range(0, len(sensors_gdf), batch_size):
            batch = sensors_gdf.iloc[i:i+batch_size].copy()
            
            # Create buffer for batch only
            batch['geometry'] = batch.geometry.buffer(50)
            
            # Load landuse data
            if hasattr(self, "landuse_gdf"):
                land = self.landuse_gdf.to_crs("EPSG:32632")
            else:
                land = gpd.read_file(
                    self.landuse_gpkg, 
                    layer="landuse_bologna"
                ).to_crs("EPSG:32632")
                land.dropna(subset=['landuse'], inplace=True)
            
            # Perform spatial join
            thematic_cols = [c for c in ["landuse", "depot", "industrial", "construction", 
                                        "meadow", "military", "residential"] if c in land.columns]
            
            joined = gpd.sjoin(
                batch[['id_uni', 'geometry']],
                land[["geometry"] + thematic_cols],
                how='left',
                predicate='intersects'
            )
            
            # Process joined data
            for id_uni in batch['id_uni'].unique():
                sensor_data = joined[joined['id_uni'] == id_uni]
                if len(sensor_data) > 0:
                    # Get dominant landuse
                    dominant = sensor_data.iloc[0]
                    thematic = self._get_thematic_landuse(dominant)
                    
                    sensor_locations[id_uni] = {
                        'thematic_landuse': thematic,
                        'dist_commercial': self._calc_dist(thematic),
                        'near_loading_zone': self._near_loading(thematic),
                        'in_ltz': False  # Will be updated next
                    }
            
            # Clean up batch memory
            del batch
            del joined
            if not hasattr(self, "landuse_gdf"):
                del land
            gc.collect()
        
        # Process LTZ zones
        ltz_zones = gpd.read_file(self.ltz_shp).to_crs("EPSG:32632")
        sensors_in_ltz = gpd.sjoin(
            sensors_gdf[['id_uni', 'geometry']],
            ltz_zones[['geometry']],
            how='left',
            predicate='intersects'
        )
        
        # Update LTZ flags
        for id_uni in sensors_in_ltz[~sensors_in_ltz.index_right.isnull()]['id_uni']:
            if id_uni in sensor_locations:
                sensor_locations[id_uni]['in_ltz'] = True
        
        self.sensor_locations = sensor_locations
        
        # Clean up
        del sensors_gdf
        del ltz_zones
        gc.collect()

    def _get_thematic_landuse(self, row):
        """Extract thematic landuse from row"""
        keywords = ["depot", "industrial", "construction", "meadow", "military", "residential", "landuse"]
        for col in keywords:
            val = row.get(col, None)
            if pd.notnull(val) and str(val).lower() not in ("nan", "none"):
                return f"{col}:{val}" if col != "landuse" else val
        return "unknown"

    def _calc_dist(self, thematic):
        """Calculate distance based on thematic landuse"""
        thematic_ = str(thematic or "")
        if any(x in thematic_ for x in ['commercial', 'retail']):
            return 0
        elif any(x in thematic_ for x in ['mixed', 'residential']):
            return 75
        else:
            return 150

    def _near_loading(self, thematic):
        """Check if near loading zone"""
        thematic_ = str(thematic or "")
        return any(x in thematic_ for x in [
            'commercial', 'retail', 'industrial', 'construction', 'depot'
        ])

    def process_traffic_data_chunked(self, output_file: str):
        """Process traffic data in chunks to save memory"""
        if self.traffic_csv is None:
            raise ValueError("traffic_csv must be provided.")
        
        # Process in chunks
        chunk_iter = pd.read_csv(
            self.traffic_csv, 
            sep=';', 
            parse_dates=['datetime'],
            chunksize=self.chunk_size,
            nrows=self.nrows
        )
        
        first_chunk = True
        for chunk in chunk_iter:
            # Add weekend and holiday flags
            chunk['is_weekend'] = chunk['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
            chunk['is_holiday'] = chunk['datetime'].dt.date.astype(str).isin(self.HOLIDAYS)
            
            # Process chunk
            results = self._process_chunk(chunk)
            
            # Save results
            if first_chunk:
                results.to_csv(output_file, index=False, mode='w')
                first_chunk = False
            else:
                results.to_csv(output_file, index=False, mode='a', header=False)
            
            # Clean up
            del chunk
            del results
            gc.collect()
        
        print(f"Results saved to {output_file}")

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a single chunk of data"""
        # Vectorized operations where possible
        results = []
        
        for _, row in chunk.iterrows():
            result = self.interpolate_row(row)
            results.append(result)
        
        return pd.DataFrame(results)

    def interpolate_row(self, row):
        """Interpolate vehicle types for a single row"""
        hour = row['datetime'].hour
        total_count = row.get('traffic', row.get('total_vehicles', 0))
        sensor = self.sensor_locations.get(row['id_uni'], {})
        in_ltz = sensor.get('in_ltz', False)
        near_loading = sensor.get('near_loading_zone', False)
        is_weekend = 'weekday' if row['is_weekend'] == 0 else 'weekend'
        
        prop = self.adjust_proportions(hour, is_weekend, in_ltz, near_loading)
        
        if row.get('is_holiday', False):
            prop['BOX_TRUCK'] *= 0.25
            prop['TRUCK'] *= 0.15
            prop['VAN'] = 1.0 - (prop['BOX_TRUCK'] + prop['TRUCK'])
        
        van_n = int(round(total_count * prop['VAN']))
        box_n = int(round(total_count * prop['BOX_TRUCK']))
        truck_n = int(round(total_count * prop['TRUCK']))
        
        if van_n + box_n + truck_n < total_count:
            van_n += (total_count - (van_n + box_n + truck_n))
        
        return {
            'datetime': row['datetime'],
            'id_uni': row['id_uni'],
            'Level': row.get('Livello'),
            'Node_da': row.get('Nodo da'),
            'Node_a': row.get('Nodo a'),
            'direction': row.get('direzione'),
            'angle': row.get('angolo'),
            'Light_Count': van_n,
            'Medium_Count': box_n,
            'Heavy_Count': truck_n,
            'proportion_VAN': prop['VAN'],
            'proportion_BOX_TRUCK': prop['BOX_TRUCK'],
            'proportion_TRUCK': prop['TRUCK'],
        }

    def adjust_proportions(self, hour, is_weekend, in_ltz, near_loading_zone):
        """Adjust vehicle proportions based on context"""
        base = self.VEHICLE_PROFILE.copy()
        
        if 7 <= hour <= 10:
            if is_weekend == 'weekday':
                if not in_ltz:
                    base['BOX_TRUCK'] = min(base['BOX_TRUCK'] * 1.30, 0.25)
                    base['TRUCK'] = min(base['TRUCK'] * 1.25, 0.12)
            else:
                base['BOX_TRUCK'] *= 0.7
                base['TRUCK'] *= 0.6
        elif 18 <= hour <= 20:
            base['BOX_TRUCK'] *= 0.6
            base['TRUCK'] *= 0.5
            
        if near_loading_zone:
            base['BOX_TRUCK'] *= 1.15
            base['TRUCK'] *= 1.20
            
        if in_ltz:
            if not (8 <= hour <= 18):
                base['BOX_TRUCK'] *= 0.3
                base['TRUCK'] *= 0.2
            base = self.VEHICLE_PROFILE.copy()
            
        s = sum(base.values())
        for k in base: 
            base[k] /= s
            
        return base

    def run_full_pipeline(self, output_file: str):
        """Run the complete pipeline with memory optimization"""
        print("Loading metadata...")
        self.load_metadata()
        
        print("Processing geodata...")
        self.process_geodata_efficiently()
        
        print("Processing traffic data in chunks...")
        self.process_traffic_data_chunked(output_file)
        
        if self.validate:
            print("Validating results...")
            self.validate_results_from_file(output_file)
        
        print("Processing complete!")

    def validate_results_from_file(self, filename: str):
        """Validate results by reading from file in chunks"""
        van_sum = 0
        box_sum = 0
        truck_sum = 0
        count = 0
        
        for chunk in pd.read_csv(filename, chunksize=self.chunk_size):
            van_sum += chunk['proportion_VAN'].sum()
            box_sum += chunk['proportion_BOX_TRUCK'].sum()
            truck_sum += chunk['proportion_TRUCK'].sum()
            count += len(chunk)
        
        van_mean = van_sum / count
        box_mean = box_sum / count
        truck_mean = truck_sum / count
        
        assert 0.74 <= van_mean <= 0.78, f"Unexpected van share: {van_mean:.2f}"
        assert 0.14 <= box_mean <= 0.18, f"Unexpected box share: {box_mean:.2f}"
        assert 0.07 <= truck_mean <= 0.09, f"Unexpected truck share: {truck_mean:.2f}"
        
        print("Validation passed!")
        print(f"Average proportions - VAN: {van_mean:.3f}, BOX_TRUCK: {box_mean:.3f}, TRUCK: {truck_mean:.3f}")

    # Additional utility functions for memory-efficient operations

    def process_large_geodata_with_spatial_index(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                                                batch_size: int = 1000) -> gpd.GeoDataFrame:
        """
        Perform spatial join with spatial indexing for better performance
        """
        # Create spatial index on the larger dataset
        gdf2_sindex = gdf2.sindex
        
        results = []
        
        for i in range(0, len(gdf1), batch_size):
            batch = gdf1.iloc[i:i+batch_size]
            
            # Use spatial index to find potential matches
            for idx, row in batch.iterrows():
                possible_matches_index = list(gdf2_sindex.intersection(row.geometry.bounds))
                possible_matches = gdf2.iloc[possible_matches_index]
                
                # Perform actual intersection test
                precise_matches = possible_matches[possible_matches.intersects(row.geometry)]
                
                for _, match in precise_matches.iterrows():
                    result_row = row.to_dict()
                    result_row.update(match.to_dict())
                    results.append(result_row)
        
        return gpd.GeoDataFrame(results)


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
