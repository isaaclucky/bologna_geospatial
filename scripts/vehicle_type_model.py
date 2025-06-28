import pandas as pd
import numpy as np
import geopandas as gpd

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

    def __init__(self, traffic_csv=None, meta_csv=None, landuse_gpkg=None, ltz_shp=None, nrows=50000):
        self.traffic_csv = traffic_csv
        self.meta_csv = meta_csv
        self.landuse_gpkg = landuse_gpkg
        self.ltz_shp = ltz_shp

        self.city_data = None
        self.metadata = None
        self.results = None
        self.land = None
        self.ltz_zones = None
        self.sensors_gdf = None
        self.buffered_gdf = None
        self.dominant_landuse = None
        self.sensor_locations = None
        self.nrows = 50000 
        

    def load_data(self):
        if self.traffic_csv is None or self.meta_csv is None:
            raise ValueError("traffic_csv and meta_csv must be provided.")

        self.city_data = pd.read_csv(self.traffic_csv, sep=';', parse_dates=['datetime'], nrows=50000)
        self.metadata = pd.read_csv(self.meta_csv, sep=';')

        self.city_data['is_weekend'] = self.city_data['datetime'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        self.city_data['is_holiday'] = self.city_data['datetime'].dt.date.astype(str).isin(self.HOLIDAYS)

    def load_geodata(self):
        if self.landuse_gpkg is None or self.ltz_shp is None:
            raise ValueError("landuse_gpkg and ltz_shp must be provided.")

        self.land = gpd.read_file(self.landuse_gpkg, layer="multipolygons").to_crs("EPSG:32632")
        self.land.dropna(subset=['landuse'], inplace=True)
        self.ltz_zones = gpd.read_file(self.ltz_shp).to_crs("EPSG:32632")

        self.sensors_gdf = gpd.GeoDataFrame(
            self.metadata[['id_uni', 'codice spira', 'codice via']],
            geometry=gpd.points_from_xy(self.metadata['longitudine'], self.metadata['latitudine']),
            crs="EPSG:4326"
        ).to_crs("EPSG:32632")
        self.sensors_gdf['buffered'] = self.sensors_gdf.geometry.buffer(50)

        self.buffered_gdf = gpd.GeoDataFrame(self.sensors_gdf, geometry='buffered', crs="EPSG:32632")
        self.buffered_gdf.drop(columns=['geometry'], inplace=True)
        self.buffered_gdf.rename(columns={'buffered': 'geometry'}, inplace=True)
        self.buffered_gdf.set_geometry('geometry', inplace=True)

    def assign_landuse(self):
        joined = gpd.sjoin(
            self.buffered_gdf[['id_uni', 'geometry']],
            self.land[['geometry', 'landuse']],
            how='left',
            predicate='intersects'
        )
        joined['original_geometry'] = joined['geometry'].copy()
        joined['intersection_area'] = joined.apply(
            lambda row: row['geometry'].intersection(row['original_geometry']).area, axis=1)
        dominant_landuse = (
            joined.sort_values('intersection_area', ascending=False)
                  .drop_duplicates(subset=['id_uni'])
                  [['id_uni', 'landuse']]
        )
        all_sensors = self.buffered_gdf[['id_uni']].drop_duplicates()
        dominant_landuse = all_sensors.merge(dominant_landuse, on='id_uni', how='left')
        dominant_landuse['landuse'] = dominant_landuse['landuse'].fillna('unknown')

        # Official LTZ flagging
        sensors_in_ltz = gpd.sjoin(
            self.sensors_gdf[['id_uni', 'geometry']],
            self.ltz_zones[['geometry']],
            how='left',
            predicate='intersects'
        )
        sensors_ltz_flags = sensors_in_ltz[['id_uni']].copy()
        sensors_ltz_flags['in_ltz_official'] = ~sensors_in_ltz.index_right.isnull()
        dominant_landuse = dominant_landuse.merge(sensors_ltz_flags, on='id_uni', how='left')
        dominant_landuse['in_ltz_official'] = dominant_landuse['in_ltz_official'].fillna(False)

        self.dominant_landuse = dominant_landuse

    def context_functions(self):
        def calc_dist(landuse):
            if landuse in ['commercial', 'retail']:
                return 0
            elif landuse in ['mixed', 'residential']:
                return 75
            else:
                return 150
        def near_loading(landuse):
            return landuse in ['commercial', 'retail', 'industrial', 'construction']

        sensor_locations = {}
        for _, row in self.dominant_landuse.iterrows():
            sensor_locations[row['id_uni']] = {
                'landuse': row['landuse'],
                'dist_commercial': calc_dist(row['landuse']),
                'near_loading_zone': near_loading(row['landuse']),
                'in_ltz': bool(row['in_ltz_official'])
            }
        self.sensor_locations = sensor_locations

    #########################
    # VEHICLE SHARE LOGIC
    #########################

    def adjust_proportions(self, hour, is_weekend, in_ltz, near_loading_zone):
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
        for k in base: base[k] /= s
        return base

    def interpolate_row(self, row):
        hour = row['datetime'].hour
        total_count = row['traffic'] if 'traffic' in row else row['total_vehicles']
        sensor = self.sensor_locations.get(row['id_uni'], {})
        landuse = sensor.get('landuse', 'unknown')
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
        out = {
            'datetime': row['datetime'],
            'id_uni': row['id_uni'],
            'Level': row['Livello'],
            'VAN_count': van_n,
            'BOX_TRUCK_count': box_n,
            'TRUCK_count': truck_n,
            'proportion_VAN': prop['VAN'],
            'proportion_BOX_TRUCK': prop['BOX_TRUCK'],
            'proportion_TRUCK': prop['TRUCK'],
        }
        return out

    def run_interpolation(self, city_data=None):
        if city_data is not None:
            self.city_data = city_data
        self.results = self.city_data.apply(lambda row: pd.Series(self.interpolate_row(row)), axis=1)
        return self.results

    def validate_data(self, df=None):
        if df is None:
            df = self.results
        van_mean = df['proportion_VAN'].mean()
        box_mean = df['proportion_BOX_TRUCK'].mean()
        truck_mean = df['proportion_TRUCK'].mean()
        assert 0.74 <= van_mean <= 0.78, f"Unexpected van share: {van_mean:.2f}"
        assert 0.14 <= box_mean <= 0.18, f"Unexpected box share: {box_mean:.2f}"
        assert 0.07 <= truck_mean <= 0.09, f"Unexpected truck share: {truck_mean:.2f}"
        if 'LTZ' in df.columns:
            ltz = df[df['LTZ']]
            non_ltz = df[~df['LTZ']]
            assert ltz['proportion_VAN'].mean() > non_ltz['proportion_VAN'].mean()
            assert ltz['proportion_TRUCK'].mean() < non_ltz['proportion_TRUCK'].mean()
        print("Validation passed!")

    def save_results(self, filename):
        self.results.to_csv(filename, index=False)
        print(f"Saved results to {filename}")

    #########################
    # COMPLETE PIPELINE
    #########################

    def run_full_pipeline(self):
        self.load_data()
        self.load_geodata()
        self.assign_landuse()
        self.context_functions()
        self.run_interpolation()
        self.validate_data()
        print("Processing complete!")
