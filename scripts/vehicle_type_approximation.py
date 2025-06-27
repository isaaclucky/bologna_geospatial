import pandas as pd
import numpy as np
from datetime import datetime

# Base proportions from Bologna freight study (https://www.researchgate.net/publication/248607503_Urban_freight_transport_in_Bologna_Planning_commercial_vehicle_loadingunloading_zones)
VEHICLE_PROFILE = {
    'LEGGERO': 0.76,  # Vans (<3.5t)
    'PESANTE': 0.24    # Box Trucks (16%) + Trucks (8%)
}

def location_factor(sensor_data: dict, hour: int) -> float:
    """Calculate PESANTE probability boost based on location"""
    factors = 1.0
    
    # Commercial district proximity
    if sensor_data.get('dist_commercial', float('inf')) < 50:
        factors *= 1.30
    
    # Historic center (LTZ) access rules
    if sensor_data.get('in_ltz', False):
        factors *= 1.25 if 8 <= hour <= 18 else 0.10
    
    # Near loading zones
    if sensor_data.get('near_loading_zone', False):
        factors *= 1.40
    
    return min(factors, 2.0)  # Cap at 200% increase

def adjust_proportions(hour: int, day_type: str) -> dict:
    """Refine vehicle proportions based on time patterns"""
    base = VEHICLE_PROFILE.copy()
    
    # Morning freight peak (7-10 AM)
    if 7 <= hour <= 10:
        if day_type == 'weekday':
            base['PESANTE'] = min(base['PESANTE'] * 1.25, 0.40)
        else:
            base['PESANTE'] *= 0.75
    
    # Evening restrictions (6-8 PM)
    elif 18 <= hour <= 20:
        base['PESANTE'] *= 0.60
    
    base['LEGGERO'] = 1 - base['PESANTE']
    return base

def interpolate_vehicle_types(row, sensor_locations):
    """Main interpolation function"""
    # Extract temporal features
    hour = row['datetime'].hour
    total_count = row['total_vehicles']
    
    # Get location data
    loc_data = sensor_locations.get(row['sensor_id'], {})
    
    # Get base profile
    profile = adjust_proportions(hour, row['day_type'])
    
    # Apply location factor to heavy vehicles
    loc_factor = location_factor(loc_data, hour)
    pesante_adj = min(profile['PESANTE'] * loc_factor, 0.65)  # Max 65% heavy
    
    # Holiday override
    if row.get('is_holiday', False):
        pesante_adj *= 0.25
    
    # Calculate counts
    return {
        'LEGGERO_count': int(total_count * (1 - pesante_adj)),
        'PESANTE_count': int(total_count * pesante_adj),
        'proportion_PESANTE': pesante_adj
    }

# Load your city traffic data (example structure)
# Columns needed: datetime, total_vehicles, sensor_id
city_data = pd.read_csv('bologna_traffic.csv')
city_data['datetime'] = pd.to_datetime(city_data['datetime'])

# Add day type
city_data['day_type'] = city_data['datetime'].dt.dayofweek.apply(
    lambda x: 'weekend' if x >= 5 else 'weekday'
)

# Add holiday flag (example)
holidays = ['2023-01-01', '2023-12-25']  # Your holiday dates
city_data['is_holiday'] = city_data['datetime'].dt.date.astype(str).isin(holidays)

# Sensor location data (should be preloaded)
# Format: {sensor_id: {'in_ltz': bool, 'dist_commercial': float, 'near_loading_zone': bool}}
sensor_locations = {
    123: {'in_ltz': True, 'dist_commercial': 35.2, 'near_loading_zone': True},
    456: {'in_ltz': False, 'dist_commercial': 120.5, 'near_loading_zone': False},
    # ... other sensors
}

# Apply interpolation
vehicle_cols = ['LEGGERO_count', 'PESANTE_count', 'proportion_PESANTE']
city_data[vehicle_cols] = city_data.apply(
    lambda row: pd.Series(interpolate_vehicle_types(row, sensor_locations)),
    axis=1
)

# Validation checks
def validate_data(df):
    # Temporal consistency
    morning = df[(df['datetime'].dt.hour.between(7,10)) & (df['day_type']=='weekday')]
    avg_morning = morning['proportion_PESANTE'].mean()
    assert 0.25 <= avg_morning <= 0.40, f"Invalid morning profile: {avg_morning:.2f}"
    
    # Spatial consistency (if LTZ data available)
    if 'in_ltz' in df.columns:
        ltz = df[df['in_ltz']]
        non_ltz = df[~df['in_ltz']]
        assert ltz['proportion_PESANTE'].mean() > non_ltz['proportion_PESANTE'].mean()
    
    # Total counts
    total_reconstructed = df['LEGGERO_count'] + df['PESANTE_count']
    mismatch = (abs(total_reconstructed - df['total_vehicles']) > 1).sum()
    assert mismatch == 0, f"Count mismatch in {mismatch} rows"
    
    print("Validation passed!")

# Run validation
validate_data(city_data)

# Feature engineering for ML
city_data['freight_activity_index'] = (
    city_data['PESANTE_count'] * 
    city_data['proportion_PESANTE'] * 
    np.where(city_data['datetime'].dt.hour.between(7,10), 1.0, 0.5)
)

# Save results
city_data.to_csv('bologna_traffic_with_vehicle_types.csv', index=False)
print("Processing complete!")