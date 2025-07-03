import numpy as np

def bearing_to_direction(bearing):
    bearing = bearing % 360
    directions = [
        (0, 22.5, "N"), (22.5, 67.5, "NE"), (67.5, 112.5, "E"), (112.5, 157.5, "SE"),
        (157.5, 202.5, "S"), (202.5, 247.5, "SO"), (247.5, 292.5, "O"), (292.5, 337.5, "NO"), (337.5, 360, "N")
    ]
    for min_angle, max_angle, direction in directions:
        if min_angle <= bearing < max_angle:
            return direction
    return "N"

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360

# Calculate forward direction
def get_forward_direction(row, gdf):
    # Get coordinates from gdf
    node_da_geom = gdf.loc[gdf['codice'] == row['Node_da'], 'geometry'].iloc[0]
    node_a_geom = gdf.loc[gdf['codice'] == row['Node_a'], 'geometry'].iloc[0]

    # Calculate bearing from node_da to node_a
    bearing = calculate_bearing(node_da_geom.y, node_da_geom.x, node_a_geom.y, node_a_geom.x)
    calculated_direction = bearing_to_direction(bearing)
    
    # Compare with given direction
    return 1 if calculated_direction == row['direction'] else 0

