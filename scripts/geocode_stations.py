import time
import logging
import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from shapely.geometry import Point
import folium
import osmnx as ox

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def geocode_station(station, geolocator, city, region, country, delay=1.0, max_retries=3):
    """Attempt to geocode a station with multiple query formats and retries."""
    queries = [
        f"{station}, {city}, {region}, {country}",
        f"{station}, {city}",
        f"Via {station}, {city}",
        f"Stazione {station}, {city}"
    ]
    
    for attempt in range(max_retries):
        for query in queries:
            try:
                time.sleep(delay)  # Respect Nominatim rate limits
                location = geolocator.geocode(query)
                if location:
                    logging.info(f"Geocoded '{station}' as '{location.address}'")
                    return {
                        'station': station,
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'address': location.address
                    }
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                logging.warning(f"Timeout or service error on query '{query}': {e}")
                continue
    logging.error(f"Failed to geocode: {station}")
    return {'station': station, 'latitude': None, 'longitude': None, 'address': None}

def geocode_stations(station_names, city="Bologna", region="Emilia-Romagna", country="Italy"):
    """Geocode a list of station names into a GeoDataFrame."""
    geolocator = Nominatim(user_agent="air_quality_mapper", timeout=10)
    results = [geocode_station(st, geolocator, city, region, country) for st in station_names]
    
    df = pd.DataFrame(results)
    # Only keep rows with valid coordinates
    df_valid = df.dropna(subset=["latitude", "longitude"])
    gdf = gpd.GeoDataFrame(
        df_valid,
        geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
        crs="EPSG:4326"
    )
    return gdf

def create_interactive_map(gdf, base_map=ox.geocode_to_gdf('Bologna, Italy').explore(fill=False)):
    """Overlay station data on an existing folium map with improved styling."""
    # Add each station as a styled folium.Marker
    for _, row in gdf.iterrows():
        location = [row['latitude'], row['longitude']]
        popup_html = f"""
        <b>Station:</b> {row['station']}<br>
        <b>Address:</b> {row['address']}<br>
        <b>Lat:</b> {row['latitude']:.5f}<br>
        <b>Lon:</b> {row['longitude']:.5f}
        """
        
        folium.Marker(
            location=location,
            tooltip=row['station'],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(base_map)
    
    return base_map


