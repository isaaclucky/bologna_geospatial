import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box, Polygon
import osmnx as ox
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union

def rename_traffic_columns(df):
    """Rename Italian traffic dataset columns to English."""

    """
    YEAR_ID = survey year in format “aaaa”
    MONTH_ID = survey month in format “aaaamm”
    GIO_ID = survey day in format “aaaammgg”
    HHMI_ID = aggregation time per quarter hour in the format “hh: mm”
    APP_ID = MTS station number
    DIRMAR_COD = direction of travel, can take on the value 0 or 1
    VEI_ID = type of vehicle, can take the value from 0 to 10, the description of the type is present in the Vehicle Classes file
    NUM_TRANSITS = number of aggregate transits per quarter of an hour of the registered vehicle type

    """
    df = df.rename(columns={
        'ANNO_ID': 'Year',
        'MESE_ID': 'Month',
        'GIO_ID': 'Day',
        'HHMI_ID': 'HourMinute',
        'APP_ID': 'MTSStationID',
        'DIRMAR_COD': 'DirectionCode',
        'VEI_ID': 'VehicleType',
        'NUM_TRANSITI': 'TransitCount'
    })
    return df


def download_and_load_data_full_year(year, month, downloader):
    """    Downloads and loads traffic data for a full year.
    Args:
        year (int): The year for which to download the data.
    Returns:
        pd.DataFrame: DataFrame containing the traffic data for the specified year.
    """
    df = pd.DataFrame()
    if month == 0:
        for m in range(1, 13):
            monthly_data = downloader.download_and_load_data(year=year, month=m)
            df = pd.concat([df, monthly_data], ignore_index=True)
    else:
        monthly_data = downloader.download_and_load_data(year=year, month=month)
        df = pd.concat([df, monthly_data], ignore_index=True)
    df = rename_traffic_columns(df)
    
    return df


class GeoStationFilter:
    """
    Handles geospatial station data filtering and visualization.
    """

    def __init__(self, excel_file: str, lat_col: str = 'LAT_WGS84', lon_col: str = 'LONG_WGS84'):
        """
        Initialize with Excel file and convert to a GeoDataFrame.
        """
        # Read Excel file
        df = pd.read_excel(excel_file)
        required = {lat_col, lon_col}
        if not required.issubset(df.columns):
            raise ValueError(f"Input file must contain {required}")
        # Create geometry column
        geometry = [Point(x, y) for x, y in zip(df[lon_col], df[lat_col])]
        self.gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        print(f"Loaded {len(self.gdf)} stations from {excel_file!r}.")

    def _get_place_geom(self, place_name: str) -> Optional[gpd.GeoDataFrame]:
        """
        Geocode place using OSMnx and return GeoDataFrame, or None if failed.
        """
        try:
            place_gdf = ox.geocode_to_gdf(place_name)
            place_gdf = place_gdf.to_crs(self.gdf.crs)
            return place_gdf
        except Exception as e:
            print(f"Could not geocode {place_name!r}: {e}")
            return None

    def filter_by_place(self, place_name: str, buffer_km: float = 0) -> gpd.GeoDataFrame:
        """
        Filter stations by a named place using OSMnx's place boundary.
        """
        place_gdf = self._get_place_geom(place_name)
        if place_gdf is None:
            return self.gdf.iloc[[]]  # Return empty GeoDataFrame
        if buffer_km > 0:
            # Buffer the place geometry
            place_proj = place_gdf.to_crs(3857)
            buffer_geom_proj = place_proj.geometry.buffer(buffer_km * 1000)
            buffer_gdf_proj = gpd.GeoDataFrame(geometry=buffer_geom_proj, crs=3857)
            place_gdf = buffer_gdf_proj.to_crs(self.gdf.crs)

        stations_in_place = gpd.sjoin(self.gdf, place_gdf, how='inner', predicate='within')
        return stations_in_place.drop(columns=['index_right'])

    def filter_by_geometry(self, geom: Union[Polygon, Point], buffer_km: Optional[float] = None) -> gpd.GeoDataFrame:
        """
        Filter stations by arbitrary Shapely geometry or point+buffer.
        """
        if buffer_km and isinstance(geom, Point):
            # Buffer in meters: use projected crs for metric accuracy
            gdf_proj = self.gdf.to_crs('EPSG:3857')
            geom_proj = gdf_proj.crs
            point_proj = gpd.GeoSeries([geom], crs='EPSG:4326').to_crs(geom_proj).iloc[0]
            buffer_geom = point_proj.buffer(buffer_km * 1000)
            geom = buffer_geom
            result = gdf_proj[gdf_proj.within(geom)]
            return result.to_crs('EPSG:4326')
        else:
            return self.gdf[self.gdf.within(geom)]

    def filter_by_bbox(self, bounds: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
        """
        Filter stations by bounding box (west, south, east, north).
        """
        bbox_geom = box(*bounds)
        return self.filter_by_geometry(bbox_geom)

    def filter_by_buffer(self, center_point: Tuple[float, float], radius_km: float) -> gpd.GeoDataFrame:
        """
        Filter stations within a radius (km) from a given (lon, lat) point.
        """
        point = Point(center_point)
        return self.filter_by_geometry(point, buffer_km=radius_km)

    def visualize(self, filtered_gdf: Optional[gpd.GeoDataFrame] = None, place_name: Optional[str] = None,
                  export_path: Optional[str] = None, show: bool = True) -> None:
        """
        Visualize all and optionally filtered stations on a map.
        Optionally export plot to file.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        self.gdf.plot(ax=ax, color='lightblue', markersize=20, alpha=0.6, label='All stations')
        if filtered_gdf is not None and not filtered_gdf.empty:
            filtered_gdf.plot(ax=ax, color='red', markersize=40, label='Filtered stations')
        if place_name:
            place_gdf = self._get_place_geom(place_name)
            if place_gdf is not None:
                place_gdf.boundary.plot(ax=ax, edgecolor='green', linewidth=2, label=place_name)
        ax.legend()
        ax.set_title('Monitoring Stations')
        if export_path:
            plt.savefig(export_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)

    def visualize_stations(self, place_name: str, filter_name: str, export_path: Optional[str] = None) -> None:
        """
        Visualize stations in a region and a subregion, with boundaries,
        color-coding, and counts shown.
        """
        region_stations = self.filter_by_place(place_name)
        print(f"Found {len(region_stations)} stations in {place_name}")
        subregion_stations = self.filter_by_place(filter_name)
        print(f"Found {len(subregion_stations)} stations in {filter_name}")
        region_gdf = self._get_place_geom(place_name)
        filter_gdf = self._get_place_geom(filter_name)
        fig, ax = plt.subplots(figsize=(12, 10))
        if region_gdf is not None:
            region_gdf.boundary.plot(ax=ax, color='blue', linewidth=2, alpha=0.7, label='Main Region')
        if filter_gdf is not None:
            filter_gdf.plot(ax=ax, color='lightblue', alpha=0.3, edgecolor='red', linewidth=2, label='Filter Area')
        if not subregion_stations.empty:
            subregion_stations.plot(ax=ax, color='darkgreen', markersize=50, alpha=0.8, edgecolor='black', label='Stations')
        plt.title(f"{len(subregion_stations)} stations in {filter_name}\nwithin {place_name}")
        plt.legend()
        if export_path:
            plt.savefig(export_path, bbox_inches='tight')
        plt.show()
        plt.close(fig)
    def show_map(self, region_name: str, city_name: str):
        """
        Interactive map with region, city boundaries, and station locations using only GeoDataFrame.explore.

        Args:
            region_name: e.g. "Emilia-Romagna, Italy"
            city_name: e.g. "Bologna"

        Returns:
            The map object created by GeoDataFrame.explore() (renders in Jupyter notebooks).
        """

        # Get city boundary
        city_gdf = self._get_place_geom(city_name)
        if city_gdf is None or city_gdf.empty:
            raise ValueError(f"Could not fetch city boundary for '{city_name}'.")

        # Get region boundary
        region_gdf = self._get_place_geom(region_name)

        # Start interactive map with all stations
        map_out = self.gdf.explore(
            color="black",
            weight=10,
            fill=True,
            marker_kwds={"fill": True, "weight": 10, "color": "black"},
            tooltip=["NAME", "LAT_WGS84", "LONG_WGS84"],
            name="All stations"
        )

        # Add region boundary
        if region_gdf is not None and not region_gdf.empty:
            region_gdf.explore(
                m=map_out,
                color="#1E90FF",
                style_kwds=dict(weight=2, fill=False),
                tooltip="name",
                name=f"Region: {region_name}"
            )
        # Add city boundary
        city_gdf.explore(
            m=map_out,
            color="#FF4500",
            style_kwds=dict(weight=3, fill=False, dashArray='2,2'),
            tooltip="name",
            name=f"City: {city_name}"
        )

        return map_out
    
    def show_buffer_map(self, region_name: str, city_name: str, buffer_km: float = 2):
        """
        Interactive Leaflet map with city boundary, optional region boundary,
        a buffer (radius buffer_km) around the city, all stations, and 
        stations within buffer highlighted.

        Args:
            city_name: Name of the city (e.g., "Bologna")
            buffer_km: Radius of buffer in kilometers
            region_name: Optional region (for context)
        Returns:
            folium.Map object via gdf.explore
        """
        # Get city boundary
        city_gdf = self._get_place_geom(city_name)
        if city_gdf is None or city_gdf.empty:
            raise ValueError(f"Could not fetch city boundary for '{city_name}'.")

        # Buffer the city polygon (in projected CRS for meters)
        city_proj = city_gdf.to_crs(3857)
        buffer_geom_proj = city_proj.geometry.buffer(buffer_km * 1000)
        buffer_gdf_proj = gpd.GeoDataFrame(geometry=buffer_geom_proj, crs=3857)
        buffer_gdf = buffer_gdf_proj.to_crs(self.gdf.crs)

        # Find stations within buffer
        stations_in_buffer = self.gdf[self.gdf.within(buffer_gdf.iloc[0].geometry)]
        print(f"Found {len(stations_in_buffer)} stations with the buffer zone of {buffer_km}km around {city_name}.")

        # Start with all stations (black)
        m = self.gdf.explore(
            color="black",
            marker_kwds={"radius": 5, "fillOpacity": 0.8, "color": "black"},
            tooltip=["NAME", "LAT_WGS84", "LONG_WGS84"],
            name="All stations"
        )

        # Add buffer stations (red, only if there are any)
        if not stations_in_buffer.empty:
            stations_in_buffer.explore(
                m=m,
                color="red",
                marker_kwds={"radius": 7, "fillOpacity": 0.7, "color": "darkred"},
                tooltip=["NAME", "LAT_WGS84", "LONG_WGS84"],
                name=f"Stations in {buffer_km}km buffer"
            )

        # Add buffer polygon (blue, partially transparent)
        buffer_gdf.explore(
            m=m,
            color="#00BFFF",
            style_kwds=dict(fill=False, fillOpacity=0.15, color="#00BFFF", weight=3),
            tooltip=None,
            name=f"{buffer_km}km buffer"
        )

        # City boundary (orange)
        city_gdf.explore(
            m=m,
            color="#FF9800",
            style_kwds=dict(weight=4, fill=False, dashArray="2,2"),
            tooltip="name",
            name=f"City: {city_name}"
        )

        # Optionally, region boundary
        if region_name:
            region_gdf = self._get_place_geom(region_name)
            if region_gdf is not None and not region_gdf.empty:
                region_gdf.explore(
                    m=m,
                    color="#1E90FF",
                    style_kwds=dict(weight=2, fill=False),
                    tooltip="name",
                    name=f"Region: {region_name}"
                )

        return m