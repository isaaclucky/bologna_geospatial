import requests
import os
from typing import Optional, List
import pandas as pd
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

class DataDownloader:
    def __init__(self, csv_file_path: Optional[str] = None):
        """
        Initialize the DataDownloader with CSV file path.
        
        Args:
            csv_file_path: Path to the CSV file. If None, uses default path.
        """
        self.data_map = {}
        
        # Default CSV file path
        if csv_file_path is None:
            csv_file_path = os.path.join('data', 'DGCTA - Flussi di Traffico.csv')
        
        self.csv_file_path = csv_file_path
        self._load_csv_data()
    
    def _load_csv_data(self):
        """Load and parse the CSV data from file."""
        try:
            print(f"Loading data from: {self.csv_file_path}")
            
            with open(self.csv_file_path, 'r', encoding='utf-8') as file:
                # Read the content
                content = file.read()
                self._parse_csv_data(content)
                
            print(f"Successfully loaded {len(self.data_map)} records")
            
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_file_path}")
            print("Please make sure the file exists at the specified path.")
            raise
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise
    
    def _parse_csv_data(self, csv_data: str):
        """Parse the CSV data and create a mapping of (year, month) -> data info."""
        lines = csv_data.strip().split('\n')
        
        # Skip header if present (check if first line contains URLs)
        start_idx = 0
        if lines[0] and 'http' not in lines[0]:
            start_idx = 1
        
        for line in lines[start_idx:]:
            parts = line.split(';')
            if len(parts) >= 8:
                dataset_id = parts[0]
                year = parts[1]
                month = parts[2]
                url = parts[3]
                frequency = parts[4]
                status_code = parts[5]
                timestamp = parts[6]
                availability = parts[7]
                
                key = (int(year), int(month))
                self.data_map[key] = {
                    'dataset_id': dataset_id,
                    'url': url,
                    'frequency': frequency,
                    'status_code': status_code,
                    'timestamp': timestamp,
                    'availability': availability
                }
    
    def get_available_periods(self) -> List[tuple]:
        """Return a list of available (year, month) tuples."""
        return sorted(self.data_map.keys())
    

    def download_data(self, year: int, month: int, output_dir: str = "../data/regional_traffic_data_downloads") -> Optional[str]:
        key = (year, month)
        if key not in self.data_map:
            print(f"Error: No data available for {year}-{month:02d}")
            return None

        data_info = self.data_map[key]
        url = data_info['url']
        os.makedirs(output_dir, exist_ok=True)
        filename = url.split('/')[-1]
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"File already exists. Skipping download: {filepath}")
            return filepath

        try:
            print(f"Downloading {year}-{month:02d} ...")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filepath}")
            return filepath
        except Exception as e:
            print(f"Failed to download {year}-{month:02d}: {e}")
            return None

    # --- New download_year method ---
    def download_year(self, year: int, output_dir: str = "../data/regional_traffic_data_downloads", max_workers: int = 4) -> List[str]:
        """
        Download all monthly files for the given year, in parallel.
        Returns a list of successfully downloaded file paths.
        """
        months = [m for (y, m) in self.data_map if y == year]
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.download_data, year, month, output_dir): month for month in months}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        return results


    def download_and_load_data(self, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Download data and load it into a pandas DataFrame.
        
        Args:
            year: The year
            month: The month
            
        Returns:
            DataFrame if successful, None otherwise
        """
        filepath = self.download_data(year, month)
        
        if filepath:
            try:
                df = pd.read_csv(filepath)
                return df
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return None
        
        return None

