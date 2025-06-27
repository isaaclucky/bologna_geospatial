import csv
import requests
import os
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd

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
        """
        Download data for the specified year and month.
        
        Args:
            year: The year (e.g., 2009, 2010)
            month: The month (1-12)
            output_dir: Directory to save the downloaded file
            
        Returns:
            Path to the downloaded file if successful, None otherwise
        """
        key = (year, month)
        
        if key not in self.data_map:
            print(f"Error: No data available for {year}-{month:02d}")
            available = self.get_available_periods()
            if available:
                print("Available periods:")
                for y, m in available[:10]:  # Show first 10
                    print(f"  - {y}-{m:02d}")
                if len(available) > 10:
                    print(f"  ... and {len(available) - 10} more")
            return None
        
        data_info = self.data_map[key]
        url = data_info['url']
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract filename from URL
        filename = url.split('/')[-1]
        filepath = os.path.join(output_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"File already exists. Skipping download: {filepath}")
            return filepath  # Return existing file path
        
        try:
            print(f"Downloading data for {year}-{month:02d}...")
            print(f"URL: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Write the file with progress indication
            downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
            
            print(f"\nSuccessfully downloaded to: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading file: {e}")
            return None
    
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
                # Try different separators and encodings
                for sep in [';', ',', '\t']:
                    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                        try:
                            df = pd.read_csv(filepath, sep=sep, encoding=encoding)
                            if len(df.columns) > 1:  # Valid separator found
                                print(f"Data loaded successfully. Shape: {df.shape}")
                                return df
                        except:
                            continue
                
                print("Warning: Could not determine proper CSV format. Loading with default settings.")
                df = pd.read_csv(filepath)
                return df
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                return None
        
        return None


# def get_csv_file_path():
#     """Get CSV file path from user or use default."""
#     print("\nCSV File Selection")
#     print("-" * 50)
#     default_path = os.path.join('data', 'DGCTA - Flussi di Traffico.csv')
#     print(f"Default CSV file: {default_path}")
    
#     choice = input("\nDo you want to use the default CSV file? (y/n): ").strip().lower()
    
#     if choice == 'y' or choice == '':
#         return default_path
#     else:
#         custom_path = input("Enter the path to your CSV file: ").strip()
#         # Remove quotes if present
#         custom_path = custom_path.strip('"').strip("'")
#         return custom_path


# def main():
#     """Main function to demonstrate usage."""
#     print("Data Downloader for Traffic Flow Data")
#     print("=" * 50)
    
#     # Get CSV file path
#     csv_path = get_csv_file_path()
    
#     try:
#         # Initialize downloader
#         downloader = DataDownloader(csv_path)
        
#         print("\nAvailable periods:")
#         periods = downloader.get_available_periods()
        
#         # Group by year for better display
#         years = {}
#         for year, month in periods:
#             if year not in years:
#                 years[year] = []
#             years[year].append(month)
        
#         for year in sorted(years.keys()):
#             months_str = ', '.join([f"{m:02d}" for m in sorted(years[year])])
#             print(f"  {year}: {months_str}")
        
#         print(f"\nTotal periods available: {len(periods)}")
#         print()
        
#         # Get user input
#         while True:
#             try:
#                 year = int(input("Enter year (e.g., 2009): "))
#                 month = int(input("Enter month (1-12): "))
                
#                 if 1 <= month <= 12:
#                     break
#                 else:
#                     print("Month must be between 1 and 12")
#             except ValueError:
#                 print("Please enter valid numbers")
        
#         # Download the data
#         filepath = downloader.download_data(year, month)
        
#         if filepath:
#             # Optionally load and display the data
#             load_choice = input("\nDo you want to load and preview the data? (y/n): ")
#             if load_choice.lower() == 'y':
#                 df = downloader.download_and_load_data(year, month)
#                 if df is not None:
#                     print(f"\nData shape: {df.shape}")
#                     print(f"Columns: {list(df.columns)}")
#                     print("\nFirst 5 rows:")
#                     print(df.head())
                    
#                     # Optionally save as Excel
#                     save_excel = input("\nDo you want to save as Excel file? (y/n): ")
#                     if save_excel.lower() == 'y':
#                         excel_path = filepath.replace('.csv', '.xlsx')
#                         df.to_excel(excel_path, index=False)
#                         print(f"Saved as Excel: {excel_path}")
    
#     except Exception as e:
#         print(f"\nError: {e}")
#         print("Please check the file path and format.")


if __name__ == "__main__":
    main()