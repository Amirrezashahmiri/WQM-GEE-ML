import pandas as pd

# Load the CSV files
in_situ_df = pd.read_csv(r'C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\filtered_min_depth.csv')
satellite_df = pd.read_csv(r'C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\Dam_Parameters_Nearest_Pixels.csv')

# Convert date columns to datetime for matching
in_situ_df['Date'] = pd.to_datetime(in_situ_df['Date'])
satellite_df['insitu_date'] = pd.to_datetime(satellite_df['insitu_date'])

# Ensure latitude and longitude are floats
in_situ_df['Lat'] = in_situ_df['Lat'].astype(float)
in_situ_df['Long'] = in_situ_df['Long'].astype(float)
satellite_df['latitude'] = satellite_df['latitude'].astype(float)
satellite_df['longitude'] = satellite_df['longitude'].astype(float)

# Split satellite data into Landsat and Sentinel-2 based on the 'product' column
landsat_df = satellite_df[satellite_df['product'] == 'Landsat']
sentinel_df = satellite_df[satellite_df['product'] == 'Sentinel-2']

# Merge in-situ data with Landsat data based on exact matches of latitude, longitude, and date
landsat_combined_df = pd.merge(
    in_situ_df,
    landsat_df,
    left_on=['Lat', 'Long', 'Date'],
    right_on=['latitude', 'longitude', 'insitu_date'],
    how='inner'
)

# Merge in-situ data with Sentinel-2 data based on exact matches of latitude, longitude, and date
sentinel_combined_df = pd.merge(
    in_situ_df,
    sentinel_df,
    left_on=['Lat', 'Long', 'Date'],
    right_on=['latitude', 'longitude', 'insitu_date'],
    how='inner'
)

# Save the combined data to separate CSV files
landsat_combined_df.to_csv(r'C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\landsat_matched_data.csv', index=False)
sentinel_combined_df.to_csv(r'C:\Users\DFMRendering\Desktop\Microplastic\GEE_Dhaygah_Dam_300m\Calibration\sentinel_matched_data.csv', index=False)

# Report the number of matches
print(f"Landsat: {len(landsat_combined_df)}")
print(f"Sentinel2 {len(sentinel_combined_df)}")