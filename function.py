import datetime
import pandas as pd
import numpy as np

def prepare_datetime_index(df, date_column='Date'):
    """Convert date column to datetime and set as index."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.drop_duplicates(subset=[date_column], keep='last')
    df.set_index(date_column, inplace=True)
    return df

def fill_pollutant_values(df):
    """Fill missing values for specific pollutant and environmental columns with their mean values."""
    df = df.copy()
    
    # Define columns to fill
    columns_to_fill = [
        'co', 'no2', 'o3', 'pm10', 'pm25', 'so2',
        'humidity', 'pressure', 'temperature', 'wind-speed'
    ]
    
    # Only fill specified columns that exist in the dataframe
    existing_columns = [col for col in columns_to_fill if col in df.columns]
    df[existing_columns] = df[existing_columns].fillna(df[existing_columns].mean())
    
    return df

def impute_missing_dates(df):
    """Handle missing dates and duplicates in the time series."""
    df = df.copy()
    
    # Create full date range
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    
    # Reindex to include missing dates
    df = df.reindex(full_range)
    
    # Fill missing values using forward fill
    df = df.ffill()
    # Drop any duplicates and sort by index
    df = df[~df.index.duplicated(keep='last')].sort_index()
    
    return df

def calculate_aqi(data, subindex_columns=['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']):
    """Calculate Air Quality Index."""
    data = data.copy()
    
    data["Checks"] = data[subindex_columns].gt(0).sum(axis=1)
    data["AQI"] = data[subindex_columns].max(axis=1)
    
    data.loc[
        (data["pm25"] == 0) & 
        (data["pm10"] == 0), 
        "AQI"
    ] = 0
    
    data.loc[data["Checks"] < 3, "AQI"] = 0
    data["AQI"] = data["AQI"].round()
    
    return data
    
import numpy as np
from scipy import stats

def handle_aqi_outliers(df, method='iqr', threshold=3):
    """
    Handle outliers in AQI data using various methods.
    
    Parameters:
    df: DataFrame or Series containing AQI values
    method: 'iqr' for IQR method, 'zscore' for Z-score method, 'rolling' for rolling median
    threshold: threshold for outlier detection (default 3 for z-score, 1.5 for IQR)
    """
    df = df.copy()
    
    if method == 'iqr':
        # IQR method
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Replace outliers with bounds
        df = df.clip(lower=lower_bound, upper=upper_bound)
        
    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(df))
        df[z_scores > threshold] = df.median()
        
    elif method == 'rolling':
        # Rolling median method
        window_size = 7  # Adjust window size as needed
        rolling_median = df.rolling(window=window_size, center=True).median()
        rolling_std = df.rolling(window=window_size, center=True).std()
        
        lower_bound = rolling_median - threshold * rolling_std
        upper_bound = rolling_median + threshold * rolling_std
        
        # Replace values outside bounds with rolling median
        mask = (df < lower_bound) | (df > upper_bound)
        df[mask] = rolling_median[mask]
    
    return df

def preprocess_aqi_components(df, columns=['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']):
    """
    Preprocess individual AQI components before calculating final AQI.
    """
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            # Handle outliers in each component
            df[col] = handle_aqi_outliers(df[col], method='iqr')
            
            # Apply smoothing and handle NaN values using ffill and bfill
            df[col] = (df[col]
                      .rolling(window=6, center=True)
                      .mean()
                      .ffill()  # Forward fill
                      .bfill())  # Backward fill for any remaining NaN at the start
    
    return df

def calculate_smooth_aqi(df, columns=['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']):
    """
    Calculate AQI with smoothed components and return complete DataFrame.
    
    Parameters:
    df: DataFrame containing AQI component columns
    columns: List of column names for AQI components
    
    Returns:
    DataFrame with original columns plus smoothed AQI
    """
    # Create a copy of the input DataFrame
    df_processed = df.copy()
    
    # Preprocess components first
    df_processed = preprocess_aqi_components(df_processed, columns)
    
    # Calculate AQI using smoothed components
    smooth_aqi = df_processed[columns].max(axis=1)
    
    # Final smoothing on AQI
    smooth_aqi = handle_aqi_outliers(smooth_aqi, method='rolling')
    
    # Add smoothed AQI as a new column
    df_processed['AQI_Smooth'] = smooth_aqi
    
    return df_processed