# %% [code] {"jupyter":{"outputs_hidden":false}}
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
    """Fill missing values for pollutant columns, dropping those with >90% missing values."""
    df = df.copy()
    
    
    pollutant_columns = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
    env_columns = ['humidity', 'pressure', 'temperature', 'wind-speed']
    
    # Calculate percentage of null values for pollutant columns
    null_percentages = df[pollutant_columns].isnull().mean() * 100
    
    # Drop columns with more than 90% null values
    columns_to_drop = null_percentages[null_percentages > 90].index
    df = df.drop(columns=columns_to_drop)
    
    # Fill remaining pollutant columns with mean
    remaining_pollutants = [col for col in pollutant_columns if col in df.columns]
    
    # Special handling for PM10 and PM2.5
    if 'pm10' in df.columns:
        df['pm10'] = df['pm10'].fillna(df['pm10'].mean())
    if 'pm25' in df.columns:
        df['pm25'] = df['pm25'].fillna(df['pm25'].mean())
    
    # Fill other pollutants
    other_pollutants = [col for col in remaining_pollutants if col not in ['pm10', 'pm25']]
    if other_pollutants:
        df[other_pollutants] = df[other_pollutants].fillna(df[other_pollutants].mean())
    
    # Fill environmental columns if they exist
    existing_env_columns = [col for col in env_columns if col in df.columns]
    if existing_env_columns:
        df[existing_env_columns] = df[existing_env_columns].fillna(df[existing_env_columns].mean())
    
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

def calculate_aqi(data, subindex_columns=None):
    """Calculate Air Quality Index based on available pollutants."""
    data = data.copy()
    
    # Determine available pollutant columns
    if subindex_columns is None:
        all_possible_columns = ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2']
        subindex_columns = [col for col in all_possible_columns if col in data.columns]
    
    # Skip if no pollutant columns are available
    if not subindex_columns:
        print(f"Warning: No pollutant columns available for {data.index}. Skipping record.")
        return None
    
    # Calculate AQI as maximum value among available pollutants
    data["AQI"] = data[subindex_columns].max(axis=1)
    data["AQI"] = data["AQI"].round()
    
    return data

def handle_aqi_outliers(series, method='iqr'):
    """Handle outliers in AQI data."""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)
    elif method == 'rolling':
        return series.rolling(window=3, center=True, min_periods=1).median()
    return series

def calculate_smooth_aqi(df):
    """Calculate smoothed AQI based on available pollutants."""
    df_processed = df.copy()
    
    # Get available pollutant columns
    pollutant_columns = [col for col in ['co', 'no2', 'o3', 'pm10', 'pm25', 'so2'] 
                        if col in df.columns]
    
    if not pollutant_columns:
        raise ValueError("No pollutant columns available for AQI calculation")
    
    # Preprocess available components
    for col in pollutant_columns:
        df_processed[col] = handle_aqi_outliers(df_processed[col], method='iqr')
        df_processed[col] = (df_processed[col]
                           .rolling(window=6, center=True)
                           .mean()
                           .ffill()
                           .bfill())
    
    # Calculate smooth AQI
    smooth_aqi = df_processed[pollutant_columns].max(axis=1)
    smooth_aqi = handle_aqi_outliers(smooth_aqi, method='rolling')
    
    df_processed['AQI_Smooth'] = smooth_aqi
    
    return df_processed