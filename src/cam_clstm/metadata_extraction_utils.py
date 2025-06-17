
import numpy as np
import pandas as pd

def extract_metadata_from_config(config):
    """
    Returns default lat, lon grid from config if dataset doesn't contain it.
    """
    height = config['grid']['height']
    width = config['grid']['width']
    lat_range = config['grid'].get('lat_range', [0, height])
    lon_range = config['grid'].get('lon_range', [0, width])

    latitudes = np.linspace(lat_range[0], lat_range[1], height)
    longitudes = np.linspace(lon_range[0], lon_range[1], width)
    return latitudes, longitudes

def extract_times_from_dataframe(df, config):
    """
    Uses config-specified prediction start time and frequency to extract time series.
    """
    pred_start = pd.to_datetime(config['temporal']['prediction_start'])
    freq = config['temporal'].get('freq', 'M')
    output_steps = config['temporal']['output_steps']
    return pd.date_range(pred_start, periods=output_steps, freq=freq).strftime("%Y-%m").tolist()
