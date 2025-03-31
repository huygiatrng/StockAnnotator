"""
Technical indicators calculation functions
"""

import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(ttl=3600, hash_funcs={pd.DataFrame: lambda df: hash(str(df.shape) + str(df.columns.tolist()))})
def add_technical_indicators(df, price_column, indicators):
    """
    Add technical indicators to the dataframe
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    price_column : str
        Column name containing price data
    indicators : dict
        Dictionary of indicators to add with parameters
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with indicators added
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Calculate each indicator
    for indicator, params in indicators.items():
        if indicator == 'sma':
            period = params
            result_df[f'SMA_{period}'] = calculate_sma(result_df[price_column], period)
        
        elif indicator == 'ema':
            period = params
            result_df[f'EMA_{period}'] = calculate_ema(result_df[price_column], period)
        
        elif indicator == 'rsi':
            period = params
            result_df[f'RSI_{period}'] = calculate_rsi(result_df[price_column], period)
        
        elif indicator == 'macd':
            fast_period, slow_period, signal_period = params
            result_df = calculate_macd(result_df, price_column, fast_period, slow_period, signal_period)
        
        elif indicator == 'bollinger':
            period, std_dev = params
            result_df = calculate_bollinger_bands(result_df, price_column, period, std_dev)
        
        elif indicator == 'atr':
            period = params
            result_df = calculate_atr(result_df, price_column, period)
        
        elif indicator == 'volume':
            volume_col, period = params
            if volume_col in result_df.columns:
                result_df[f'Volume_SMA_{period}'] = calculate_sma(result_df[volume_col], period)
                result_df['Volume_Ratio'] = result_df[volume_col] / result_df[f'Volume_SMA_{period}']
        
        elif indicator == 'adx':
            period = params
            result_df = calculate_adx(result_df, price_column, period)
        
        elif indicator == 'ichimoku':
            conversion_period, base_period = params
            result_df = calculate_ichimoku(result_df, price_column, conversion_period, base_period)
    
    return result_df


def calculate_sma(series, period):
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period):
    """Calculate Relative Strength Index"""
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df, price_column, fast_period=12, slow_period=26, signal_period=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    # Calculate EMAs
    fast_ema = calculate_ema(df[price_column], fast_period)
    slow_ema = calculate_ema(df[price_column], slow_period)
    
    # Calculate MACD line
    df['MACD_Line'] = fast_ema - slow_ema
    
    # Calculate signal line
    df['MACD_Signal'] = calculate_ema(df['MACD_Line'], signal_period)
    
    # Calculate histogram
    df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
    
    return df


def calculate_bollinger_bands(df, price_column, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    # Calculate SMA
    df['BB_Middle'] = calculate_sma(df[price_column], period)
    
    # Calculate standard deviation
    rolling_std = df[price_column].rolling(window=period).std()
    
    # Calculate upper and lower bands
    df['BB_Upper'] = df['BB_Middle'] + (rolling_std * std_dev)
    df['BB_Lower'] = df['BB_Middle'] - (rolling_std * std_dev)
    
    # Calculate bandwidth and %B
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_PercentB'] = (df[price_column] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df


def calculate_atr(df, price_column, period=14):
    """Calculate Average True Range"""
    # Check if we have high and low columns
    high_column = 'High' if 'High' in df.columns else price_column
    low_column = 'Low' if 'Low' in df.columns else price_column
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # Calculate true range
    if high_column == price_column or low_column == price_column:
        # If we only have close prices, approximate TR using close-to-close
        df['TR'] = abs(df[price_column].diff())
    else:
        # If we have high and low, calculate proper TR
        df['TR'] = np.maximum(
            df[high_column] - df[low_column],
            np.maximum(
                abs(df[high_column] - df[price_column].shift()),
                abs(df[low_column] - df[price_column].shift())
            )
        )
    
    # Calculate ATR
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Drop the temporary column
    df = df.drop(columns=['TR'])
    
    return df


def calculate_adx(df, price_column, period=14):
    """Calculate Average Directional Index"""
    # Check if we have high and low columns
    high_column = 'High' if 'High' in df.columns else price_column
    low_column = 'Low' if 'Low' in df.columns else price_column
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # If we don't have high/low data, we can't properly calculate ADX
    if high_column == price_column or low_column == price_column:
        df['ADX'] = np.nan
        return df
    
    # Calculate +DM and -DM
    df['UpMove'] = df[high_column].diff()
    df['DownMove'] = df[low_column].shift() - df[low_column]
    
    df['PosDM'] = np.where(
        (df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0),
        df['UpMove'],
        0
    )
    
    df['NegDM'] = np.where(
        (df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0),
        df['DownMove'],
        0
    )
    
    # Calculate TR
    df['TR'] = np.maximum(
        df[high_column] - df[low_column],
        np.maximum(
            abs(df[high_column] - df[price_column].shift()),
            abs(df[low_column] - df[price_column].shift())
        )
    )
    
    # Calculate smoothed values
    for i in ['TR', 'PosDM', 'NegDM']:
        df[f'{i}_Smooth'] = df[i].rolling(window=period).sum()
    
    # Calculate +DI and -DI
    df['PosDI'] = 100 * (df['PosDM_Smooth'] / df['TR_Smooth'])
    df['NegDI'] = 100 * (df['NegDM_Smooth'] / df['TR_Smooth'])
    
    # Calculate DX
    df['DX'] = 100 * abs(df['PosDI'] - df['NegDI']) / (df['PosDI'] + df['NegDI'])
    
    # Calculate ADX
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # Drop temporary columns
    df = df.drop(columns=['UpMove', 'DownMove', 'PosDM', 'NegDM', 'TR', 
                          'TR_Smooth', 'PosDM_Smooth', 'NegDM_Smooth', 'DX'])
    
    return df


def calculate_ichimoku(df, price_column, conversion_period=9, base_period=26):
    """Calculate Ichimoku Cloud"""
    # Check if we have high and low columns
    high_column = 'High' if 'High' in df.columns else price_column
    low_column = 'Low' if 'Low' in df.columns else price_column
    
    # Create a copy of the dataframe
    df = df.copy()
    
    # For approximation with only close prices
    if high_column == price_column or low_column == price_column:
        high_values = df[price_column]
        low_values = df[price_column]
    else:
        high_values = df[high_column]
        low_values = df[low_column]
    
    # Conversion Line (Tenkan-sen)
    df['Ichimoku_Conversion'] = (
        high_values.rolling(window=conversion_period).max() +
        low_values.rolling(window=conversion_period).min()
    ) / 2
    
    # Base Line (Kijun-sen)
    df['Ichimoku_Base'] = (
        high_values.rolling(window=base_period).max() +
        low_values.rolling(window=base_period).min()
    ) / 2
    
    # Leading Span A (Senkou Span A)
    df['Ichimoku_SpanA'] = ((df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2).shift(base_period)
    
    # Leading Span B (Senkou Span B)
    span_b_period = base_period * 2
    df['Ichimoku_SpanB'] = (
        high_values.rolling(window=span_b_period).max() +
        low_values.rolling(window=span_b_period).min()
    ).shift(base_period) / 2
    
    # Lagging Span (Chikou Span)
    df['Ichimoku_Lagging'] = df[price_column].shift(-base_period)
    
    return df 