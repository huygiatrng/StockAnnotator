"""
Chart utility functions for data validation and visualization
"""

import pandas as pd
import numpy as np


def verify_price_data(x_data, y_data, is_datetime=False):
    """
    Verify if price data appears valid (not just indices)
    
    Parameters:
    -----------
    x_data : pd.Series
        X-axis data (dates or categories)
    y_data : pd.Series
        Y-axis data (prices)
    is_datetime : bool
        Whether x-axis contains dates
        
    Returns:
    --------
    tuple
        (x_data, y_data, is_valid)
    """
    is_valid = True
    
    # Check for index-like values (common issue)
    if pd.api.types.is_numeric_dtype(y_data):
        # Check for sequence 0,1,2,3... or 1,2,3,4...
        if len(y_data) > 5:  # Need enough data to check
            y_diff = y_data.diff().dropna()
            
            # Check if it's a strictly increasing sequence with step=1
            is_sequential = (y_diff == 1).mean() > 0.9  # 90% of diffs are 1
            starts_at_zero_or_one = y_data.iloc[0] <= 1
            
            if is_sequential and starts_at_zero_or_one:
                print("ERROR: Y values appear to be an index sequence (0,1,2,3...), not price data!")
                is_valid = False
    
    # Check for reasonable price variance
    if is_valid and pd.api.types.is_numeric_dtype(y_data):
        if len(y_data) > 10:
            # Real price data should have some variance
            variance = y_data.var()
            mean = y_data.mean()
            cv = (variance**0.5) / mean if mean != 0 else 0  # Coefficient of variation
            
            if cv < 0.0001:  # Extremely low variation
                print(f"WARNING: Price data has very low variation (CV={cv:.6f})")
                is_valid = False
    
    # For real price data, we expect some up and down movement
    if is_valid and pd.api.types.is_numeric_dtype(y_data) and len(y_data) > 20:
        ups = (y_data.diff() > 0).sum()
        downs = (y_data.diff() < 0).sum()
        flats = (y_data.diff() == 0).sum()
        
        # If price only goes up or only goes down, it's suspicious
        if ups == 0 or downs == 0:
            print(f"WARNING: Unusual price pattern - ups: {ups}, downs: {downs}, flats: {flats}")
            if (ups / len(y_data) > 0.95) or (downs / len(y_data) > 0.95):
                print("ERROR: Price only moves in one direction!")
                is_valid = False
    
    return x_data, y_data, is_valid


def detect_integer_sequences(df):
    """
    Detect columns that appear to be integer sequences rather than data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of column names that appear to be sequences
    """
    sequence_columns = []
    
    for col in df.select_dtypes(include=['int', 'int64', 'float', 'float64']).columns:
        # Skip small columns
        if len(df[col]) < 10:
            continue
            
        diffs = df[col].diff().dropna()
        
        # Check if column is a regular sequence (like 1,2,3,4...)
        is_regular_sequence = False
        
        # Check for step=1 sequence (most common for index)
        if (diffs == 1).mean() > 0.9 and df[col].iloc[0] in [0, 1]:
            is_regular_sequence = True
        
        # Check for any other fixed step sequence
        elif diffs.nunique() == 1:
            is_regular_sequence = True
            
        if is_regular_sequence:
            sequence_columns.append(col)
            
    return sequence_columns


def suggest_price_column(df, exclude_cols=None):
    """
    Suggest the best column to use for price data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : list, optional
        Columns to exclude from consideration
        
    Returns:
    --------
    str or None
        Suggested column name
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # First look for columns with common price names
    price_names = ['close', 'price', 'adjclose', 'adj close', 'last', 'value']
    for col in df.columns:
        if col.lower() in price_names or any(name in col.lower() for name in price_names):
            if col not in exclude_cols:
                return col
    
    # Next, look for numeric columns with reasonable price variance
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    
    # Exclude sequence columns
    sequence_cols = detect_integer_sequences(df)
    candidates = [col for col in numeric_cols if col not in exclude_cols and col not in sequence_cols]
    
    if not candidates:
        return None
        
    # Prefer columns with more variance (more likely to be price data)
    best_col = None
    best_score = -1
    
    for col in candidates:
        # Skip date-related columns
        if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'year']):
            continue
            
        # Calculate variance normalized by mean (coefficient of variation)
        mean = df[col].mean()
        if mean == 0:
            continue
            
        std = df[col].std()
        cv = std / abs(mean)
        
        # Check up/down movement (prices should have both)
        diffs = df[col].diff().dropna()
        ups = (diffs > 0).sum()
        downs = (diffs < 0).sum()
        
        # Score is higher for columns with more balanced movements and higher variance
        if ups > 0 and downs > 0:
            movement_balance = min(ups, downs) / max(ups, downs)
            score = cv * movement_balance
            
            if score > best_score:
                best_score = score
                best_col = col
    
    return best_col 