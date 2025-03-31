"""
Automatic labeling functions for price action
"""

import pandas as pd
import numpy as np
from numba import jit
import streamlit as st
import time
from scipy.signal import find_peaks

from app.utils.gpu_helpers import HAS_GPU


def auto_label_data(df, price_column, date_column, rsi_threshold=30, macd_threshold=0, bb_threshold=2):
    """
    Auto-label data based on technical indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with technical indicators
    price_column : str
        Column containing price data
    date_column : str
        Column containing date data
    rsi_threshold : float
        RSI threshold for buy signals (oversold)
    macd_threshold : float
        MACD threshold for signals
    bb_threshold : float
        Bollinger band threshold for signals
        
    Returns:
    --------
    list
        List of annotation dicts with x, y, label keys
    """
    # Create a copy to avoid modifying the original
    df_labeled = df.copy()
    
    # List to store all annotations
    annotations = []
    
    # RSI-based signals
    if 'rsi' in df_labeled.columns:
        # Buy when RSI below threshold (oversold)
        buy_signals = df_labeled[df_labeled['rsi'] < rsi_threshold]
        
        for idx, row in buy_signals.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'long'
            })
        
        # Sell when RSI above 70 (overbought)
        sell_signals = df_labeled[df_labeled['rsi'] > 70]
        
        for idx, row in sell_signals.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'short'
            })
    
    # MACD signals
    if all(col in df_labeled.columns for col in ['macd', 'macd_signal']):
        # Calculate MACD crossovers
        df_labeled['macd_prev'] = df_labeled['macd'].shift(1)
        df_labeled['signal_prev'] = df_labeled['macd_signal'].shift(1)
        
        # Buy when MACD crosses above signal line
        macd_buy = df_labeled[(df_labeled['macd'] > df_labeled['macd_signal']) & 
                              (df_labeled['macd_prev'] <= df_labeled['signal_prev'])]
        
        for idx, row in macd_buy.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'long'
            })
        
        # Sell when MACD crosses below signal line
        macd_sell = df_labeled[(df_labeled['macd'] < df_labeled['macd_signal']) & 
                               (df_labeled['macd_prev'] >= df_labeled['signal_prev'])]
        
        for idx, row in macd_sell.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'short'
            })
    
    # Bollinger Bands signals
    if all(col in df_labeled.columns for col in ['bb_lower', 'bb_upper', 'bb_middle']):
        # Calculate distance from price to bands in terms of standard deviations
        df_labeled['bb_lower_dist'] = (df_labeled[price_column] - df_labeled['bb_lower']) / (df_labeled['bb_upper'] - df_labeled['bb_lower'])
        df_labeled['bb_upper_dist'] = (df_labeled['bb_upper'] - df_labeled[price_column]) / (df_labeled['bb_upper'] - df_labeled['bb_lower'])
        
        # Buy when price is near lower band
        bb_buy = df_labeled[df_labeled['bb_lower_dist'] < 0.1]
        
        for idx, row in bb_buy.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'long'
            })
        
        # Sell when price is near upper band
        bb_sell = df_labeled[df_labeled['bb_upper_dist'] < 0.1]
        
        for idx, row in bb_sell.iterrows():
            annotations.append({
                'x': row[date_column], 
                'y': row[price_column],
                'label': 'short'
            })
    
    return annotations


@st.cache_data(ttl=3600, hash_funcs={pd.DataFrame: lambda df: hash(str(df.shape) + str(df.columns.tolist()))})
def smart_auto_label(df, price_column, date_column, frequency_threshold=0.5, min_profit_threshold=0.5, optimize_for_winrate=True, min_spacing=5, random_seed=None):
    """
    Smart auto-labeling of price action with optimized algorithm
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price data
    price_column : str
        Column containing price data
    date_column : str
        Column containing date data
    frequency_threshold : float
        Threshold for filtering signals (0-1)
    min_profit_threshold : float
        Minimum profit percentage to consider a trading point
    optimize_for_winrate : bool
        If True, apply stricter criteria to optimize for high win rate (near 100%)
    min_spacing : int
        Minimum spacing between trade points (controls trade frequency)
    random_seed : int or None
        Random seed for reproducible results, if None uses non-deterministic behavior
        
    Returns:
    --------
    list
        List of annotation dicts with x, y, label keys
    """
    start_time = time.time()
    
    # Set random seed for reproducible results if provided
    if random_seed is not None:
        # Ensure seed is within valid range (0 to 2^32-1)
        if not isinstance(random_seed, int) or random_seed < 0:
            random_seed = abs(hash(str(random_seed))) % (2**32 - 1)
        np.random.seed(random_seed)
    
    # Always force optimize_for_winrate to True to ensure 100% win rate
    optimize_for_winrate = True
    
    # Create a copy to avoid modifying the original dataframe
    df_labeled = df.copy()
    
    # Convert to numpy array for faster processing
    prices = np.array(df_labeled[price_column])
    dates = np.array(df_labeled[date_column])
    
    # Use pattern recognition to find key points
    
    # Set parameters for optimal trade points
    pattern_window = 3  # Days to look on each side of a potential pattern point
    # Use min_spacing from parameter 
    lookahead_min = 5  # Minimum lookahead days
    lookahead_max = 20  # Maximum lookahead days
    
    # Find extrema (peaks and valleys) in the price data
    if HAS_GPU:
        # Use GPU implementation if available
        peaks, valleys = find_extrema_numba(prices, pattern_window)
    else:
        # Use CPU implementation
        peaks, valleys = find_extrema_cpu(prices, pattern_window)
    
    # Create empty labels array
    labels = [None] * len(prices)
    
    # Store all potential valid indices for each position type before filtering
    long_indices = []
    short_indices = []
    
    # Store profit potential for each valid point to help with filtering
    profit_potentials = {}
    
    # Modified process_buy_point and process_sell_point to ensure 100% winning trades
    def process_buy_point_with_guaranteed_profit(buy_idx, prices, lookahead_min, lookahead_max, min_profit_threshold, min_spacing, last_buy_idx, labels):
        """Process a potential buy point to ensure it's a profitable entry"""
        # Skip if too close to previous buy
        if buy_idx - last_buy_idx < min_spacing:
            return labels, False
        
        # Skip if too close to end of data
        if buy_idx + lookahead_min >= len(prices):
            return labels, False
        
        # Look ahead to find if there's enough profit potential
        lookahead_end = min(buy_idx + lookahead_max, len(prices) - 1)
        price_at_buy = prices[buy_idx]
        
        # Find highest price in lookahead window
        max_price_ahead = max(prices[buy_idx + lookahead_min:lookahead_end + 1])
        
        # Calculate potential profit percent
        profit_potential = (max_price_ahead / price_at_buy - 1) * 100
        
        # GUARANTEE PROFIT: Only mark as buy point if there is guaranteed profit
        if profit_potential >= min_profit_threshold:
            labels[buy_idx] = 'long'
            return labels, True
        
        return labels, False
    
    def process_sell_point_with_guaranteed_profit(sell_idx, prices, lookahead_min, lookahead_max, min_profit_threshold, min_spacing, last_sell_idx, last_buy_idx, labels):
        """Process a potential sell point to ensure it's a profitable short entry"""
        # Skip if too close to previous sell
        if sell_idx - last_sell_idx < min_spacing:
            return labels, False
            
        # Skip if too close to end of data
        if sell_idx + lookahead_min >= len(prices):
            return labels, False
        
        # Skip if we just bought recently (avoid conflicting signals)
        if sell_idx - last_buy_idx < min_spacing:
            return labels, False
        
        # Look ahead to find if there's enough profit potential for short
        lookahead_end = min(sell_idx + lookahead_max, len(prices) - 1)
        price_at_sell = prices[sell_idx]
        
        # Find lowest price in lookahead window
        min_price_ahead = min(prices[sell_idx + lookahead_min:lookahead_end + 1])
        
        # Calculate potential profit percent for short
        profit_potential = (price_at_sell / min_price_ahead - 1) * 100
        
        # GUARANTEE PROFIT: Only mark as sell point if there is guaranteed profit
        if profit_potential >= min_profit_threshold:
            labels[sell_idx] = 'short'
            return labels, True
        
        return labels, False
    
    # Process buy points (valleys) - now using the guaranteed profit version
    last_buy_idx = -min_spacing * 2
    for buy_idx in valleys:
        labels, success = process_buy_point_with_guaranteed_profit(
            buy_idx, prices, lookahead_min, lookahead_max, 
            min_profit_threshold, min_spacing, last_buy_idx, labels
        )
        
        # If this was identified as a valid long point, store its index
        if success:
            last_buy_idx = buy_idx
            long_indices.append(buy_idx)
            
            # Calculate and store profit potential
            lookahead_end = min(buy_idx + lookahead_max, len(prices) - 1)
            price_at_buy = prices[buy_idx]
            max_price_ahead = max(prices[buy_idx + lookahead_min:lookahead_end + 1])
            profit_potential = (max_price_ahead / price_at_buy - 1) * 100
            profit_potentials[buy_idx] = profit_potential
    
    # Process sell points (peaks) - now using the guaranteed profit version
    last_sell_idx = -min_spacing * 2
    last_buy_idx = -min_spacing * 2
    for sell_idx in peaks:
        labels, success = process_sell_point_with_guaranteed_profit(
            sell_idx, prices, lookahead_min, lookahead_max, 
            min_profit_threshold, min_spacing, last_sell_idx, last_buy_idx, labels
        )
        
        # If this was identified as a valid short point, store its index
        if success:
            last_sell_idx = sell_idx
            short_indices.append(sell_idx)
            
            # Calculate and store profit potential
            lookahead_end = min(sell_idx + lookahead_max, len(prices) - 1)
            price_at_sell = prices[sell_idx]
            min_price_ahead = min(prices[sell_idx + lookahead_min:lookahead_end + 1])
            profit_potential = (price_at_sell / min_price_ahead - 1) * 100
            profit_potentials[sell_idx] = profit_potential
    
    print(f"Found {len(long_indices)} long and {len(short_indices)} short potential trade points before filtering")
    total_valid_indices = len(long_indices) + len(short_indices)
    
    # Apply frequency filtering separately for each position type while preserving profitable trades
    if frequency_threshold > 0.1 and total_valid_indices > 0:
        # Higher threshold means fewer points
        filtered_labels = [None] * len(prices)
        
        # Sort indices by profit potential (higher profit first) - results will be deterministic with seed
        long_indices.sort(key=lambda idx: profit_potentials.get(idx, 0), reverse=True)
        short_indices.sort(key=lambda idx: profit_potentials.get(idx, 0), reverse=True)
        
        # Calculate how many trades to keep based on frequency threshold
        # Higher frequency threshold = fewer trades
        long_keep_count = max(1, int(len(long_indices) * (1 - frequency_threshold)))
        short_keep_count = max(1, int(len(short_indices) * (1 - frequency_threshold)))
        
        # Always keep at least one of each type if available
        if len(long_indices) > 0:
            long_keep_count = max(1, long_keep_count)
        if len(short_indices) > 0:
            short_keep_count = max(1, short_keep_count)
            
        # Take the top N most profitable positions based on keep count
        selected_long_indices = sorted(long_indices[:long_keep_count])
        selected_short_indices = sorted(short_indices[:short_keep_count])
        
        # Set labels for selected indices
        for idx in selected_long_indices:
            filtered_labels[idx] = 'long'
        
        for idx in selected_short_indices:
            filtered_labels[idx] = 'short'
        
        # Replace original labels with filtered ones
        labels = filtered_labels
        
        print(f"After frequency filtering: kept {len(selected_long_indices)} long and {len(selected_short_indices)} short positions")
    
    # Reset random seed to avoid affecting other parts of the program
    if random_seed is not None:
        np.random.seed(None)
    
    # Convert labels to annotations
    annotations = []
    for i, label in enumerate(labels):
        if label:
            annotations.append({
                'x': dates[i],
                'y': prices[i],
                'label': label
            })
    
    # Count final distribution
    long_count = sum(1 for a in annotations if a['label'] == 'long')
    short_count = sum(1 for a in annotations if a['label'] == 'short')
    
    print(f"Smart auto-labeling complete in {time.time() - start_time:.2f} seconds")
    print(f"Generated {len(annotations)} trade points with 100% win rate guarantee")
    print(f"Distribution: {long_count} long positions, {short_count} short positions")
    
    return annotations


@jit(nopython=True)
def find_extrema_numba(prices, pattern_window):
    """
    Optimized function to find local extrema (peaks and valleys) 
    
    Parameters:
    -----------
    prices : np.array
        Array of price values
    pattern_window : int
        Window size for pattern detection
        
    Returns:
    --------
    tuple
        Arrays of peak and valley indices
    """
    n = len(prices)
    peaks = []
    valleys = []
    
    # We need at least 2*pattern_window+1 points to find extrema
    if n < 2 * pattern_window + 1:
        return np.array(peaks), np.array(valleys)
    
    # Start and end offset for edge cases
    offset = pattern_window
    
    # Iterate through the array to find peaks and valleys
    for i in range(offset, n - offset):
        # Check if this is a peak
        is_peak = True
        for j in range(1, pattern_window + 1):
            if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                is_peak = False
                break
                
        if is_peak:
            peaks.append(i)
            continue
            
        # Check if this is a valley
        is_valley = True
        for j in range(1, pattern_window + 1):
            if prices[i] >= prices[i - j] or prices[i] >= prices[i + j]:
                is_valley = False
                break
                
        if is_valley:
            valleys.append(i)
    
    return np.array(peaks), np.array(valleys)


def find_extrema_cpu(prices, pattern_window):
    """
    Find local extrema (peaks and valleys) in price data using CPU
    Returns two arrays with indices of peaks and valleys
    """
    # Find peaks - local maxima
    peaks, _ = find_peaks(prices, distance=pattern_window)
    # Find valleys - local minima (we invert the array to use find_peaks)
    valleys, _ = find_peaks(-prices, distance=pattern_window)
    
    return peaks, valleys


def process_buy_point(buy_idx, prices, dates, lookahead_min, lookahead_max, min_profit_threshold, min_spacing, last_buy_idx, labels, optimize_for_winrate):
    """Process a potential buy point to determine if it's a good entry"""
    # Skip if too close to previous buy
    if buy_idx - last_buy_idx < min_spacing:
        return labels
    
    # Skip if too close to end of data
    if buy_idx + lookahead_min >= len(prices):
        return labels
    
    # Look ahead to find if there's enough profit potential
    lookahead_end = min(buy_idx + lookahead_max, len(prices) - 1)
    price_at_buy = prices[buy_idx]
    
    # Find highest price in lookahead window
    max_price_ahead = max(prices[buy_idx + lookahead_min:lookahead_end + 1])
    
    # Calculate potential profit percent
    profit_potential = (max_price_ahead / price_at_buy - 1) * 100
    
    # GUARANTEE PROFIT: Only mark as buy point if there is guaranteed profit
    # With optimize_for_winrate always True, we ensure only profitable points are labeled
    if profit_potential >= min_profit_threshold:
        labels[buy_idx] = 'long'
    
    return labels


def process_sell_point(sell_idx, prices, dates, lookahead_min, lookahead_max, min_profit_threshold, min_spacing, last_sell_idx, last_buy_idx, labels, optimize_for_winrate):
    """Process a potential sell point to determine if it's a good entry for short"""
    # Skip if too close to previous sell
    if sell_idx - last_sell_idx < min_spacing:
        return labels
        
    # Skip if too close to end of data
    if sell_idx + lookahead_min >= len(prices):
        return labels
    
    # Skip if we just bought recently (avoid conflicting signals)
    if sell_idx - last_buy_idx < min_spacing:
        return labels
    
    # Look ahead to find if there's enough profit potential for short
    lookahead_end = min(sell_idx + lookahead_max, len(prices) - 1)
    price_at_sell = prices[sell_idx]
    
    # Find lowest price in lookahead window
    min_price_ahead = min(prices[sell_idx + lookahead_min:lookahead_end + 1])
    
    # Calculate potential profit percent for short
    profit_potential = (price_at_sell / min_price_ahead - 1) * 100
    
    # GUARANTEE PROFIT: Only mark as sell point if there is guaranteed profit
    # With optimize_for_winrate always True, we ensure only profitable points are labeled
    if profit_potential >= min_profit_threshold:
        labels[sell_idx] = 'short'
    
    return labels 