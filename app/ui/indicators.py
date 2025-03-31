"""
UI components for technical indicators
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.core.indicators import add_technical_indicators


def render_indicator_controls():
    """Render UI controls for technical indicators"""
    st.subheader("Technical Indicators")
    
    # Only enable if data is loaded
    if st.session_state.df is None or st.session_state.y_column is None:
        st.warning("Load data first to add technical indicators")
        return
    
    # Check if we already have indicators
    has_indicators = any(col for col in st.session_state.df.columns 
                         if col.startswith(('SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR')))
    
    # Column for selecting indicators
    indicator_types = [
        "Moving Averages",
        "Oscillators",
        "Volatility",
        "Volume",
        "Trend"
    ]
    
    selected_type = st.selectbox("Indicator Type", indicator_types)
    
    if selected_type == "Moving Averages":
        col1, col2 = st.columns(2)
        
        with col1:
            use_sma = st.checkbox("Simple Moving Average (SMA)")
            if use_sma:
                sma_period = st.slider("SMA Period", 5, 200, 20)
        
        with col2:
            use_ema = st.checkbox("Exponential Moving Average (EMA)")
            if use_ema:
                ema_period = st.slider("EMA Period", 5, 200, 20)
    
    elif selected_type == "Oscillators":
        col1, col2 = st.columns(2)
        
        with col1:
            use_rsi = st.checkbox("Relative Strength Index (RSI)")
            if use_rsi:
                rsi_period = st.slider("RSI Period", 5, 30, 14)
        
        with col2:
            use_macd = st.checkbox("MACD")
            if use_macd:
                macd_fast = st.slider("MACD Fast", 5, 30, 12)
                macd_slow = st.slider("MACD Slow", 10, 50, 26)
                macd_signal = st.slider("MACD Signal", 5, 20, 9)
    
    elif selected_type == "Volatility":
        col1, col2 = st.columns(2)
        
        with col1:
            use_bollinger = st.checkbox("Bollinger Bands")
            if use_bollinger:
                bb_period = st.slider("BB Period", 5, 50, 20)
                bb_std = st.slider("BB Standard Deviation", 1.0, 4.0, 2.0, 0.1)
        
        with col2:
            use_atr = st.checkbox("Average True Range (ATR)")
            if use_atr:
                atr_period = st.slider("ATR Period", 5, 30, 14)
    
    elif selected_type == "Volume":
        use_volume = st.checkbox("Volume Analysis")
        if use_volume:
            # Check if volume column exists
            volume_cols = [col for col in st.session_state.df.columns 
                           if col.lower() in ('volume', 'vol', 'quantity')]
            
            if volume_cols:
                volume_col = st.selectbox("Volume Column", volume_cols)
                volume_ma_period = st.slider("Volume MA Period", 5, 50, 20)
            else:
                st.warning("No volume column detected in the data")
    
    elif selected_type == "Trend":
        col1, col2 = st.columns(2)
        
        with col1:
            use_adx = st.checkbox("Average Directional Index (ADX)")
            if use_adx:
                adx_period = st.slider("ADX Period", 5, 30, 14)
        
        with col2:
            use_ichimoku = st.checkbox("Ichimoku Cloud")
            if use_ichimoku:
                ichimoku_conversion = st.slider("Conversion Line Period", 5, 30, 9)
                ichimoku_base = st.slider("Base Line Period", 10, 60, 26)
    
    # Button to calculate indicators
    if st.button("Calculate Indicators"):
        # Collect selected indicators
        indicators = {}
        
        if selected_type == "Moving Averages":
            if 'use_sma' in locals() and use_sma:
                indicators['sma'] = sma_period
            if 'use_ema' in locals() and use_ema:
                indicators['ema'] = ema_period
        
        elif selected_type == "Oscillators":
            if 'use_rsi' in locals() and use_rsi:
                indicators['rsi'] = rsi_period
            if 'use_macd' in locals() and use_macd:
                indicators['macd'] = (macd_fast, macd_slow, macd_signal)
        
        elif selected_type == "Volatility":
            if 'use_bollinger' in locals() and use_bollinger:
                indicators['bollinger'] = (bb_period, bb_std)
            if 'use_atr' in locals() and use_atr:
                indicators['atr'] = atr_period
        
        elif selected_type == "Volume":
            if 'use_volume' in locals() and use_volume and 'volume_col' in locals():
                indicators['volume'] = (volume_col, volume_ma_period)
        
        elif selected_type == "Trend":
            if 'use_adx' in locals() and use_adx:
                indicators['adx'] = adx_period
            if 'use_ichimoku' in locals() and use_ichimoku:
                indicators['ichimoku'] = (ichimoku_conversion, ichimoku_base)
        
        # Calculate indicators
        if indicators:
            with st.spinner("Calculating technical indicators..."):
                try:
                    st.session_state.df = add_technical_indicators(
                        st.session_state.df,
                        st.session_state.y_column,
                        indicators
                    )
                    st.success("Technical indicators calculated successfully")
                except Exception as e:
                    st.error(f"Error calculating indicators: {str(e)}")
        else:
            st.warning("No indicators selected")
    
    # If we have indicators, show option to remove them
    if has_indicators:
        if st.button("Remove All Indicators"):
            # Keep only original columns
            indicator_cols = [col for col in st.session_state.df.columns 
                             if col.startswith(('SMA', 'EMA', 'RSI', 'MACD', 'BB_', 'ATR', 'ADX', 'Ichimoku'))]
            
            st.session_state.df = st.session_state.df.drop(columns=indicator_cols)
            st.success("All technical indicators removed") 