"""
UI components for data loading
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

from app.core.data_loader import load_file, detect_date_column


def render_data_loader():
    """Render UI controls for data loading"""
    st.subheader("Load Data")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading data..."):
                # Load the data
                df = load_file(uploaded_file)
                
                if df is not None and not df.empty:
                    # Store in session state
                    st.session_state.df = df
                    
                    # Try to detect date column
                    date_column = detect_date_column(df)
                    
                    # Column selectors for X and Y axes
                    st.session_state.x_column = st.selectbox(
                        "Select X-axis column (date)", 
                        options=df.columns.tolist(),
                        index=df.columns.get_loc(date_column) if date_column in df.columns else 0
                    )
                    
                    # Try to find a likely price column
                    price_candidates = ['Close', 'Price', 'close', 'price', 'Adj Close', 'value']
                    default_y = next((col for col in price_candidates if col in df.columns), 
                                     df.columns[1] if len(df.columns) > 1 else df.columns[0])
                    
                    st.session_state.y_column = st.selectbox(
                        "Select Y-axis column (price)", 
                        options=df.columns.tolist(),
                        index=df.columns.get_loc(default_y) if default_y in df.columns else 0
                    )
                    
                    # Display data preview - using a container instead of an expander to avoid nesting
                    st.checkbox("Show Data Preview", key="show_data_preview")
                    if st.session_state.show_data_preview:
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Display summary statistics
                        st.markdown("**Data Summary**")
                        st.write(f"- Rows: {df.shape[0]:,}")
                        st.write(f"- Columns: {df.shape[1]:,}")
                        st.write(f"- Date range: {df[st.session_state.x_column].min()} to {df[st.session_state.x_column].max()}")
                        
                    st.success(f"Data loaded successfully: {df.shape[0]:,} rows, {df.shape[1]:,} columns")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        # Clear data if no file is uploaded
        if 'df' in st.session_state and st.session_state.df is not None:
            st.info("Upload a new file to replace the current data")
            
            # Show current data info
            if st.session_state.df is not None:
                st.write(f"Current data: {st.session_state.df.shape[0]:,} rows, {st.session_state.df.shape[1]:,} columns")
                
                # Option to clear data
                if st.button("Clear Data"):
                    st.session_state.df = None
                    st.session_state.x_column = None
                    st.session_state.y_column = None
                    st.session_state.annotations = []
                    st.experimental_rerun() 