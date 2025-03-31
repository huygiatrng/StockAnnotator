"""
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

# Import app modules
from app.ui.data_loader import render_data_loader
from app.ui.chart import render_chart
from app.ui.label_manager import render_labeling_controls, render_auto_labeling, handle_chart_interaction, display_label_stats, export_labels
from app.ui.indicators import render_indicator_controls
from app.ui.simulation import render_simulation_controls, display_simulation_results
from app.utils.gpu_helpers import check_gpu


def run_app():
    """Main application entry point"""
    # App configuration
    st.set_page_config(
        page_title="Stock Label Trading Simulator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check for GPU
    check_gpu()
    
    # Initialize session state variables
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    
    if 'x_column' not in st.session_state:
        st.session_state.x_column = None
    
    if 'y_column' not in st.session_state:
        st.session_state.y_column = None
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    if 'click_mode' not in st.session_state:
        st.session_state.click_mode = 'add'
    
    if 'default_label' not in st.session_state:
        st.session_state.default_label = 'long'
    
    if 'selected_points' not in st.session_state:
        st.session_state.selected_points = []
    
    if 'frequency_threshold' not in st.session_state:
        st.session_state.frequency_threshold = 0.5
    
    if 'min_profit_threshold' not in st.session_state:
        st.session_state.min_profit_threshold = 0.5
        
    # Initialize slippage and commission settings
    if 'use_slippage' not in st.session_state:
        st.session_state.use_slippage = False
        
    if 'slippage_pct' not in st.session_state:
        st.session_state.slippage_pct = 0.1
        
    if 'use_commission' not in st.session_state:
        st.session_state.use_commission = False
        
    if 'commission_type' not in st.session_state:
        st.session_state.commission_type = 'fixed'
        
    if 'commission_value' not in st.session_state:
        st.session_state.commission_value = 5.0
    
    # App title
    st.title("Stock Label Trading Simulator")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # Data loading section
        with st.expander("Data Loading", expanded=True):
            render_data_loader()
        
        # Labeling controls REMOVED FROM SIDEBAR
        
        # Technical indicators
        with st.expander("Technical Indicators", expanded=False):
            render_indicator_controls()
        
        # Remove Trading Simulation from sidebar
    
    # Main content area
    if st.session_state.df is not None:
        # Chart area
        selected_points = render_chart()
        
        # Handle chart interactions
        if selected_points:
            st.session_state.selected_points = selected_points
            if handle_chart_interaction(selected_points):
                st.rerun()
        
        # ADD LABELING CONTROLS TO MAIN CONTENT AREA
        # Use a consistent layout for labeling section
        labeling_col1, labeling_col2 = st.columns([1, 1])
        
        with labeling_col1:
            # Basic labeling controls in a collapsed expander
            with st.expander("Labeling Controls", expanded=False):
                render_labeling_controls()
                display_label_stats()
        
        with labeling_col2:
            # Export labels if annotations exist
            if st.session_state.annotations:
                with st.expander("Export Labels", expanded=False):
                    export_labels()
        
        # Create a two-column layout for Smart Auto-Labeling and Trading Simulation
        auto_label_col, simulation_col = st.columns([1, 1])
        
        # Add auto-labeling to the left column
        with auto_label_col:
            with st.expander("Smart Auto-Labeling", expanded=False):
                render_auto_labeling()
        
        # Add trading simulation to the right column
        with simulation_col:
            with st.expander("Trading Simulation", expanded=False):
                render_simulation_controls()
        
        # Display simulation results if available
        if 'simulation_results' in st.session_state:
            display_simulation_results(st.session_state.simulation_results)
    else:
        st.info("Please upload data to begin")
        
        # Display sample instructions
        st.markdown("""
        ## How to use this app
        
        1. Upload a CSV file with price data using the sidebar
        2. Choose the date and price columns
        3. Add labels manually by clicking on the chart or use auto-labeling
        4. Run a trading simulation to see performance
        
        This application allows you to:
        - Label price points as 'long', 'short', or 'neutral'  
        - Use smart auto-labeling to find optimal trade points
        - Add technical indicators to aid in analysis
        - Run trading simulations based on your labels with realistic slippage and commission
        - Evaluate trading performance with metrics and charts
        """)
    
    # Footer
    st.markdown("---")
    st.caption("📊 Stock Label Trading Simulator • Built with Streamlit")


if __name__ == "__main__":
    run_app() 