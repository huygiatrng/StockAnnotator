"""
Chart creation and visualization utilities
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def create_line_chart(df, x_column, y_column, annotations, previous_layout=None):
    """
    Create an interactive line chart with annotations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price data
    x_column : str
        Column to use for x-axis
    y_column : str
        Column to use for y-axis
    annotations : list
        List of annotation dictionaries
    previous_layout : dict
        Previous chart layout to preserve zoom level
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create the figure
    fig = go.Figure()
    
    # Add the price line
    fig.add_trace(go.Scatter(
        x=df[x_column],
        y=df[y_column],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Filter annotations by type
    long_points = [ann for ann in annotations if ann['label'] == 'long']
    short_points = [ann for ann in annotations if ann['label'] == 'short']
    exit_points = [ann for ann in annotations if ann['label'] == 'neutral']
    
    # Add buy (long) points
    if long_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in long_points],
            y=[point['y'] for point in long_points],
            mode='markers',
            name='Long',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    # Add sell short points
    if short_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in short_points],
            y=[point['y'] for point in short_points],
            mode='markers',
            name='Short',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Add exit points
    if exit_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in exit_points],
            y=[point['y'] for point in exit_points],
            mode='markers',
            name='Exit',
            marker=dict(color='blue', size=10, symbol='circle')
        ))
        
    # Set layout (preserving previous state if available)
    if previous_layout and all(key in previous_layout for key in ['xaxis.range[0]', 'xaxis.range[1]', 'yaxis.range[0]', 'yaxis.range[1]']):
        fig.update_layout(
            title=f"{y_column} vs {x_column}",
            xaxis=dict(
                title=x_column,
                range=[previous_layout['xaxis.range[0]'], previous_layout['xaxis.range[1]']]
            ),
            yaxis=dict(
                title=y_column,
                range=[previous_layout['yaxis.range[0]'], previous_layout['yaxis.range[1]']]
            ),
            height=600,
            hovermode='closest'
        )
    else:
        fig.update_layout(
            title=f"{y_column} vs {x_column}",
            xaxis=dict(title=x_column),
            yaxis=dict(title=y_column),
            height=600,
            hovermode='closest'
        )
        
    return fig


@st.cache_data(ttl=3600, hash_funcs={pd.DataFrame: lambda df: hash(str(df.shape) + str(df.columns.tolist()))})
def create_technical_chart(df, x_column, price_column, annotations):
    """
    Create chart with technical indicators and annotations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with price and indicator data
    x_column : str
        Column to use for x-axis
    price_column : str
        Column containing price data
    annotations : list
        List of annotation dictionaries
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    # Create the figure with subplots
    fig = go.Figure()
    
    # Add the price line
    fig.add_trace(go.Scatter(
        x=df[x_column],
        y=df[price_column],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add Bollinger Bands if available
    if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df['bb_upper'],
            mode='lines',
            name='Upper BB',
            line=dict(color='rgba(200, 200, 200, 0.7)', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df['bb_middle'],
            mode='lines',
            name='Middle BB',
            line=dict(color='rgba(150, 150, 150, 0.7)', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df['bb_lower'],
            mode='lines',
            name='Lower BB',
            line=dict(color='rgba(200, 200, 200, 0.7)', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(200, 200, 200, 0.1)'
        ))
    
    # Add Moving Averages if available
    if 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1)
        ))
        
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df[x_column],
            y=df['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='purple', width=1)
        ))
    
    # Filter annotations by type and add markers
    long_points = [ann for ann in annotations if ann['label'] == 'long']
    short_points = [ann for ann in annotations if ann['label'] == 'short']
    exit_points = [ann for ann in annotations if ann['label'] == 'neutral']
    
    # Add buy (long) points
    if long_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in long_points],
            y=[point['y'] for point in long_points],
            mode='markers',
            name='Long',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ))
    
    # Add sell short points
    if short_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in short_points],
            y=[point['y'] for point in short_points],
            mode='markers',
            name='Short',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ))
    
    # Add exit points
    if exit_points:
        fig.add_trace(go.Scatter(
            x=[point['x'] for point in exit_points],
            y=[point['y'] for point in exit_points],
            mode='markers',
            name='Exit',
            marker=dict(color='blue', size=10, symbol='circle')
        ))
    
    # Set layout
    fig.update_layout(
        title=f"Technical Analysis Chart",
        xaxis=dict(title=x_column),
        yaxis=dict(title=price_column),
        height=600,
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig 