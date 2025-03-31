"""
UI components for chart rendering and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Try to import plotly_events, which is used for chart interactions
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    # If not available, provide a fallback
    def plotly_events(fig, **kwargs):
        st.plotly_chart(fig, use_container_width=True)
        st.warning("Interactive chart selection requires streamlit-plotly-events package.")
        return []


def render_chart():
    """Render the main price chart with annotations"""
    if st.session_state.df is None or st.session_state.x_column is None or st.session_state.y_column is None:
        st.warning("Please load data and select columns first")
        return None
    
    # Create the chart figure
    fig = create_price_chart(
        st.session_state.df, 
        st.session_state.x_column, 
        st.session_state.y_column, 
        st.session_state.annotations
    )
    
    # Display using plotly_events for interactivity
    selected_points = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=600,
        override_width="100%",
        key=f"price_chart_{int(time.time())}"  # Unique key to force refresh
    )
    
    return selected_points


def create_price_chart(df, x_column, y_column, annotations=None):
    """
    Create an interactive price chart with annotations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe containing price data
    x_column : str
        Column name for x-axis (dates)
    y_column : str
        Column name for y-axis (prices)
    annotations : list, optional
        List of annotation points with x, y, and label keys
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The chart figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df[x_column],
            y=df[y_column],
            mode='lines',
            name=y_column,
            line=dict(color='rgba(49, 130, 189, 0.9)', width=1.5),
            hovertemplate=
                f"{x_column}: %{{x}}<br>" +
                f"{y_column}: %{{y:.2f}}<br>" +
                "<extra></extra>"
        )
    )
    
    # Add annotations if provided
    if annotations and len(annotations) > 0:
        # Create separate traces for each label type
        long_points = [p for p in annotations if p.get('label') == 'long']
        short_points = [p for p in annotations if p.get('label') == 'short']
        neutral_points = [p for p in annotations if p.get('label') == 'neutral']
        
        # Long points (green triangles up)
        if long_points:
            fig.add_trace(
                go.Scatter(
                    x=[p['x'] for p in long_points],
                    y=[p['y'] for p in long_points],
                    mode='markers',
                    name='Long',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='rgba(76,175,80,0.9)',
                        line=dict(color='rgba(76,175,80,1.0)', width=1)
                    ),
                    hovertemplate=
                        f"{x_column}: %{{x}}<br>" +
                        f"{y_column}: %{{y:.2f}}<br>" +
                        "Label: Long<br>" +
                        "<extra></extra>"
                )
            )
        
        # Short points (red triangles down)
        if short_points:
            fig.add_trace(
                go.Scatter(
                    x=[p['x'] for p in short_points],
                    y=[p['y'] for p in short_points],
                    mode='markers',
                    name='Short',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='rgba(255,87,87,0.9)',
                        line=dict(color='rgba(255,87,87,1.0)', width=1)
                    ),
                    hovertemplate=
                        f"{x_column}: %{{x}}<br>" +
                        f"{y_column}: %{{y:.2f}}<br>" +
                        "Label: Short<br>" +
                        "<extra></extra>"
                )
            )
        
        # Neutral points (yellow circles)
        if neutral_points:
            fig.add_trace(
                go.Scatter(
                    x=[p['x'] for p in neutral_points],
                    y=[p['y'] for p in neutral_points],
                    mode='markers',
                    name='Neutral',
                    marker=dict(
                        symbol='circle',
                        size=10,
                        color='rgba(255,193,7,0.9)',
                        line=dict(color='rgba(255,193,7,1.0)', width=1)
                    ),
                    hovertemplate=
                        f"{x_column}: %{{x}}<br>" +
                        f"{y_column}: %{{y:.2f}}<br>" +
                        "Label: Neutral<br>" +
                        "<extra></extra>"
                )
            )
    
    # Layout configuration
    fig.update_layout(
        title=f"{y_column} vs {x_column}",
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Add marker click instructions
    fig.add_annotation(
        text=f"Click Mode: {st.session_state.click_mode.capitalize()} • Default Label: {st.session_state.default_label.capitalize()}",
        xref="paper", yref="paper",
        x=0.5, y=1.06,
        showarrow=False
    )
    
    # Make sure y-axis includes a bit of padding
    y_min = df[y_column].min() * 0.98
    y_max = df[y_column].max() * 1.02
    fig.update_yaxes(range=[y_min, y_max])
    
    # Enable clicking
    fig.update_layout(clickmode='event')
    
    return fig 