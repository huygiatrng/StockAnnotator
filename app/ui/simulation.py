"""
UI components for trading simulation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

from app.core.simulation import cached_simulate_trading


def render_simulation_controls():
    """Render UI controls for simulation parameters"""
    st.subheader("Simulation Parameters")
    
    # Trading budget
    st.session_state.initial_budget = st.number_input(
        "Initial Budget ($)", 
        min_value=1000, 
        max_value=1000000, 
        value=10000,
        step=1000
    )
    
    # Order size control
    order_size_col1, order_size_col2 = st.columns(2)
    
    with order_size_col1:
        st.session_state.order_size_type = st.selectbox(
            "Order Size Type",
            ["percent", "fixed", "contract"],
            index=0
        )
    
    with order_size_col2:
        if st.session_state.order_size_type == "percent":
            st.session_state.order_size_value = st.slider("Order Size (%)", 1, 100, 10)
        elif st.session_state.order_size_type == "fixed":
            st.session_state.order_size_value = st.number_input("Fixed $ Amount", min_value=100, value=1000, step=100)
        else:  # contracts
            st.session_state.order_size_value = st.number_input("Number of Contracts", min_value=1, value=1, step=1)
    
    # Slippage controls
    slippage_col1, slippage_col2 = st.columns(2)
    
    with slippage_col1:
        st.session_state.use_slippage = st.checkbox("Use Slippage", value=False)
    
    with slippage_col2:
        if st.session_state.use_slippage:
            st.session_state.slippage_pct = st.slider("Slippage (%)", 0.0, 1.0, 0.1, 0.05)
        else:
            st.session_state.slippage_pct = 0.0
            st.markdown("*Slippage disabled*")
    
    # Commission controls
    commission_col1, commission_col2 = st.columns(2)
    
    with commission_col1:
        st.session_state.use_commission = st.checkbox("Use Commission", value=False)
    
    with commission_col2:
        if st.session_state.use_commission:
            st.session_state.commission_type = st.selectbox("Commission Type", ["fixed", "percent"], index=0)
            
            if st.session_state.commission_type == "fixed":
                st.session_state.commission_value = st.number_input("Commission ($)", min_value=0.0, value=5.0, step=1.0)
            else:  # percent
                st.session_state.commission_value = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01)
        else:
            st.session_state.commission_type = "fixed"
            st.session_state.commission_value = 0.0
            st.markdown("*Commission disabled*")
    
    # Run simulation button
    if st.button("Run Simulation"):
        run_simulation()


def run_simulation():
    """Run the trading simulation and display results"""
    if not st.session_state.annotations:
        st.warning("Please add trading points before running simulation")
        return
    
    if not hasattr(st.session_state, 'df') or st.session_state.df is None:
        st.error("No data loaded. Please upload data first.")
        return
    
    with st.spinner("Running trading simulation..."):
        # Extract simulation parameters
        initial_budget = st.session_state.initial_budget
        order_size_type = st.session_state.order_size_type
        order_size_value = st.session_state.order_size_value
        
        # Extract slippage and commission parameters
        use_slippage = st.session_state.get('use_slippage', False)
        slippage_pct = st.session_state.get('slippage_pct', 0.0)
        use_commission = st.session_state.get('use_commission', False)
        commission_type = st.session_state.get('commission_type', 'fixed')
        commission_value = st.session_state.get('commission_value', 0.0)
        
        # Run simulation
        result = cached_simulate_trading(
            st.session_state.df,
            st.session_state.y_column,
            st.session_state.x_column,
            st.session_state.annotations,
            initial_budget,
            order_size_type,
            order_size_value,
            use_slippage,
            slippage_pct,
            use_commission,
            commission_type,
            commission_value
        )
        
        # Store results in session
        st.session_state.simulation_results = result
        
        # NOTE: We're not calling display_simulation_results here 
        # because results will be displayed in the main content area
        st.success("Simulation complete! Scroll down to see results.")


def display_simulation_results(result):
    """Display the simulation results in the main content area"""
    if not result or not result.get('trades'):
        st.warning("No trades were executed in the simulation")
        return
    
    st.header("Trading Simulation Results")
    
    # Check if this is auto-labeled data
    is_auto_labeled = result.get('is_auto_labeled', False)
    
    # If auto-labeled, show a message explaining the win rate
    if is_auto_labeled:
        st.info("""
        ### Auto-Labeled Data Detected
        This simulation uses auto-labeled data, which guarantees 100% win rate under perfect conditions.
        Any reduction in win rate is solely due to slippage and commission costs.
        """)
    
    # Calculate original win rate (without slippage/commission)
    theoretical_win_rate = result.get('theoretical_win_rate', 100.0) if is_auto_labeled else 0
    
    # Performance metrics with tabs for different views
    basic_tab, detailed_tab = st.tabs(["Basic Metrics", "Detailed Analysis"])
    
    with basic_tab:
        # Top level metrics row
        metrics_row = st.columns(4)
        with metrics_row[0]:
            st.metric("Initial Budget", f"${result['initial_budget']:.2f}")
        with metrics_row[1]:
            st.metric("Final Equity", f"${result['final_equity']:.2f}")
        with metrics_row[2]:
            profit_change = result['profit_percent']
            st.metric("Total Profit", f"${result['total_profit']:.2f}", f"{profit_change:.2f}%")
        with metrics_row[3]:
            st.metric("Number of Trades", result['num_trades'])
        
        # Win rate comparison
        win_rate_row = st.columns(3)
        with win_rate_row[0]:
            st.metric("Theoretical Win Rate", f"{theoretical_win_rate:.2f}%", 
                     help="Win rate without slippage/commission")
        with win_rate_row[1]:
            win_rate_change = result['win_rate'] - theoretical_win_rate
            st.metric("Actual Win Rate", f"{result['win_rate']:.2f}%", 
                     f"{win_rate_change:.2f}%" if win_rate_change != 0 else None,
                     delta_color="inverse" if win_rate_change < 0 else "normal",
                     help="Win rate after slippage/commission")
        with win_rate_row[2]:
            st.metric("Avg Profit/Trade", f"${result['avg_profit']:.2f}")
        
        # Transaction costs impact
        if 'total_commission' in result and 'total_slippage' in result:
            cost_row = st.columns(3)
            with cost_row[0]:
                st.metric("Total Commission", f"${result['total_commission']:.2f}")
            with cost_row[1]:
                st.metric("Total Slippage", f"${result['total_slippage']:.2f}")
            with cost_row[2]:
                total_costs = result['total_commission'] + result['total_slippage']
                cost_percent = (total_costs / result['initial_budget']) * 100
                st.metric("Total Costs", f"${total_costs:.2f}", f"{cost_percent:.2f}% of initial capital")
        
        # Risk metrics
        risk_row = st.columns(2)
        with risk_row[0]:
            st.metric("Max Drawdown", f"${result['max_drawdown']:.2f}")
        with risk_row[1]:
            st.metric("Max Drawdown %", f"{result['max_drawdown_percent']:.2f}%")
    
    with detailed_tab:
        # Add a toggle for buy and hold comparison
        show_buy_hold = st.checkbox("Compare with Buy & Hold Strategy", value=False)
        
        # Equity curve
        st.subheader("Equity Curve")
        
        if len(result['equity_curve']) > 1:
            # Create interactive equity chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=result['equity_curve']['date'],
                    y=result['equity_curve']['equity'],
                    mode='lines',
                    name='Strategy Equity',
                    line=dict(color='rgb(49,130,189)', width=2)
                )
            )
            
            # Add buy and hold comparison if requested
            if show_buy_hold and 'df' in st.session_state and st.session_state.df is not None:
                # Get dataframe and columns
                df = st.session_state.df
                price_col = st.session_state.y_column
                date_col = st.session_state.x_column
                
                # Calculate buy & hold equity curve properly aligned with strategy dates
                first_price = df[price_col].iloc[0]
                shares_bought = result['initial_budget'] / first_price
                
                # Create a date-to-price mapping from original dataframe
                date_to_price = dict(zip(df[date_col], df[price_col]))
                
                # Map strategy equity curve dates to corresponding prices
                # and calculate buy & hold equity at those exact same dates
                buy_hold_equity = []
                buy_hold_dates = []
                
                for date in result['equity_curve']['date']:
                    if date in date_to_price:
                        price = date_to_price[date]
                        equity = price * shares_bought
                        buy_hold_equity.append(equity)
                        buy_hold_dates.append(date)
                
                # Only add trace if we have data points
                if buy_hold_equity:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_hold_dates,
                            y=buy_hold_equity,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='rgb(214,39,40)', width=2, dash='dot')
                        )
                    )
                else:
                    st.warning("Could not align Buy & Hold data with strategy dates")
            
            # Add initial equity as horizontal line
            fig.add_hline(
                y=result['initial_budget'], 
                line_dash="dash", 
                line_color="gray",
                annotation_text="Initial Budget"
            )
            
            fig.update_layout(
                title="Equity Curve",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data points to plot equity curve")
        
        # Add a Drawdown chart
        st.subheader("Drawdown Chart")
        
        if len(result['equity_curve']) > 1:
            # Calculate drawdown series
            equity_data = result['equity_curve']['equity'].values
            peak = np.maximum.accumulate(equity_data)
            drawdown = (equity_data - peak) / peak * 100  # Convert to percentage
            
            # Create drawdown chart
            fig_dd = go.Figure()
            
            fig_dd.add_trace(
                go.Scatter(
                    x=result['equity_curve']['date'],
                    y=drawdown,
                    mode='lines',
                    name='Drawdown',
                    line=dict(color='rgb(214,39,40)', width=2),
                    fill='tozeroy'
                )
            )
            
            # Add max drawdown line
            fig_dd.add_hline(
                y=float(-result['max_drawdown_percent']),
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Max Drawdown: {result['max_drawdown_percent']:.2f}%"
            )
            
            fig_dd.update_layout(
                title="Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template="plotly_white",
                height=300,
                yaxis=dict(autorange=True)
            )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.info("Not enough data points to plot drawdown chart")
        
        # Add a Cumulative Profit chart
        st.subheader("Cumulative Profit Chart")
        
        if len(result['equity_curve']) > 1:
            # Calculate cumulative profit
            equity_data = result['equity_curve']['equity'].values
            initial_equity = result['initial_budget']
            cumulative_profit = equity_data - initial_equity
            cumulative_profit_pct = (equity_data / initial_equity - 1) * 100
            
            # Create profit chart with two y-axes
            fig_profit = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Absolute profit
            fig_profit.add_trace(
                go.Scatter(
                    x=result['equity_curve']['date'],
                    y=cumulative_profit,
                    mode='lines',
                    name='Profit ($)',
                    line=dict(color='rgb(49,130,189)', width=2)
                ),
                secondary_y=False
            )
            
            # Percentage profit
            fig_profit.add_trace(
                go.Scatter(
                    x=result['equity_curve']['date'],
                    y=cumulative_profit_pct,
                    mode='lines',
                    name='Profit (%)',
                    line=dict(color='rgb(0,100,0)', width=2, dash='dot')
                ),
                secondary_y=True
            )
            
            # Add zero line
            fig_profit.add_hline(
                y=0, 
                line_dash="dash", 
                line_color="gray"
            )
            
            fig_profit.update_layout(
                title="Cumulative Profit",
                xaxis_title="Date",
                template="plotly_white",
                height=350,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
            )
            
            # Update y-axis labels
            fig_profit.update_yaxes(title_text="Profit ($)", secondary_y=False)
            fig_profit.update_yaxes(title_text="Profit (%)", secondary_y=True)
            
            st.plotly_chart(fig_profit, use_container_width=True)
            
            # Add performance comparison if buy & hold is enabled
            if show_buy_hold and 'df' in st.session_state and st.session_state.df is not None:
                # Calculate buy & hold performance
                df = st.session_state.df
                price_col = st.session_state.y_column
                
                first_price = df[price_col].iloc[0]
                last_price = df[price_col].iloc[-1]
                
                buy_hold_return = (last_price / first_price - 1) * 100
                strategy_return = result['profit_percent']
                
                # Create a comparison box
                comparison_col1, comparison_col2 = st.columns(2)
                
                with comparison_col1:
                    st.metric("Strategy Return", f"{strategy_return:.2f}%")
                
                with comparison_col2:
                    relative_performance = strategy_return - buy_hold_return
                    st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%", 
                             f"Strategy {'+' if relative_performance > 0 else ''}{relative_performance:.2f}%",
                             delta_color="normal" if relative_performance > 0 else "inverse")
        else:
            st.info("Not enough data points to plot profit chart")
    
    # Trades table
    st.subheader("Trade Details")
    
    trades_df = pd.DataFrame(result['trades'])
    
    if len(trades_df) > 0:
        # Handle large dataframes - check if we might exceed Pandas Styler limits
        num_cells = len(trades_df) * len(trades_df.columns)
        MAX_STYLER_CELLS = 250000  # Slightly below the 262144 limit
        
        if num_cells > MAX_STYLER_CELLS:
            st.warning(f"Large trade dataset detected ({len(trades_df)} trades). Displaying limited view to avoid performance issues.")
            
            # Offer pagination or filtering options
            display_options = st.columns([2, 1])
            with display_options[0]:
                display_mode = st.radio(
                    "Display mode:",
                    ["First 1000 trades", "Last 1000 trades", "Sample of trades", "Unstyled full dataset"],
                    horizontal=True
                )
            
            with display_options[1]:
                if display_mode == "Sample of trades":
                    sample_size = st.slider("Sample size", 100, 1000, 500, 100)
            
            # Apply the selected display mode
            if display_mode == "First 1000 trades":
                display_df = trades_df.head(1000)
            elif display_mode == "Last 1000 trades":
                display_df = trades_df.tail(1000)
            elif display_mode == "Sample of trades":
                display_df = trades_df.sample(min(sample_size, len(trades_df)))
            else:  # Unstyled full dataset
                # Use regular dataframe display without styling for the full dataset
                st.dataframe(trades_df, use_container_width=True)
                # Skip the styling part and continue with the rest of the function
                full_display = True
        else:
            display_df = trades_df
            full_display = False
        
        # Calculate additional metrics for each trade and apply styling
        if not full_display and 'profit' in display_df.columns:
            display_df['win'] = display_df['profit'] > 0
            
            # Add impact of slippage and commission if available
            if 'commission' in display_df.columns and 'slippage' in display_df.columns:
                display_df['transaction_costs'] = display_df['commission'] + display_df['slippage']
                display_df['would_be_profitable'] = (display_df['profit'] + display_df['transaction_costs']) > 0
            
            # Style the dataframe
            styled_df = display_df.style.format({
                'price': '${:.2f}',
                'value': '${:.2f}',
                'cash': '${:.2f}',
                'equity': '${:.2f}',
                'profit': '${:.2f}',
                'profit_pct': '{:.2f}%',
                'commission': '${:.2f}',
                'slippage': '${:.2f}',
                'transaction_costs': '${:.2f}' if 'transaction_costs' in display_df.columns else None
            })
            
            # Apply conditional styling for win/loss
            styled_df = styled_df.apply(
                lambda x: ['background-color: #d4f7d4' if v else 'background-color: #ffcccb' 
                          for v in x.win] if 'win' in x else [''] * len(x),
                axis=0
            )
            
            st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No trades were executed")
    
    # Trade distribution analysis
    st.subheader("Trade Analysis")
    
    if 'profit' in trades_df.columns and len(trades_df) > 0:
        profit_col, win_col = st.columns(2)
        
        with profit_col:
            # Profit distribution
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=trades_df['profit'],
                    nbinsx=20,
                    marker_color='rgba(49,130,189, 0.7)'
                )
            )
            
            fig.update_layout(
                title="Profit Distribution",
                xaxis_title="Profit ($)",
                yaxis_title="Count",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with win_col:
            # Win/loss pie chart
            wins = (trades_df['profit'] > 0).sum()
            losses = (trades_df['profit'] <= 0).sum()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Winning Trades', 'Losing Trades'],
                    values=[wins, losses],
                    marker=dict(colors=['rgba(76,175,80,0.8)', 'rgba(255,87,87,0.8)']),
                    hole=.3
                )
            ])
            
            fig.update_layout(
                title="Trade Outcomes"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display transaction costs impact if available
            if 'would_be_profitable' in trades_df.columns:
                actual_wins = wins
                potential_wins = trades_df['would_be_profitable'].sum()
                win_rate_impact = (potential_wins - actual_wins) / len(trades_df) * 100
                
                st.info(f"Transaction costs reduced win rate by approximately {win_rate_impact:.2f}%. {int(potential_wins - actual_wins)} trades would have been profitable without slippage and commission.") 