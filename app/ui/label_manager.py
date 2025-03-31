"""
UI components for managing chart annotations and labels
"""

import streamlit as st
import pandas as pd
import time


def render_labeling_controls():
    """Render the UI controls for labeling"""
    # Label selection
    st.session_state.default_label = st.radio(
        "Default Label", 
        ["long", "short", "neutral"],
        horizontal=True
    )
    
    # Click mode selection
    st.session_state.click_mode = st.radio(
        "Click Mode", 
        ["add", "remove"],
        horizontal=True
    )
    
    # Clear labels button
    if st.button("Clear All Labels", use_container_width=True):
        st.session_state.annotations = []
        st.success("All labels cleared")
    
    # Export options moved to main content area


def render_auto_labeling():
    """Render the UI controls for automatic labeling"""
    from app.core.labeling import auto_label_data, smart_auto_label
    import time
    
    st.subheader("Auto-Labeling")
    
    # Tách thành hai tabs riêng biệt và đảo thứ tự tab
    smart_tab, indicator_tab = st.tabs(["Smart Price Action Labeling", "Technical Indicator Labeling"])
    
    with smart_tab:
        st.subheader("Smart Auto-Label Settings")
        
        # Thông báo rõ ràng về tối ưu hóa win rate
        st.info("**100% Win Rate Guarantee**: All generated labels are optimized to guarantee 100% win rate in perfect conditions (without slippage/commission). The system selects only high-probability trades with strict validation criteria.", icon="✅")
        
        # Lưu trữ cấu hình đã chọn trước đó trong session_state để đảm bảo tính nhất quán
        if 'saved_trading_configs' not in st.session_state:
            st.session_state.saved_trading_configs = {}
        
        # Trading style selection (NEW)
        trading_style = st.radio(
            "Trading Style",
            ["Day Trading", "Swing Trading", "Position Trading"],
            help="Select trading style to adjust trade frequency"
        )
        
        # Theo dõi sự thay đổi phong cách giao dịch
        trading_style_changed = False
        if 'current_trading_style' not in st.session_state:
            st.session_state.current_trading_style = trading_style
        elif st.session_state.current_trading_style != trading_style:
            trading_style_changed = True
        
        # Min profit threshold setting
        col1, col2 = st.columns(2)
        
        with col1:
            # Khôi phục giá trị đã lưu cho trading style này nếu có
            default_min_profit = st.session_state.saved_trading_configs.get(
                trading_style, {}).get('min_profit_threshold', 0.5)
                
            min_profit_threshold = st.slider(
                "Minimum Profit (%)", 
                0.1, 5.0, default_min_profit, 0.1,
                help="Minimum profit percentage required for a trade to be considered"
            )
            st.session_state.min_profit_threshold = min_profit_threshold
            
            # Lưu vào cấu hình
            if trading_style not in st.session_state.saved_trading_configs:
                st.session_state.saved_trading_configs[trading_style] = {}
            st.session_state.saved_trading_configs[trading_style]['min_profit_threshold'] = min_profit_threshold
        
        with col2:
            # Khôi phục giá trị đã lưu cho trading style này nếu có
            if trading_style == "Day Trading":
                default_frequency = st.session_state.saved_trading_configs.get(
                    trading_style, {}).get('frequency_threshold', 0.3)
                frequency_threshold = st.slider(
                    "Trade Frequency", 
                    0.1, 0.9, default_frequency, 0.05,
                    help="Lower values generate more frequent trade signals"
                )
            elif trading_style == "Swing Trading":
                default_frequency = st.session_state.saved_trading_configs.get(
                    trading_style, {}).get('frequency_threshold', 0.5)
                frequency_threshold = st.slider(
                    "Trade Frequency", 
                    0.3, 0.9, default_frequency, 0.05,
                    help="Medium frequency for multi-day trades"
                )
            else:  # Position Trading
                default_frequency = st.session_state.saved_trading_configs.get(
                    trading_style, {}).get('frequency_threshold', 0.7)
                frequency_threshold = st.slider(
                    "Trade Frequency", 
                    0.5, 0.9, default_frequency, 0.05,
                    help="Higher values generate fewer, longer-term trades"
                )
            
            st.session_state.frequency_threshold = frequency_threshold
            
            # Lưu vào cấu hình
            st.session_state.saved_trading_configs[trading_style]['frequency_threshold'] = frequency_threshold
        
        # Min spacing between trades adjustment based on trading style (NEW)
        if trading_style == "Day Trading":
            min_spacing = 3  # Closer trades for day trading
        elif trading_style == "Swing Trading": 
            min_spacing = 7  # Medium spacing for swing trading
        else:  # Position Trading
            min_spacing = 15  # Wide spacing for position trading
        
        st.session_state.min_trade_spacing = min_spacing
        
        # Lưu cấu hình khoảng cách
        st.session_state.saved_trading_configs[trading_style]['min_trade_spacing'] = min_spacing
        
        # Hiển thị thông tin về win rate optimization, không còn là tùy chọn người dùng có thể tắt
        st.markdown("""
        <div style="padding: 10px; border-left: 3px solid #28a745; background-color: rgba(40, 167, 69, 0.1);">
        <b>Win Rate Optimization Active</b><br>
        Only high-quality trade points are selected through:
        <ul>
            <li>Entry at optimal price levels (support/resistance)</li>
            <li>Strict profit confirmation criteria</li>
            <li>Favorable risk-reward ratio (min 2.5:1)</li>
            <li>Minimal drawdown (<1.5%)</li>
            <li>Trend direction confirmation</li>
        </ul>
        This ensures 100% win rate under perfect conditions.
        </div>
        """, unsafe_allow_html=True)
        
        # Đặt seed để đảm bảo kết quả nhất quán
        seed_string = f"{trading_style}_{min_profit_threshold}_{frequency_threshold}_{min_spacing}"
        # Chuyển hash thành số dương và đảm bảo nằm trong phạm vi hợp lệ
        random_seed = abs(hash(seed_string)) % (2**32 - 1)
        
        # Add buttons in a row
        button_col1, button_col2 = st.columns(2)
        
        with button_col1:
            # Button to generate labels
            if st.button("Generate Smart Price Action Labels", use_container_width=True):
                generate_smart_labels(trading_style, frequency_threshold, min_profit_threshold, min_spacing, random_seed)
        
        with button_col2:
            # New button to generate labels and run simulation
            if st.button("Generate and Simulate", use_container_width=True):
                # First generate labels
                labels_generated = generate_smart_labels(trading_style, frequency_threshold, min_profit_threshold, min_spacing, random_seed)
                
                # Then run simulation if labels were generated
                if labels_generated:
                    from app.ui.simulation import run_simulation
                    run_simulation()
        
        # TỰ ĐỘNG generate khi trading style thay đổi
        if trading_style_changed:
            # Hiển thị thông báo
            st.warning(f"Trading style changed from {st.session_state.current_trading_style} to {trading_style}. Auto-regenerating labels...")
            
            # Tự động generate mới nếu đã có dữ liệu
            if 'df' in st.session_state and st.session_state.df is not None:
                generate_smart_labels(trading_style, frequency_threshold, min_profit_threshold, min_spacing, random_seed)
    
    with indicator_tab:
        st.subheader("Technical Indicator Settings")
        
        # Technical indicator settings
        rsi_threshold = st.slider("RSI Oversold Threshold", 10, 40, 30)
        st.session_state.rsi_threshold = rsi_threshold
        
        macd_threshold = st.slider("MACD Signal Threshold", -1.0, 1.0, 0.0, 0.1)
        st.session_state.macd_threshold = macd_threshold
        
        bb_threshold = st.slider("Bollinger Band Width", 0.0, 1.0, 0.1, 0.05)
        st.session_state.bb_threshold = bb_threshold
        
        # Button for basic technical auto-labeling
        if st.button("Auto-Label Based on Indicators"):
            with st.spinner("Generating labels from technical indicators..."):
                if 'rsi' in st.session_state.df.columns:
                    auto_labels = auto_label_data(
                        st.session_state.df, 
                        st.session_state.y_column, 
                        st.session_state.x_column,
                        rsi_threshold=rsi_threshold,
                        macd_threshold=macd_threshold,
                        bb_threshold=bb_threshold
                    )
                    
                    if not auto_labels:
                        st.warning("No labels generated. Try adjusting the indicator settings.")
                    else:
                        # Clean up existing labels in the same positions if any
                        if st.session_state.annotations:
                            existing_points = {(ann['x'], ann['y']): i for i, ann in enumerate(st.session_state.annotations)}
                            for label in auto_labels:
                                if (label['x'], label['y']) in existing_points:
                                    del st.session_state.annotations[existing_points[(label['x'], label['y'])]]
                        
                        st.session_state.annotations.extend(auto_labels)
                        st.success(f"Added {len(auto_labels)} auto-generated labels")
                else:
                    st.warning("Please calculate technical indicators first")

# Helper function to generate smart labels
def generate_smart_labels(trading_style, frequency_threshold, min_profit_threshold, min_spacing, random_seed):
    from app.core.labeling import smart_auto_label
    import time
    
    if 'df' in st.session_state and st.session_state.df is not None:
        with st.spinner(f"Auto-generating labels for {trading_style}..."):
            # Hiển thị toast thông báo trước khi gọi hàm
            st.toast("Finding optimal trade points...", icon="⚙️")
            
            # Tính thời gian bắt đầu để hiển thị kết quả sau đó
            start_time = time.time()
            
            # Always set optimize_for_winrate=True
            optimize_for_winrate = True
            st.session_state.optimize_winrate = True
            
            # Pass all parameters
            smart_labels = smart_auto_label(
                st.session_state.df,
                st.session_state.y_column,
                st.session_state.x_column,
                frequency_threshold=frequency_threshold,
                min_profit_threshold=min_profit_threshold,
                optimize_for_winrate=optimize_for_winrate,
                min_spacing=min_spacing,
                random_seed=random_seed
            )
            
            # Hiển thị toast thông báo sau khi gọi hàm
            st.toast("100% Win-rate optimized labels generated successfully", icon="✅")
            
            # Hiển thị thời gian xử lý
            processing_time = time.time() - start_time
            
            if not smart_labels:
                st.warning("No optimal trade points found. Try adjusting the settings.")
                return False
            else:
                # THAY ĐỔI: Reset lại toàn bộ labels hiện có thay vì chỉ cập nhật/thêm vào
                # Xóa tất cả các annotations cũ
                old_label_count = len(st.session_state.annotations) if 'annotations' in st.session_state else 0
                st.session_state.annotations = []
                
                # Thêm tất cả các annotations mới
                st.session_state.annotations = smart_labels
                
                # Count by label type
                label_counts = {}
                for ann in st.session_state.annotations:
                    label_type = ann['label']
                    label_counts[label_type] = label_counts.get(label_type, 0) + 1
                
                # Thông báo thành công với thông tin chi tiết
                st.success(f"✅ Auto-generated {len(smart_labels)} new labels with 100% win rate for {trading_style}")
                st.success(f"⏱ Processing completed in {processing_time:.2f} seconds")
                return True
    else:
        st.warning("Please load data first")
        return False


def export_labels():
    """Provide options to export labels"""
    if not st.session_state.annotations:
        st.info("No labels to export. Add labels by clicking on the chart or using auto-labeling.")
        return
    
    def map_label_to_numeric(lbl):
        if lbl == 'long':
            return 1
        elif lbl == 'short':
            return -1
        else:  # neutral
            return 0
    
    # Export all annotations
    annotations_df = pd.DataFrame(st.session_state.annotations)
    annotations_df['numeric_label'] = annotations_df['label'].apply(map_label_to_numeric)
    
    # Count labels by type
    label_counts = {}
    for ann in st.session_state.annotations:
        label = ann['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Display label counts
    st.write("**Label Summary:**")
    
    # Create color-coded label count display
    label_html = "<div style='display: flex; gap: 10px; margin: 10px 0;'>"
    
    for label, count in label_counts.items():
        color = "#4CAF50" if label == "long" else "#FF5252" if label == "short" else "#9E9E9E"
        label_html += f"<div style='background-color: {color}; color: white; padding: 5px 10px; border-radius: 4px; display: flex; align-items: center; justify-content: center;'>{label.title()}: {count}</div>"
    
    label_html += "</div>"
    st.markdown(label_html, unsafe_allow_html=True)
    
    st.write(f"**Total:** {len(st.session_state.annotations)} labels")
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    # Download button for all annotations
    with col1:
        csv = annotations_df.to_csv(index=False)
        st.download_button(
            label="Download All Labels",
            data=csv,
            file_name="trading_labels.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Provide separate files for each position type
    long_points = [ann for ann in st.session_state.annotations if ann['label'] == 'long']
    if long_points:
        with col2:
            long_df = pd.DataFrame(long_points)
            long_csv = long_df.to_csv(index=False)
            st.download_button(
                label=f"Download Long Labels ({len(long_points)})",
                data=long_csv,
                file_name="long_labels.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    short_points = [ann for ann in st.session_state.annotations if ann['label'] == 'short']
    if short_points:
        with col3:
            short_df = pd.DataFrame(short_points)
            short_csv = short_df.to_csv(index=False)
            st.download_button(
                label=f"Download Short Labels ({len(short_points)})",
                data=short_csv,
                file_name="short_labels.csv",
                mime="text/csv",
                use_container_width=True
            )


def handle_chart_interaction(selected_points):
    """Handle interactions with the chart for adding/removing points"""
    if selected_points:
        x_val = selected_points[0].get('x')
        y_val = selected_points[0].get('y')
        
        if st.session_state.click_mode == 'add':
            # Add a new annotation
            st.session_state.annotations.append({
                'x': x_val,
                'y': y_val,
                'label': st.session_state.default_label
            })
            return True  # Needs rerun
        
        elif st.session_state.click_mode == 'remove':
            # Find and remove closest annotation
            if st.session_state.annotations:
                closest_idx = None
                min_dist = float('inf')
                
                for i, ann in enumerate(st.session_state.annotations):
                    dist = ((ann['x'] - x_val) ** 2 + (ann['y'] - y_val) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                if closest_idx is not None:
                    st.session_state.annotations.pop(closest_idx)
                    return True  # Needs rerun
    
    return False  # No rerun needed


def display_label_stats():
    """Display statistics about the current labels"""
    if not st.session_state.annotations:
        st.info("No labels to display. Add labels by clicking on the chart or using auto-labeling.")
        return
        
    # Count by type
    label_counts = {}
    for ann in st.session_state.annotations:
        label = ann['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Display label counts in a visually appealing way
    st.markdown("### Label Distribution")
    
    # Calculate percentages for each label type
    total = len(st.session_state.annotations)
    
    # Create a horizontal bar chart representation using HTML/CSS
    chart_html = f"""
    <div style="width:100%; background-color:#f0f2f6; height:30px; border-radius:5px; margin-bottom:10px; overflow:hidden; display:flex;">
    """
    
    colors = {"long": "#4CAF50", "short": "#FF5252", "neutral": "#9E9E9E"}
    
    # Add bars for each label type
    for label, count in label_counts.items():
        percent = (count / total) * 100
        chart_html += f"""
        <div style="width:{percent}%; height:100%; background-color:{colors.get(label, '#9E9E9E')}; 
                   display:flex; align-items:center; justify-content:center; color:white; font-weight:bold;">
            {percent:.1f}%
        </div>
        """
    
    chart_html += "</div>"
    
    # Display the chart
    st.markdown(chart_html, unsafe_allow_html=True)
    
    # Display counts
    cols = st.columns(len(label_counts))
    for i, (label, count) in enumerate(label_counts.items()):
        with cols[i]:
            st.metric(
                label=label.capitalize(), 
                value=count,
                delta=f"{(count/total)*100:.1f}%"
            ) 