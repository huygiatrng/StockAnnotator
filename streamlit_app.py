import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Stock Data Labeler", page_icon="📈")

# Custom CSS to center content within containers
st.markdown("""
<style>
.stApp {
    max-width: 1400px;
    margin: 0 auto;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 0rem;
    padding-right: 0rem;
}
.stButton > button {
    display: block;
    margin: 0 auto;
}
.stDownloadButton > button {
    display: block;
    margin: 0 auto;
}
div[data-testid="stDataFrame"] > div {
    display: flex;
    justify-content: center;
}
button.selected {
    color: red !important;
}
</style>
""", unsafe_allow_html=True)


def load_data(file):
    return pd.read_csv(file)


def find_date_column(df):
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        return date_columns[0]
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'index' in col.lower()]
    if len(date_columns) > 0:
        return date_columns[0]
    return df.columns[0]


def create_line_chart(df, x_column, y_column, annotations, previous_layout=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode='lines'))

    # Apply the saved layout if available
    if previous_layout:
        fig.update_layout(previous_layout)

    for ann in annotations:
        fig.add_annotation(
            x=ann['x'],
            y=ann['y'],
            text=ann['label'],
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green" if ann['label'] == 'Buy' else "red",
            font=dict(size=10, color="green" if ann['label'] == 'Buy' else "red"),
            align="center",
            ax=0,
            ay=40 if ann['label'] == 'Buy' else -40
        )
    fig.update_layout(
        title='Stock Price Chart with Buy/Sell Annotations',
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    return fig



st.title('Interactive Stock Data Labeler with Line Chart')

# Initialize session state variables
if 'annotations' not in st.session_state:
    st.session_state.annotations = []
if 'label_mode' not in st.session_state:
    st.session_state.label_mode = 'None'
if 'df' not in st.session_state:
    st.session_state.df = None
if 'x_column' not in st.session_state:
    st.session_state.x_column = None
if 'y_column' not in st.session_state:
    st.session_state.y_column = None
if 'previous_y_column' not in st.session_state:
    st.session_state.previous_y_column = None

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    st.session_state.df = load_data(uploaded_file)
    st.dataframe(st.session_state.df, use_container_width=True, height=300)
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center; color: #ffeded'>Select Axis</h3>", unsafe_allow_html=True)
        if st.session_state.x_column is None:
            st.session_state.x_column = find_date_column(st.session_state.df)

        st.session_state.x_column = st.selectbox('Select X-axis', options=st.session_state.df.columns,
                                                 index=st.session_state.df.columns.get_loc(st.session_state.x_column))

        # Store the previous y_column before updating
        st.session_state.previous_y_column = st.session_state.y_column

        st.session_state.y_column = st.selectbox('Select Y-axis (Price)',
                                                 options=[col for col in st.session_state.df.columns if
                                                          col != st.session_state.x_column])

        # Check if y_column has changed and clear annotations if it has
        if st.session_state.y_column != st.session_state.previous_y_column and st.session_state.previous_y_column is not None:
            st.session_state.annotations = []
            st.warning("Y-axis changed. Annotations have been cleared.")

    with st.container(border=True):
        st.markdown("<h3 style='text-align: center; color: #ffeded'>Select Mode</h3>", unsafe_allow_html=True)
        # Mode selection
        col0, col1, col2, col3, col5 = st.columns([2, 2, 2, 2, 2])

        buy_button = col1.button('Buy Mode')
        sell_button = col2.button('Sell Mode')
        no_label_button = col3.button('No Label Mode')

        # Update label mode based on button press
        if buy_button:
            st.session_state.label_mode = 'Buy'
        elif sell_button:
            st.session_state.label_mode = 'Sell'
        elif no_label_button:
            st.session_state.label_mode = 'No Label Mode'

        # Display buttons with red text when selected
        buy_button_style = 'selected' if st.session_state.label_mode == 'Buy' else ''
        sell_button_style = 'selected' if st.session_state.label_mode == 'Sell' else ''
        no_label_button_style = 'selected' if st.session_state.label_mode == 'No Label Mode' else ''

        st.markdown(f"<p style='text-align: center; color: red !important;'> Mode: {st.session_state.label_mode}</p>",
                    unsafe_allow_html=True)

        # Generate and display chart
        previous_layout = st.session_state.get("layout", {})
        fig = create_line_chart(st.session_state.df, st.session_state.x_column, st.session_state.y_column,
                                st.session_state.annotations, previous_layout)

        clicked_points = plotly_events(fig, click_event=True, hover_event=False)

        if clicked_points and not st.session_state.label_mode in[ 'None', '','No Label Mode']:
            point = clicked_points[0]
            x, y = point['x'], point['y']
            st.session_state.annotations.append({'x': x, 'y': y, 'label': st.session_state.label_mode})
            st.session_state.layout = fig.layout
            print(fig.layout)
            st.rerun()

        col0, col1, col2, col4 = st.columns([2, 3, 3, 2])
        with col1:
            if st.button('Undo Last Annotation'):
                if st.session_state.annotations:
                    st.session_state.annotations.pop()
                    st.rerun()
        with col2:
            if st.button('Clear All Annotations'):
                st.session_state.annotations = []
                st.rerun()
    with st.container(border=True):
        col0, col1, col2 = st.columns([4, 2, 4])
        with col1:
            if st.button('Export Points'):
                buy_points = [ann for ann in st.session_state.annotations if ann['label'] == 'Buy']
                buy_df = pd.DataFrame(buy_points)
                buy_csv = buy_df.to_csv(index=False)
                st.download_button(
                    label="Download Buy points as CSV",
                    data=buy_csv,
                    file_name="buy.csv",
                    mime="text/csv",
                )

                sell_points = [ann for ann in st.session_state.annotations if ann['label'] == 'Sell']
                sell_df = pd.DataFrame(sell_points)
                sell_csv = sell_df.to_csv(index=False)
                st.download_button(
                    label="Download Sell points as CSV",
                    data=sell_csv,
                    file_name="sell.csv",
                    mime="text/csv",
                )
