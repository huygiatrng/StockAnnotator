# Label Trading Stock

A Streamlit application for interactive stock labeling and trading simulation.

## Features

- Interactive chart labeling for stock price action
- Automatic trade signal generation using technical indicators
- Trading simulation with customizable parameters
- Performance metrics and equity curve visualization
- GPU acceleration for large datasets (optional)

## Project Structure

```
label_trading_stock/
├── app/                      # Main application package
│   ├── core/                 # Core business logic
│   │   ├── data_loader.py    # Data loading utilities
│   │   ├── indicators.py     # Technical indicator calculations
│   │   ├── labeling.py       # Automatic labeling algorithms
│   │   └── simulation.py     # Trading simulation engine
│   ├── utils/                # Utility functions
│   │   └── gpu_helpers.py    # GPU acceleration utilities
│   ├── visualization/        # Visualization components
│   │   └── charts.py         # Chart creation functions
│   ├── ui/                   # User interface components
│   │   └── main_app.py       # Main Streamlit UI
│   ├── main.py               # Application entry point
│   └── __init__.py           # Package initialization
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run main.py
   ```

## Optional GPU Acceleration

For improved performance with large datasets, you can install GPU acceleration:

```
pip install cupy-cuda11x  # Replace with appropriate CUDA version
```

## Usage

1. Upload a CSV file containing stock price data
2. Select date and price columns
3. Calculate technical indicators (optional)
4. Label trading points manually by clicking on the chart or use auto-labeling
5. Run the trading simulation to evaluate performance

## License

MIT 