"""
Main entry point for the Label Trading Stock application.
This script ensures proper module imports by adding the parent directory to the path.
"""

import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import and run the app
from app.ui.main_app import run_app

if __name__ == "__main__":
    run_app() 