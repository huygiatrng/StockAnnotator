"""
Core functions for loading and processing data
"""

import pandas as pd
import numpy as np
import io
import os
import time
import traceback
from datetime import datetime


def load_file(file_obj):
    """
    Load a file object into a pandas DataFrame with improved error handling
    
    Parameters:
    -----------
    file_obj : file object
        File object from st.file_uploader
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    # Sanity check on file object
    if file_obj is None:
        raise ValueError("No file provided")
    
    # Check if file is empty
    if file_obj.size == 0:
        raise ValueError("File is empty")
        
    # Get file extension and show file details for debugging
    file_name = file_obj.name
    file_ext = os.path.splitext(file_name)[1].lower()
    print(f"Loading file: {file_name} (Size: {file_obj.size/1024:.1f} KB, Type: {file_ext})")
    
    # Create a buffer copy to avoid file access issues
    try:
        # Read the file content into memory first to avoid streaming issues
        file_content = file_obj.read()
        file_obj.seek(0)  # Reset position for subsequent reads
        
        # Use BytesIO to create a file-like object from the content
        buffer = io.BytesIO(file_content)
        print(f"Successfully created buffer from file, size: {len(file_content)} bytes")
    except Exception as e:
        print(f"Error creating buffer from file: {str(e)}")
        traceback.print_exc()
        raise IOError(f"Failed to read file content: {str(e)}")
    
    # Print some debug info about the file content for CSV files
    if file_ext == '.csv':
        try:
            preview = file_content[:200].decode('utf-8')
            print(f"File preview: {preview}")
        except:
            print("Could not decode file preview")
    
    # Load based on file type with better error handling
    try:
        if file_ext == '.csv':
            print("Attempting to load CSV file...")
            # First try with default settings
            try:
                df = pd.read_csv(buffer)
                print(f"Successfully loaded CSV with default settings. Shape: {df.shape}")
            except UnicodeDecodeError as decode_error:
                print(f"Unicode decode error, trying latin1 encoding: {str(decode_error)}")
                # Try with different encoding
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding='latin1')
                print(f"Successfully loaded CSV with latin1 encoding. Shape: {df.shape}")
            except pd.errors.ParserError as parser_error:
                print(f"Parser error, trying different delimiter: {str(parser_error)}")
                # Try with different delimiter
                buffer.seek(0)
                try:
                    df = pd.read_csv(buffer, sep=';')
                    print(f"Successfully loaded CSV with semicolon delimiter. Shape: {df.shape}")
                except Exception as sep_error:
                    print(f"Failed with semicolon delimiter, trying python engine: {str(sep_error)}")
                    buffer.seek(0)
                    # Try with more flexible options as last resort
                    df = pd.read_csv(buffer, sep=None, engine='python')
                    print(f"Successfully loaded CSV with python engine. Shape: {df.shape}")
        elif file_ext in ['.xlsx', '.xls']:
            print("Attempting to load Excel file...")
            try:
                df = pd.read_excel(buffer)
                print(f"Successfully loaded Excel file. Shape: {df.shape}")
            except Exception as excel_err:
                print(f"Excel read error: {str(excel_err)}")
                traceback.print_exc()
                raise ValueError(f"Failed to read Excel file: {str(excel_err)}")
        else:
            raise ValueError(f"Unsupported file type: {file_ext}. Please use CSV or Excel files.")
    except Exception as e:
        print(f"Error loading file {file_name}: {str(e)}")
        traceback.print_exc()
        raise e
    
    # Verify the dataframe was loaded correctly
    if df is None or df.empty:
        print("Loaded dataframe is empty")
        raise ValueError("No data could be loaded from the file")
    
    print(f"Successfully loaded dataframe with shape: {df.shape}")
    
    # Show column information for debugging
    print(f"Columns: {df.columns.tolist()}")
    
    # Basic data cleaning
    df = clean_dataframe(df)
    
    return df


def clean_dataframe(df):
    """
    Perform basic data cleaning operations
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    # Remove completely empty columns and rows
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    
    # Convert date columns to datetime
    date_col = detect_date_column(df)
    if date_col is not None:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=[date_col])
        except Exception as e:
            print(f"Error converting {date_col} to datetime: {str(e)}")
            pass  # If conversion fails, leave as is
    
    # Sort by date if available
    if date_col is not None and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.sort_values(by=date_col)
    
    return df


def detect_date_column(df):
    """
    Attempt to automatically detect which column contains dates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    str or None
        Name of date column if found, None otherwise
    """
    # Common date column names
    date_names = ['date', 'time', 'datetime', 'Date', 'Time', 'DateTime', 'timestamp', 'Timestamp']
    
    # First check column names
    for col in df.columns:
        if col in date_names or any(date_name.lower() in col.lower() for date_name in date_names):
            return col
    
    # Then check if any column is datetime type
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Finally check if any string column looks like dates
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check first few non-null values
            sample = df[col].dropna().head(5)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample, errors='raise')
                    return col
                except:
                    pass
    
    # If no date column found, return first column as fallback
    return df.columns[0] if len(df.columns) > 0 else None 