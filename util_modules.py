import pandas as pd
import numpy as np

def normalize(pandas_df, norm_type):
    """Normalize dataframe values.

    Args:
        pandas_df: Pandas dataframe.
        norm_type: 'full' for normalization using all the value in the dataframe,
                   'freq' for normalization along each column.

    Returns:
        Pandas dataframe with normalized values.
    """

    if norm_type == 'full': # across the entire dataset
        min_val = pandas_df.min().min()
        max_val = pandas_df.max().max()
        
        return (pandas_df - min_val) / (max_val - min_val)

    elif norm_type == 'freq': # across each frequency (feature)
        min_vals = pandas_df.min()
        max_vals = pandas_df.max()
        diff = max_vals - min_vals

        return pandas_df.sub(min_vals, axis=1).div(diff, axis=1)

    else:
        raise Exception('Unknown normalization type string.')

def standardize(pandas_df):
    """Standardize dataframe values using Z-score standardization.

    Args:
        pandas_df: Pandas dataframe.

    Returns:
        Pandas dataframe with standardized values.
    """
    return pandas_df.sub(pandas_df.mean(0), axis=1) / pandas_df.std(0)