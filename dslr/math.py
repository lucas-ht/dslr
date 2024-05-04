import pandas as pd
import numpy as np
from math import ceil, floor

def get_numeric_values(column: pd.Series):
    """
        Get the numeric values from a given column.
    """
    numeric_values = []
    for x in column:
        if pd.notnull(x):
            numeric_values.append(x)
    return numeric_values

def ft_sum(column):
    """
        Calculate the sum of a given column.
    """
    numeric_values = get_numeric_values(column)
    sum_total = 0
    for value in numeric_values:
        if pd.notnull(value):
            sum_total += value 
    return sum_total

def ft_len(column):
    """
        Calculate the length of a given column.
    """
    length = 0
    for _ in column:
        length += 1
    return length

def ft_min(column: pd.Series):
    """
        Calculate the minimum value of a given column.
    """
    numeric_values = get_numeric_values(column)
    min_value = None
    if numeric_values:
        for value in numeric_values:
            if min_value is None or value < min_value:
                min_value = value
        return min_value
    else:
        return np.nan
    
def ft_max(column):
    """
        Calculate the maximum value of a given column.
    """
    numeric_values = get_numeric_values(column)
    max_value = None
    if numeric_values:
        for value in numeric_values:
            if max_value is None or value > max_value:
                max_value = value
        return max_value
    else:
        return np.nan
    
def ft_count(column):
    numeric_values = get_numeric_values(column)
    return ft_len(numeric_values)

def ft_mean(column):
    """
        Calculate the mean of a given column.
    """
    numeric_values = get_numeric_values(column)
    if numeric_values:
        return ft_sum(numeric_values) / ft_len(numeric_values) 
    else:
        np.nan

def ft_variance(column):
    """
        Calculate the variance of a given column.
    """
    numeric_values = get_numeric_values(column)
    mean = ft_mean(numeric_values)
    items = 0
    for x in numeric_values:
        if pd.notnull(x):
            items += (x - mean) ** 2 
    return items / (ft_len(numeric_values) - 1)

def ft_ecart_type(column):
    """
        Calculate the standard deviation of a given column.
    """
    V = ft_variance(column)
    return np.sqrt(V)

def calculate_quartile(column, quartile: float):
    """
    Calculate the quartile value for a given column.
    Args:
        column (pandas.Series): The column containing numeric values.
        quartile (float): The quartile value to calculate (0.25 for first quartile, 0.5 for median, 0.75 for third quartile).
    Returns:
        float: The calculated quartile value.
    """
    numeric_values = sorted(x for x in column if pd.notnull(x))
    if numeric_values:
        index = (ft_len(numeric_values) - 1) * quartile
        if index.is_integer():
            return numeric_values[int(index)]
        else:
            lower = numeric_values[floor(index)] * (ceil(index) - index)
            upper = numeric_values[ceil(index)] * (index - floor(index))
            return (upper + lower)
    else:
        return np.nan
    
def percentile_25(column):
    """
        Calculate the 25th percentile of a given column.
    """           
    return calculate_quartile(column, 0.25)
    
def percentile_50(column):
    """
        Calculate the median of a given column.
    """
    return calculate_quartile(column, 0.5)

def percentile_75(column):
    """
        Calculate the 75th percentile of a given column.
    """
    return calculate_quartile(column, 0.75)

def ft_std(column):
    """
        Calculate the standard deviation of a column.
    """
    numeric_values = get_numeric_values(column)
    mean = ft_mean(numeric_values)
    sum_std = 0
    for x in numeric_values:
        if pd.notnull(x):
            sum_std += (x - mean) ** 2 
    std = np.sqrt(sum_std / (ft_len(numeric_values) - 1))
    return std

def ft_unique(column):
    """
        Calculate the number of unique values in a column.
    """
    numeric_values = get_numeric_values(column)
    return len(set(numeric_values))
