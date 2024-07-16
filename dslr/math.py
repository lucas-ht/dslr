"""
This module contains functions to calculate statistics of a given column.
"""
from typing import Callable, Iterable, Any
from math import ceil, floor, inf, sqrt
from functools import wraps

import pandas as pd


def dropna(func: Callable) -> Callable:
    """
    Decorator to drop NaN values from a column before applying a function.

    Args:
        func (Callable): The function to apply.
        *args: The positional arguments to pass to the function.
        **kwargs: The keyword arguments to pass to the function.

    Returns:
        Callable: The wrapper function.
    """

    @wraps(func)
    def wrapper(column: pd.Series, *args, **kwargs) -> Any:
        return func(column.dropna(), *args, **kwargs)

    return wrapper


@dropna
def ft_sum(column: pd.Series) -> float:
    """
    Calculate the sum of a given column.

    Args:
        column (pd.Series): The column to calculate the sum of.

    Returns:
        float: The sum of the column.
    """

    total = 0
    for value in column:
        total += value
    return total


def ft_len(column: pd.Series | Iterable) -> int:
    """
    Calculate the length of a given column.

    Args:
        column (pd.Series | Iterable): The column to calculate the length of.

    Returns:
        int: The length of the column.
    """

    length = 0
    for _ in column:
        length += 1
    return length


@dropna
def ft_min(column: pd.Series) -> float:
    """
    Calculate the minimum value of a given column.

    Args:
        column (pd.Series): The column to calculate the minimum value of.

    Returns:
        float: The minimum value of the column.
    """

    min_value = inf
    for value in column:
        min_value = value if value < min_value else min_value
    return min_value


@dropna
def ft_max(column: pd.Series) -> float:
    """
    Calculate the maximum value of a given column.

    Args:
        column (pd.Series): The column to calculate the maximum value of.

    Returns:
        float: The maximum value of the column.
    """

    max_value = -inf
    for value in column:
        max_value = value if value > max_value else max_value
    return max_value


@dropna
def ft_mean(column: pd.Series) -> float:
    """
    Calculate the mean of a given column.

    Args:
        column (pd.Series): The column to calculate the mean of.

    Returns:
        float: The mean of the column.
    """

    return ft_sum(column) / ft_len(column)


@dropna
def ft_variance(column: pd.Series) -> float:
    """
    Calculate the variance of a given column.

    Args:
        column (pd.Series): The column to calculate the variance of.

    Returns:
        float: The variance of the column
    """

    mean = ft_mean(column)
    total = 0

    for value in column:
        total += (value - mean) ** 2

    return total / (ft_len(column) - 1)


@dropna
def ft_std(column: pd.Series) -> float:
    """
    Calculate the standard deviation of a column.

    Args:
        column (pd.Series): The column to calculate the standard deviation of.

    Returns:
        float: The standard deviation of the column.
    """

    variance = ft_variance(column)
    return sqrt(variance)


@dropna
def ft_percentile(column: pd.Series, percentile: float) -> float:
    """
    Calculate the percentile of a given column.

    Args:
        column (pd.Series): The column to calculate the percentile of.
        percentile (float): The percentile to calculate.

    Returns:
        float: The percentile of the column.
    """

    values = column.sort_values().reset_index(drop=True)
    index = (ft_len(values) - 1) * percentile

    if index.is_integer():
        return values[int(index)]

    lower = values[floor(index)] * (ceil(index) - index)
    upper = values[ceil(index)] * (index - floor(index))

    return upper + lower


def ft_unique(column: pd.Series) -> int:
    """
    Calculate the number of unique values in a column.

    Args:
        column (pd.Series): The column to calculate the number of unique values of.

    Returns:
        int: The number of unique values in the column.
    """

    return ft_len(set(column))
