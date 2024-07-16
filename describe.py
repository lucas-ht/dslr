"""
This module contains the describe implementation.
"""

import pandas as pd
import numpy as np

from dslr.parser import Parser
from dslr.math import (
    ft_len,
    ft_min,
    ft_max,
    ft_mean,
    ft_variance,
    ft_percentile,
    ft_std,
    ft_unique
)


def format_df(df: pd.Series) -> pd.Series:
    """
    Format the DataFrame.

    Args:
        df: The DataFrame to format.

    Returns:
        The formatted DataFrame.
    """

    return df.map(lambda x: f"{x:.6f}")


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the dataset.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The descriptive statistics.
    """

    df = df.select_dtypes(include=np.number)
    if df.empty:
        return pd.DataFrame()

    df = df.agg([
        ft_len,
        ft_mean,
        ft_std,
        ft_min,
        lambda x: ft_percentile(x, .25),
        lambda x: ft_percentile(x, .50),
        lambda x: ft_percentile(x, .75),
        ft_max,
        ft_unique,
        ft_variance,
    ], axis=0)

    df.index = pd.Index([
        'count',
        'mean',
        'std',
        'min',
        '25%',
        '50%',
        '75%',
        'max',
        'unique',
        'variance',
    ])

    return df.apply(format_df)


def main() -> None:
    """
    The main function of the describe module.
    """

    df = Parser().read_dataset()

    print('Our Describe')
    print(f'{describe(df)}\n')

    print('Pandas\' Describe')
    print(df.describe())


if __name__ == '__main__':
    main()
