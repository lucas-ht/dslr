"""
This module contains the describe implementation.
"""

import pandas as pd
import numpy as np
from dslr.parser import Parser
from dslr.math import (
    ft_min,
    ft_max,
    ft_count,
    ft_mean,
    ft_variance,
    ft_gaps,
    percentile_25,
    percentile_50,
    percentile_75,
    ft_std,
    ft_unique
)


def format_series(series):
    return series.map("{:.6f}".format)


def describe(df: pd.DataFrame):
    """
        Calculate descriptive statistics for the dataset..
    """
    data = df[df.select_dtypes(include=np.number).columns]

    result = data.agg([
        ft_count,
        ft_mean,
        ft_std,
        ft_min,
        percentile_25,
        percentile_50,
        percentile_75,
        ft_max,
        ft_unique,
        ft_variance,
        ft_gaps
    ], axis=0)

    result.index = [
        'count', 'mean', 'std', 'min',
        '25%', '50%', '75%', 'max', 'unique',
        'variance', 'grp'
    ]
    result = result.apply(format_series)
    print(result)
    return result


def main():
    """
    The main function of the describe module.
    """

    data = Parser().read_dataset()
    print('Describe Function')
    describe(data)
    print()
    print('Describe from Pandas')
    print(data.describe())


if __name__ == '__main__':
    main()
