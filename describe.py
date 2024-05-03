import pandas as pd
import numpy as np
from dslr.parser import Parser
from dslr.math import *

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
        ft_ecart_type
    ], axis=0)

    result.index = [
        'count', 'mean', 'std', 'min',
        '25%', '50%', '75%', 'max', 'unique',
        'variance', 'ecart_type'
    ]
    result = result.apply(format_series)
    print(result)
    return result

def main():
    parser = Parser()
    data = parser.read()
    print(f'Describe Function')
    describe(data)
    print()
    print(f'Describe from Pandas')
    print(data.head())
    print(data.describe())

if __name__ == '__main__':
    main()
