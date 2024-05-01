import pandas as pd
import numpy as np
from math import ceil, floor


class Describe:
    """
    Class to describe a dataset.

    Methods:
    - __init__(self, df: pd.DataFrame): Initializes the Describe object with a DataFrame.
    - min(self, column): Returns the minimum value in a column.
    - max(self, column): Returns the maximum value in a column.
    - count(self, column): Returns the count of non-null values in a column.
    - mean(self, column): Returns the mean value of a column.
    - std(self, column): Returns the standard deviation of a column.
    - unique(self, column): Returns the number of unique values in a column.
    - describe(self, percentiles=None): Generates descriptive statistics of the dataset.

    Attributes:
    - data: The DataFrame used for describing the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        self.data = df[df.select_dtypes(include=np.number).columns]

    def min(self, column):
        numeric_values = [x for x in column if pd.notnull(x)]
        return min(numeric_values) if numeric_values else np.nan

    def max(self, column):
        numeric_values = [x for x in column if pd.notnull(x)]
        return max(numeric_values) if numeric_values else np.nan

    def count(self, column):
        numeric_values = [x for x in column if pd.notnull(x)]
        return len(numeric_values)

    def mean(self, column):
        """
        Calculate the mean of a given column.

        Parameters:
        column (list): A list of numeric values.

        Returns:
        float: The mean value of the column. If the column is empty or contains only NaN values, returns NaN.
        """
        numeric_values = [x for x in column if pd.notnull(x)]
        return sum(numeric_values) / len(numeric_values) if numeric_values else np.nan
    
    def variance(self, column):
        """
        Calculate the variance of a given column.

        Parameters:
        column (list): A list of numeric values.

        Returns:
        float: The variance of the column.
        """
        numeric_values = [x for x in column if pd.notnull(x)]
        mean = self.mean(numeric_values)
        return sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)

    def ecart_type(self, column):
        """
        Calculate the standard deviation of a given column.

        Parameters:
        column (list): A list of numeric values.

        Returns:
        float: The standard deviation of the column.
        """
        V = self.variance(column)
        return np.sqrt(V)

    def _calculate_quartile(self, column, quartile: float):
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
            index = (len(numeric_values) - 1) * quartile
            if index.is_integer():
                return numeric_values[int(index)]
            else:
                lower = numeric_values[floor(index)] * (ceil(index) - index)
                upper = numeric_values[ceil(index)] * (index - floor(index))
                return (upper + lower)
        else:
            return np.nan

    """ def std(self, column):
        
        Calculate the standard deviation of a given column.

        Parameters:
        column (list): A list of numeric values.

        Returns:
        float: The standard deviation of the column.

        
        numeric_values = [x for x in column if np.isnan(x) == False]
        if numeric_values:
            mean = sum(numeric_values) / len(numeric_values)
            return (sum((x - mean) ** 2 for x in numeric_values) / (len(numeric_values) - 1)) ** 0.5
        else:
            return np.nan """
    
    def std(self, column):
        numeric_values = [x for x in column if x is not None and np.isnan(x) == False]
        mean = self.mean(numeric_values)
        return np.sqrt(sum((xi - mean) ** 2 for xi in numeric_values) / (len(numeric_values) - 1))
    
    def unique(self, column):
        """
        Calculate the number of unique values in a column.

        Parameters:
            column (list): The column to calculate unique values for.

        Returns:
            int: The number of unique values in the column.
        """
        numeric_values = [x for x in column if pd.notnull(x)]
        return len(set(numeric_values))

    def describe(self, percentiles=None):
        """
        Generates descriptive statistics of the dataset.

        Args:
        - percentiles (list): List of percentiles to include in the output. 
            Default is [0.25, 0.5, 0.75].

        Returns:
        - result (pd.DataFrame): DataFrame containing the descriptive statistics.
        """
        if percentiles is None:
            percentiles = [0.25, 0.5, 0.75]
        if percentiles:
            for q in percentiles:
                def percentile_func(column, q=q):
                    return self._calculate_quartile(column, q)
                percentile_func.__name__ = f'{int(q * 100)}%'
                setattr(self, percentile_func.__name__, percentile_func)
            percentiles = [getattr(self, f'{int(q * 100)}%') for q in percentiles]

        result = self.data.agg([
            self.count,
            self.mean,
            self.std,
            self.min,
            *percentiles,
            self.max,
            self.unique,
            self.variance,
            self.ecart_type
        ], axis=0)

        result = result.apply(lambda x: x.apply('{:.6f}'.format))
        print(result)
        return result


def get_file(file_path):
    fd = pd.read_csv(file_path)
    return fd


data = get_file('../datasets/dataset_train.csv')
describe = Describe(data)
describe.describe()
