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
        """
            Initializes the Describe object with a DataFrame.
        """
        self.data = df[df.select_dtypes(include=np.number).columns]

    def get_numeric_values(self, column):
        """
            Get the numeric values from a given column.
        """
        numeric_values = []
        for x in column:
            if pd.notnull(x):
                numeric_values.append(x)
        return numeric_values

    def sum_(self, column):
        """
            Calculate the sum of a given column.
        """
        numeric_values = self.get_numeric_values(column)
        sum_total = 0
        for value in numeric_values:
            if pd.notnull(value):
                sum_total += value
        return sum_total
    
    def len(self, column):
        """
            Calculate the length of a given column.
        """
        len_ = 0
        for _ in column:
            len_ += 1
        return len_


    def min(self, column):
        """
            Calculate the minimum value of a given column.
        """
        numeric_values = self.get_numeric_values(column)
        min_value = None
        if numeric_values:
            for value in numeric_values:
                if min_value is None or value < min_value:
                    min_value = value
            return min_value
        else:
            return np.nan
        
    def max(self, column):
        """
            Calculate the maximum value of a given column.
        """
        numeric_values = self.get_numeric_values(column)
        max_value = None
        if numeric_values:
            for value in numeric_values:
                if max_value is None or value > max_value:
                    max_value = value
            return max_value
        else:
            return np.nan

    def count(self, column):
        numeric_values = self.get_numeric_values(column)
        return self.len(numeric_values)

    def mean(self, column):
        """
            Calculate the mean of a given column.
        """
        numeric_values = self.get_numeric_values(column)
        if numeric_values:
            return self.sum_(numeric_values) / self.len(numeric_values) 
        else:
            np.nan
    
    def variance(self, column):
        """
            Calculate the variance of a given column.
        """
        numeric_values = self.get_numeric_values(column)
        mean = self.mean(numeric_values)
        itmes = 0
        for x in numeric_values:
            if pd.notnull(x):
                itmes += (x - mean) ** 2 
        return itmes / (self.len(numeric_values) - 1)

    def ecart_type(self, column):
        """
            Calculate the standard deviation of a given column.
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
            index = (self.len(numeric_values) - 1) * quartile
            if index.is_integer():
                return numeric_values[int(index)]
            else:
                lower = numeric_values[floor(index)] * (ceil(index) - index)
                upper = numeric_values[ceil(index)] * (index - floor(index))
                return (upper + lower)
        else:
            return np.nan
        
    def percentile_25(self, column):
        """
            Calculate the 25th percentile of a given column.
        """           
        return self._calculate_quartile(column, 0.25)
        
    def percentile_50(self, column):
        """
            Calculate the median of a given column.
        """
        return self._calculate_quartile(column, 0.5)

    def percentile_75(self, column):
        """
            Calculate the 75th percentile of a given column.
        """
        return self._calculate_quartile(column, 0.75)
    
    def std(self, column):
        """
            Calculate the standard deviation of a column.
        """
        numeric_values = self.get_numeric_values(column)
        mean = self.mean(numeric_values)
        sum_std = 0
        for x in numeric_values:
            if pd.notnull(x):
                sum_std += (x - mean) ** 2 
        std = np.sqrt(sum_std / (len(numeric_values) - 1))
        return std

    def unique(self, column):
        """
            Calculate the number of unique values in a column.
        """
        numeric_values = [x for x in column if pd.notnull(x)]
        return len(set(numeric_values))

    def describe(self):
        """
            Calculate descriptive statistics for the dataset..
        """
        result = self.data.agg([
            self.count,
            self.mean,
            self.std,
            self.min,
            self.percentile_25,
            self.percentile_50,
            self.percentile_75,  
            self.max,
            self.unique,
            self.variance,    
            self.ecart_type
        ], axis=0)

        result.index = [
            'count', 'mean', 'std', 'min',
            '25%', '50%', '75%', 'max', 'unique',
            'variance', 'ecart_type'
        ]
        result = result.applymap("{:.6f}".format)
        print(result)
        return result
