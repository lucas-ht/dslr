import unittest
from math import isclose
import numpy as np
import pandas as pd
from dslr.math import (
    get_numeric_values,
    ft_sum,
    ft_len,
    ft_min,
    ft_max,
    ft_count,
    ft_mean,
    ft_variance,
    calculate_quartile,
    percentile_25,
    percentile_50,
    percentile_75,
    ft_std,
    ft_unique,
    ft_gaps
)
from describe import describe


class TestMathFunctions(unittest.TestCase):
    """ Test the math functions."""
    def setUp(self):
        """ Set up the test data."""
        self.column = pd.Series([1, 2, 3, 4, 5, np.nan])

    def test_describe(self):
        """ Test the describe function."""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.5, 2.5, 3.5, 4.5, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        expected_result = pd.DataFrame({
            'A': [5, 3.0, 1.581139, 1, 2, 3, 4, 5, 5, 2.5, 1.581139],
            'B': [5, 3.5, 1.581139, 1.5, 2.5, 3.5, 4.5, 5.5, 5, 2.5, 1.581139]
        }, index=[
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
            'grp'
        ]
        )

        result = describe(df)
        result['A'] = pd.to_numeric(result['A'], errors='coerce')
        result['B'] = pd.to_numeric(result['B'], errors='coerce')
        pd.testing.assert_frame_equal(result, expected_result)

    def test_get_numeric_values(self):
        """ Test the get_numeric_values function."""
        result = get_numeric_values(self.column)
        expected_result = [1, 2, 3, 4, 5]
        self.assertEqual(result, expected_result)

    def test_ft_sum(self):
        """ Test the ft_sum function."""
        result = ft_sum(self.column)
        expected_result = 15
        self.assertEqual(result, expected_result)

    def test_ft_len(self):
        """ Test the ft_len function."""
        result = ft_len(self.column)
        expected_result = 6
        self.assertEqual(result, expected_result)

    def test_ft_min(self):
        """ Test the ft_min function."""
        result = ft_min(self.column)
        expected_result = 1
        self.assertEqual(result, expected_result)

    def test_ft_max(self):
        """ Test the ft_max function."""
        result = ft_max(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)

    def test_ft_count(self):
        """ Test the ft_count function."""
        result = ft_count(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)

    def test_ft_mean(self):
        """ Test the ft_mean function."""
        result = ft_mean(self.column)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_ft_variance(self):
        """ Test the ft_variance function."""
        result = ft_variance(self.column)
        expected_result = 2.5
        self.assertEqual(result, expected_result)

    def test_FteCart_type(self):
        """ Test the ft_gaps function."""
        result = ft_gaps(self.column)
        expected_result = 1.5811388300841898
        self.assertTrue(isclose(result, expected_result))

    def test_calculate_quartile(self):
        """ Test the calculate_quartile function."""
        result = calculate_quartile(self.column, 0.25)
        expected_result = 2
        self.assertEqual(result, expected_result)

    def test_percentile_25(self):
        """ Test the percentile_25 function."""
        result = percentile_25(self.column)
        expected_result = 2
        self.assertEqual(result, expected_result)

    def test_percentile_50(self):
        """ Test the percentile_50 function."""
        result = percentile_50(self.column)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_percentile_75(self):
        """ Test the percentile_75 function."""
        result = percentile_75(self.column)
        expected_result = 4
        self.assertEqual(result, expected_result)

    def test_ft_std(self):
        """ Test the ft_std function. """
        result = ft_std(self.column)
        expected_result = 1.5811388300841898
        self.assertTrue(isclose(result, expected_result))

    def test_ft_unique(self):
        """ Test the ft_unique function. """
        result = ft_unique(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
