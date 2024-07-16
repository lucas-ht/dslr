"""
Test the math functions.
"""

import unittest

import numpy as np
import pandas as pd

from dslr.math import (
    ft_sum,
    ft_len,
    ft_min,
    ft_max,
    ft_mean,
    ft_variance,
    ft_std,
    ft_percentile,
    ft_unique,
)


class TestMath(unittest.TestCase):
    """
    Test the math functions.
    """

    def setUp(self):
        """
        Set up the test data.
        """

        self.column = pd.Series([1, 2, 3, 4, 5, np.nan])


    def test_ft_sum(self):
        """
        Test the ft_sum function.
        """

        result = ft_sum(self.column)
        self.assertEqual(result, 15)


    def test_ft_len(self):
        """
        Test the ft_len function.
        """

        result = ft_len(self.column)
        self.assertEqual(result, 6)


    def test_ft_min(self):
        """
        Test the ft_min function.
        """

        result = ft_min(self.column)
        self.assertEqual(result, 1)


    def test_ft_max(self):
        """
        Test the ft_max function.
        """

        result = ft_max(self.column)
        self.assertEqual(result, 5)


    def test_ft_mean(self):
        """
        Test the ft_mean function.
        """

        result = ft_mean(self.column)
        self.assertEqual(result, 3)


    def test_ft_variance(self):
        """
        Test the ft_variance function.
        """

        result = ft_variance(self.column)
        self.assertEqual(result, 2.5)


    def test_ft_percentile(self):
        """
        Test the ft_percentile function.
        """

        result = ft_percentile(self.column, 0.25)
        self.assertEqual(result, 2)

        result = ft_percentile(self.column, 0.50)
        self.assertEqual(result, 3)

        result = ft_percentile(self.column, 0.75)
        self.assertEqual(result, 4)


    def test_ft_std(self):
        """
        Test the ft_std function.
        """

        result = ft_std(self.column)
        expected_result = 1.581138

        self.assertAlmostEqual(result, expected_result, places=5)


    def test_ft_unique(self):
        """
        Test the ft_unique function.
        """

        result = ft_unique(self.column)
        self.assertEqual(result, 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
