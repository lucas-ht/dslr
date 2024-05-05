import unittest
import pandas as pd
import numpy as np
from math import isclose
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
    def setUp(self):
        self.column = pd.Series([1, 2, 3, 4, 5, np.nan])

    def test_describe(self):
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
            'ecart_type'
        ]
        )

        result = describe(df)
        result['A'] = result['A'].astype(float)
        result['B'] = result['B'].astype(float)
        pd.testing.assert_frame_equal(result, expected_result)

    def test_describe_with_large_data(self):
        large_df = pd.DataFrame({
            'A': range(1, 10001),
            'B': np.random.rand(10000)  
        })

        result = describe(large_df)

        result = result.apply(pd.to_numeric, errors='coerce')

        expected_count_A = 10000
        expected_mean_A = 5000.5
        expected_std_A = 2886.895680
        expected_min_A = 1
        expected_25_A = 2500.75
        expected_50_A = 5000.5
        expected_75_A = 7500.25
        expected_max_A = 10000
        expected_unique_A = 10000
        expected_variance_A = 8334166.666667
        expected_ecart_type_A = 2886.895680

        self.assertAlmostEqual(result.loc['count', 'A'], expected_count_A)
        self.assertAlmostEqual(result.loc['mean', 'A'], expected_mean_A)
        self.assertAlmostEqual(result.loc['std', 'A'], expected_std_A)
        self.assertAlmostEqual(result.loc['min', 'A'], expected_min_A)
        self.assertAlmostEqual(result.loc['25%', 'A'], expected_25_A)
        self.assertAlmostEqual(result.loc['50%', 'A'], expected_50_A)
        self.assertAlmostEqual(result.loc['75%', 'A'], expected_75_A)
        self.assertAlmostEqual(result.loc['max', 'A'], expected_max_A)
        self.assertAlmostEqual(result.loc['unique', 'A'], expected_unique_A)
        self.assertAlmostEqual
        (result.loc['variance', 'A'], expected_variance_A)
        self.assertAlmostEqual
        (result.loc['ecart_type', 'A'], expected_ecart_type_A)

    def test_get_numeric_values(self):
        result = get_numeric_values(self.column)
        expected_result = [1, 2, 3, 4, 5]
        self.assertEqual(result, expected_result)

    def test_ft_sum(self):
        result = ft_sum(self.column)
        expected_result = 15
        self.assertEqual(result, expected_result)

    def test_ft_len(self):
        result = ft_len(self.column)
        expected_result = 6
        self.assertEqual(result, expected_result)

    def test_ft_min(self):
        result = ft_min(self.column)
        expected_result = 1
        self.assertEqual(result, expected_result)

    def test_ft_max(self):
        result = ft_max(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)

    def test_ft_count(self):
        result = ft_count(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)

    def test_ft_mean(self):
        result = ft_mean(self.column)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_ft_variance(self):
        result = ft_variance(self.column)
        expected_result = 2.5
        self.assertEqual(result, expected_result)

    def test_FteCart_type(self):
        result = ft_gaps(self.column)
        expected_result = 1.5811388300841898
        self.assertTrue(isclose(result, expected_result))

    def test_calculate_quartile(self):
        result = calculate_quartile(self.column, 0.25)
        expected_result = 2
        self.assertEqual(result, expected_result)

    def test_percentile_25(self):
        result = percentile_25(self.column)
        expected_result = 2
        self.assertEqual(result, expected_result)

    def test_percentile_50(self):
        result = percentile_50(self.column)
        expected_result = 3
        self.assertEqual(result, expected_result)

    def test_percentile_75(self):
        result = percentile_75(self.column)
        expected_result = 4
        self.assertEqual(result, expected_result)

    def test_ft_std(self):
        result = ft_std(self.column)
        expected_result = 1.5811388300841898
        self.assertTrue(isclose(result, expected_result))

    def test_ft_unique(self):
        result = ft_unique(self.column)
        expected_result = 5
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main(verbosity=2)