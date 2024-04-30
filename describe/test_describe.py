import unittest
import pandas as pd
import numpy as np
from describe import Describe


class TestDescribe(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.5, 2.5, 3.5, 4.5, 5.5],
            'C': [1, 1, 1, 1, 1],
            'D': [1, 2, 3, 4, 5],
            'E': [1, 2, 3, 4, 5]
        })
        self.describe = Describe(self.data)

    def test_min(self):
        self.assertEqual(self.describe.min(self.data['A']), 1)
        self.assertEqual(self.describe.min(self.data['B']), 1.5)
        self.assertEqual(self.describe.min(self.data['C']), 1)
        self.assertEqual(self.describe.min(self.data['D']), 1)
        self.assertEqual(self.describe.min(self.data['E']), 1)

    def test_max(self):
        self.assertEqual(self.describe.max(self.data['A']), 5)
        self.assertEqual(self.describe.max(self.data['B']), 5.5)
        self.assertEqual(self.describe.max(self.data['C']), 1)
        self.assertEqual(self.describe.max(self.data['D']), 5)
        self.assertEqual(self.describe.max(self.data['E']), 5)

    def test_count(self):
        self.assertEqual(self.describe.count(self.data['A']), 5)
        self.assertEqual(self.describe.count(self.data['B']), 5)
        self.assertEqual(self.describe.count(self.data['C']), 5)
        self.assertEqual(self.describe.count(self.data['D']), 5)
        self.assertEqual(self.describe.count(self.data['E']), 5)

    def test_mean(self):
        self.assertEqual(self.describe.mean(self.data['A']), 3)
        self.assertEqual(self.describe.mean(self.data['B']), 3.5)
        self.assertEqual(self.describe.mean(self.data['C']), 1)
        self.assertEqual(self.describe.mean(self.data['D']), 3)
        self.assertEqual(self.describe.mean(self.data['E']), 3)

    def test_std(self):
        self.assertAlmostEqual(self.describe.std(self.data['A']), 1.581139, places=6)
        self.assertAlmostEqual(self.describe.std(self.data['B']), 1.581139, places=6)
        self.assertEqual(self.describe.std(self.data['C']), 0)
        self.assertAlmostEqual(self.describe.std(self.data['D']), 1.581139, places=6)
        self.assertAlmostEqual(self.describe.std(self.data['E']), 1.581139, places=6)

    def test_unique(self):
        self.assertEqual(self.describe.unique(self.data['A']), 5)
        self.assertEqual(self.describe.unique(self.data['B']), 5)
        self.assertEqual(self.describe.unique(self.data['C']), 1)
        self.assertEqual(self.describe.unique(self.data['D']), 5)
        self.assertEqual(self.describe.unique(self.data['E']), 5)

    def test_describe(self):
        expected_result = pd.DataFrame({
            'A': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5],
            'B': [5, 3.500000, 1.581139, 1.5, 2.5, 3.5, 4.5, 5.5, 5],
            'C': [5, 1.000000, 0.000000, 1, 1, 1, 1, 1, 1],
            'D': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5],
            'E': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5]
        }, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'unique'])

        result = self.describe.describe()
        result = result.astype(float)  # Convert all columns to float64

        pd.testing.assert_frame_equal(result, expected_result)

    def test_describe_with_none_values(self):
        self.describe.data = pd.DataFrame({
            'A': [1, 2, 3, None, 5],
            'B': [1.5, 2.5, 3.5, 4.5, None],
            'C': [None, None, None, None, None],
            'D': [1, 2, 3, 4, 5],
            'E': [1, 2, 3, 4, 5]
        })

    def test_describe_with_large_dataset(self):
        np.random.seed(0)  # pour la reproductibilité
        data = pd.DataFrame({
            'A': np.random.randint(0, 100, size=10000),
            'B': np.random.normal(0, 1, size=10000),
            'C': np.random.choice([1, 2, 3, None], size=10000),
            'D': np.random.uniform(0, 1, size=10000),
            'E': np.random.lognormal(0, 1, size=10000)
        })

        self.describe.data = data
        result = self.describe.describe()
        self.assertEqual(result.shape, (9, 5))


if __name__ == '__main__':
    unittest.main()
