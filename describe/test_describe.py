import unittest
import pandas as pd
import numpy as np
from math import sqrt
from describe import Describe  # Assuming `describe.py` contains the Describe class implementation


class TestDescribe(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.5, 2.5, 3.5, 4.5, 5.5],
            'C': [1, 1, 1, 1, -1],
            'D': [1, 2, 3, 4, 5],
            'E': [1, 2, 3, 4, 5],
            'Variance': [7, 8, 10, 11, 13]
        })
        self.describe = Describe(self.data)


    def test_sum(self):
        self.assertEqual(self.describe.sum_(self.data['A']), 15)
        self.assertEqual(self.describe.sum_(self.data['B']), 17.5)
        self.assertEqual(self.describe.sum_(self.data['C']), 3)
        self.assertEqual(self.describe.sum_(self.data['D']), 15)
        self.assertEqual(self.describe.sum_(self.data['E']), 15)

    def test_len(self):
        self.assertEqual(self.describe.len(self.data['A']), 5)
        self.assertEqual(self.describe.len(self.data['B']), 5)
        self.assertEqual(self.describe.len(self.data['C']), 5)
        self.assertEqual(self.describe.len(self.data['D']), 5)
        self.assertEqual(self.describe.len(self.data['E']), 5)

    def test_min(self):
        self.assertEqual(self.describe.min(self.data['A']), 1)
        self.assertEqual(self.describe.min(self.data['B']), 1.5)
        self.assertEqual(self.describe.min(self.data['C']), -1)
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
        self.assertEqual(self.describe.mean(self.data['C']), 0.6)
        self.assertEqual(self.describe.mean(self.data['D']), 3)
        self.assertEqual(self.describe.mean(self.data['E']), 3)

    def test_variance(self):
        self.assertEqual(self.describe.variance(self.data['Variance']), 5.699999999999999)
        self.assertEqual(self.describe.variance(self.data['A']), 2.5)
        self.assertEqual(self.describe.variance(self.data['B']), 2.5)
        self.assertEqual(self.describe.variance(self.data['C']), 0.8000000000000002 )
        self.assertEqual(self.describe.variance(self.data['D']), 2.5)
        self.assertEqual(self.describe.variance(self.data['E']), 2.5)

    def test_ecart_type(self):
        self.assertEqual(self.describe.ecart_type(self.data['A']), sqrt(2.5))
        self.assertEqual(self.describe.ecart_type(self.data['B']), sqrt(2.5))
        self.assertEqual(self.describe.ecart_type(self.data['C']),  sqrt(0.8000000000000002))
        self.assertEqual(self.describe.ecart_type(self.data['D']), sqrt(2.5))
        self.assertEqual(self.describe.ecart_type(self.data['E']),  sqrt(2.5))
        self.assertEqual(self.describe.ecart_type(self.data['Variance']), sqrt(5.699999999999999))

    def test_std(self):
        self.assertAlmostEqual(self.describe.std(self.data['A']), 1.581139, places=6)
        self.assertAlmostEqual(self.describe.std(self.data['B']), 1.581139, places=6)
        self.assertEqual(self.describe.std(self.data['C']), 0.894427190999916)
        self.assertAlmostEqual(self.describe.std(self.data['D']), 1.581139, places=6)
        self.assertAlmostEqual(self.describe.std(self.data['E']), 1.581139, places=6)

    def test_unique(self):
        self.assertEqual(self.describe.unique(self.data['A']), 5)
        self.assertEqual(self.describe.unique(self.data['B']), 5)
        self.assertEqual(self.describe.unique(self.data['C']), 2)
        self.assertEqual(self.describe.unique(self.data['D']), 5)
        self.assertEqual(self.describe.unique(self.data['E']), 5)


if __name__ == '__main__':
    unittest.main()

"""  
        To run the test, run the command below:
            python -m unittest -v test_describe.py
        The output should look like this:
            test_count (__main__.TestDescribe) ... ok
            test_max (__main__.TestDescribe) ... ok
            test_mean (__main__.TestDescribe) ... ok
            test_min (__main__.TestDescribe) ... ok
            test_std (__main__.TestDescribe) ... ok
            test_unique (__main__.TestDescribe) ... ok
            ----------------------------------------------------------------------
            Ran 6 tests in 0.002s
            OK
"""
