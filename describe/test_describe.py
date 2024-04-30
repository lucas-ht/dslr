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
