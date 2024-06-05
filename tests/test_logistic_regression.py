"""
This module contains the tests for the LogisticRegression class.
"""

import unittest
import numpy as np

from dslr.model.logistic_regression import LogisticRegression

class TestLogisticRegression(unittest.TestCase):
    """
    This class contains the tests for the LogisticRegression class.
    """

    def setUp(self):
        self.log_reg = LogisticRegression(learning_rate=0.01, epochs=1000)


    def test_sigmoid(self):
        """
        Test the sigmoid function.
        """

        # pylint: disable=protected-access
        self.assertEqual(self.log_reg._sigmoid(0), 0.5)


    def test_fit(self):
        """
        Test the fit method.
        """

        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.log_reg.fit(x, y)

        self.assertEqual(self.log_reg.m, 3)
        self.assertEqual(self.log_reg.n, 2)


    def test_predict(self):
        """
        Test the predict method.
        """

        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        self.log_reg.fit(x, y)

        pred = self.log_reg.predict(np.array([2, 3]))

        self.assertTrue(0 <= pred <= 1)

if __name__ == '__main__':
    unittest.main()
