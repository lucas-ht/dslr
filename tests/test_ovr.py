"""
Test the One-vs-Rest classifier.
"""

import unittest
import numpy as np
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_batch import LogRegBatch

class TestOvrClassifier(unittest.TestCase):
    """
    Test the One-vs-Rest classifier.
    """

    def setUp(self):
        self.ovr = OvrClassifier(LogRegBatch)
        self.x = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([[0, 1], [1, 0], [0, 1]])


    def test_fit(self):
        """
        Test fitting the model.
        """

        self.ovr.fit(self.x, self.y)
        self.assertEqual(len(self.ovr.models), 2)


    def test_predict(self):
        """
        Test predicting the class of the input.
        """

        self.ovr.fit(self.x, self.y)
        predictions = self.ovr.predict(self.x)
        self.assertEqual(predictions.shape, (2, 3))

if __name__ == '__main__':
    unittest.main()
