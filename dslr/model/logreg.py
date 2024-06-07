"""
This module implements a logistic regression model using gradient descent.
"""

import numpy as np


class LogReg:
    """
    This class implements a logistic regression model using gradient descent.
    """

    learning_rate: float
    epochs: int

    weights: np.ndarray
    bias: float


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.

        Args:
            x (np.ndarray): The input values.
            y (np.ndarray): The target values.
        """

        raise NotImplementedError


    def predict(self, x: np.ndarray) -> float:
        """
        Predict the class of the input.

        Args:
            x (np.ndarray): The input values.

        Returns:
            float: The probability of the input being in the class.
        """

        raise NotImplementedError

    def sigmoid(self, z: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the sigmoid function.

        Args:
            z (float | np.ndarray): The input values.

        Returns:
            float | np.ndarray: The output values (between 0 and 1).
        """

        return 1 / (1 + np.exp(-z))
