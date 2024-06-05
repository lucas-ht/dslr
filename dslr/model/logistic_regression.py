"""
This module implements a logistic regression model using gradient descent.
"""

import logging
import numpy as np

# pylint: disable=too-many-instance-attributes
class LogisticRegression:
    """
    This class implements a logistic regression model using gradient descent.
    """

    learning_rate: float
    epochs: int

    weights: np.ndarray
    bias: float

    x: np.ndarray
    y: np.ndarray

    m: int
    n: int


    def __init__(self, learning_rate: float = 0.01, epochs: int = 10000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs


    def _sigmoid(self, z: float) -> float:
        return 1 / (1 + np.exp(-z))


    def fit(self, x: np.ndarray, y) -> None:
        """
        Fit the model to the data.
        """

        self.m, self.n = x.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.x = x
        self.y = y

        logging.info('Starting training')

        for _ in range(self.epochs):
            self._update_weights()

        logging.info('Training complete')
        logging.info('Weights: %s', self.weights)


    def _update_weights(self) -> None:
        linear_model = np.dot(self.x, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)

        delta_weight = (1 / self.m) * np.dot(self.x.T, (y_predicted - self.y))
        delta_bias = (1 / self.m) * np.sum(y_predicted - self.y)

        self.weights -= self.learning_rate * delta_weight
        self.bias -= float(self.learning_rate * delta_bias)


    def predict(self, x) -> float:
        """
        Predict the output for the given input.
        """

        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return y_predicted
