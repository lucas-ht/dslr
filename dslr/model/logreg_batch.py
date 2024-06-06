"""
This module implements a logistic regression model using gradient descent.
"""

import logging
import numpy as np

from dslr.model.logreg import LogReg


BATCH_LEARNING_RATE = 1e-2
BATCH_EPOCHS        = int(5e+4)


# pylint: disable=too-many-instance-attributes
class LogRegBatch(LogReg):
    """
    This class implements a logistic regression model using gradient descent.
    """

    x: np.ndarray
    y: np.ndarray

    m: int
    n: int


    def __init__(
        self,
        learning_rate: float = BATCH_LEARNING_RATE,
        epochs: int = BATCH_EPOCHS
    ) -> None:
        """
        Initialize the model.

        Args:
            learning_rate (float): The learning rate.
            epochs (int): The number of epochs.
        """

        self.learning_rate = learning_rate
        self.epochs = epochs


    def predict(self, x: np.ndarray) -> float:
        """
        Predict the class of the input.

        Args:
            x (np.ndarray): The input values.

        Returns:
            float: The probability of the input being in the class.
        """

        linear_model = np.dot(x, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        return y_predicted # type: ignore


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data using batch gradient descent.

        Args:
            x (np.ndarray): The input values.
            y (np.ndarray): The target values.

        Returns:
            None
        """

        self.m, self.n = x.shape
        self.x, self.y = x, y

        self.weights = np.zeros(self.n)
        self.bias = 0

        logging.info('Starting training')

        for _ in range(self.epochs):
            self._update_weights()

        logging.info('Training complete')

        logging.debug('Weights: %s', self.weights)
        logging.debug('Bias: %s', self.bias)


    def _update_weights(self) -> None:
        linear_model = np.dot(self.x, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        delta_prediction = y_predicted - self.y
        delta_weight = (1 / self.m) * np.dot(self.x.T, delta_prediction)
        delta_bias = (1 / self.m) * np.sum(delta_prediction)

        self.weights -= self.learning_rate * delta_weight
        self.bias -= float(self.learning_rate * delta_bias)
