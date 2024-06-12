"""
this is model of the stochastic gradient descent
"""

import logging

import numpy as np

from dslr.model.logreg import LogReg


LEARNING_RATE = 0.01
EPOCHS        = 5000


class LogRegStochastic(LogReg):
    """
        This class implements a logistic regression model using gradient descent
    """

    m: int

    x: np.ndarray
    y : np.ndarray


    def __init__(self, learning_rate: float = LEARNING_RATE, epochs: int = EPOCHS) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    def initialize_weights(self, row):
        """ In this function, we will initialize our weights and bias"""

        w = np.zeros(row.shape[1])
        b = 0
        return w,b


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data using stochastic gradient descent.

        Args:
            x (np.ndarray): The input values.
            y (np.ndarray): The target values.
        Returns:
            None
        """

        self.m = x.shape[0]

        self.x, self.y = x, y
        self.weights , self.bias = self.initialize_weights(self.x)

        logging.info('Starting training')

        for i in range(self.epochs):
            j = i % self.m
            self._update_weights(j)

        logging.info('Training complete')

        logging.debug('Weights: %s', self.weights)
        logging.debug('Bias: %s', self.bias)


    def _update_weights(self, i) -> None:
        linear_model = np.dot(self.x[i], self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)

        delta_prediction = y_predicted - self.y[i]
        delta_weight = self.x[i] * delta_prediction + (self.weights / self.m)
        delta_bias = (1 / self.m) * delta_prediction

        self.weights -= self.learning_rate * delta_weight
        self.bias -= self.learning_rate * delta_bias


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

        return y_predicted
