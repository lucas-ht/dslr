"""
this is model of the stochastic gradient descent
"""

import logging

import numpy as np

from dslr.model.logreg import LogReg


LEARNING_RATE = 0.01
EPOCHS        = 1


class Sgd(LogReg):
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

    def gradient_dw(self,x, y, w, b):
        """In this function, we will compute the gradient w.r.to w """

        yhat = self.sigmoid(np.dot(x, w) + b)
        gradient = x * (yhat - y).reshape(-1,1) + (self.learning_rate * w/self.m)
        return gradient

    def gradient_db(self, x, y, w, b):
        """In this function, we will compute the gradient w.r.to b """

        yhat = self.sigmoid(np.dot(x, w) + b)
        gradient = yhat - y
        return gradient


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

        for _ in range(self.epochs):
            for i in range(self.m):
                dw = self.gradient_dw(self.x[i],
                                    self.y[i],
                                    self.weights,
                                    self.bias)
                db = self.gradient_db(self.x[i], self.y[i], self.weights, self.bias)
                dw = np.sum(dw, axis=0)
                db = np.sum(db, axis=0)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        logging.info('Training complete')
        logging.debug('Weights: %s', self.weights)
        logging.debug('Bias: %s', self.bias)


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
