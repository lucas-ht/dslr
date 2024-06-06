"""
OVR (One vs Rest) classifier
"""

import logging
from typing import Type

import numpy as np

from dslr.model.logreg import LogReg


class OvrClassifier:
    """
    This class implements a One vs Rest classifier.
    """

    def __init__(self, model: Type[LogReg]):
        self.model = model
        self.models = []


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        """

        self.models = []
        for i in range(y.shape[1]):
            logging.info('Training model for class #%s', i)

            model = self.model()
            model.fit(x, y[:, i])

            self.models.append(model)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of the input.
        """

        return np.array(
            [model.predict(x) for model in self.models]
        )
