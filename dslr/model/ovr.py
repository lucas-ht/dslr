"""
OVR (One vs Rest) classifier
"""

import logging
from typing import Type

import numpy as np

from dslr.hogwarts import HOGWARTS_HOUSES
from dslr.model.logistic_regression import LogisticRegression

class OvrClassifier:
    """
    This class implements a One vs Rest classifier.
    """

    def __init__(self, model: Type[LogisticRegression]):
        self.model = model
        self.models = []


    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.
        """

        self.models = []
        for (key, value) in enumerate(HOGWARTS_HOUSES):
            logging.info('Training model for class %s', value)

            model = self.model()
            model.fit(x, y[:, key])

            self.models.append(model)


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the class of the input.
        """

        return np.array(
            [model.predict(x) for model in self.models]
        )
