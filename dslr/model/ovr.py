"""
OVR (One vs Rest) classifier
"""

import logging
from typing import Type
import json

import numpy as np
from sklearn.metrics import accuracy_score

from dslr.parser import Parser
from dslr.hogwarts import HOGWARTS_HOUSES

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


    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the accuracy of the model.
        """

        total_ok = 0
        for (k, v) in enumerate(x):
            result = self.predict(v)

            predicted_class = np.argmax(result)
            expected_class = np.argmax(y[k])

            predicted_label = Parser.convert_predictions_to_labels(result, HOGWARTS_HOUSES)
            print(f'Predicted: {predicted_label} ')
            if predicted_class == expected_class:
                total_ok += 1
        return total_ok / len(x)


    def save_models(self, path: str) -> None:
        """
        Save the model to a file.
        """

        models_data = []
        for model in self.models:
            model_data = {
                'weights': model.weights.tolist(),
                'bias': model.bias
            }
            models_data.append(model_data)

        try:
            with open(path, 'w', encoding='utf-8') as file:
                logging.debug('Saving the model to %s', path)
                json.dump(models_data, file)
        # pylint: disable=broad-except
        except Exception as e:
            logging.error('An error occurred while saving the model: %s', e)


    def load_models(self, path: str) -> None:
        """
        Load the model from a file.
        """

        model_data = []
        try:
            with open(path, 'r', encoding='utf-8') as file:
                logging.debug('Loading the model from %s', path)
                models_data = json.load(file)
        except FileNotFoundError:
            logging.error('The file does not exist.')
            return
        except PermissionError:
            logging.error('You do not have permission to read this file.')
            return
        # pylint: disable=broad-except
        except Exception as e:
            logging.error('An error occurred while loading the model: %s', e)

        self.models = []
        for model_data in models_data:
            model = self.model()
            model.weights = np.array(model_data['weights'])
            model.bias = model_data['bias']
            self.models.append(model)
