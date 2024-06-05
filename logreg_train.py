"""
This module is used to train a logistic regression model on the dataset.
"""

import logging

import numpy as np

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logistic_regression import LogisticRegression

def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    df = Parser().read_dataset().dropna()
    x = Parser.get_x(df)
    y = Parser.get_y(df)

    model = OvrClassifier(LogisticRegression)
    model.fit(x, y)

    total_ok = 0

    for (k, v) in enumerate(x):

        result = model.predict(v)

        predicted_class = np.argmax(result)
        expected_class = np.argmax(y[k])

        if predicted_class == expected_class:
            total_ok += 1

        logging.debug('Prediction: %s, Expected: %s', predicted_class, expected_class)

    logging.info('Accuracy: %.4f', total_ok / len(x))


if __name__ == '__main__':
    main()
