"""
This module is used to train a logistic regression model on the dataset.
"""

import logging

import numpy as np

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_batch import LogRegBatch


def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    df = Parser().read_dataset().dropna()
    x = Parser.get_x(df)
    y = Parser.get_y(df)

    model = OvrClassifier(LogRegBatch)

    logging.info('Training models')
    model.fit(x, y)
    logging.info('Training complete')

    total_ok = 0
    for (k, v) in enumerate(x):
        result = model.predict(v)

        predicted_class = np.argmax(result)
        expected_class = np.argmax(y[k])

        if predicted_class == expected_class:
            total_ok += 1

    logging.info('Accuracy: %.4f', total_ok / len(x))


if __name__ == '__main__':
    main()
