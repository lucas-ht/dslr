"""
This module is used to train a logistic regression model on the dataset.
"""
# pylint:disable=duplicate-code

import logging

import numpy as np
from sklearn.metrics import accuracy_score

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_batch import LogRegBatch


def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = Parser()

    parser.add_arg('model', str, 'The path to the model file')
    model_path = parser.read_arg('model')

    df = parser.read_dataset().dropna()

    y = Parser.get_y(df)
    x = Parser.get_x(df)

    ovr = OvrClassifier(LogRegBatch)
    ovr.load_models(model_path)

    logging.info('Predicting classes')

    predictions = []
    for (_, v) in enumerate(x):
        prediction = ovr.predict(v)
        predictions.append(np.argmax(prediction))

    logging.info('Prediction complete')

    truth = [np.argmax(i) for i in y]
    accuracy = accuracy_score(truth, predictions)

    logging.info('Accuracy: %.4f', accuracy)


if __name__ == '__main__':
    main()
