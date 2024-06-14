"""
This module is used to train a logistic regression model on the dataset.
"""
# pylint:disable=duplicate-code

import logging

import numpy as np
from sklearn.metrics import accuracy_score

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier


def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = Parser()

    parser.add_arg('model_path', str, 'The path to the model file')
    parser.add_arg('--model', str, 'The model used',
                   required=False, choices=['batch', 'stochastic'])

    model_path = parser.read_arg('model_path')

    model = parser.read_model()
    logging.debug('Using model %s', model)

    df = parser.read_dataset()
    df = Parser.fill_dataset(df)

    y = Parser.get_y(df)
    x = Parser.get_x(df)

    ovr = OvrClassifier(model)
    ovr.load_models(model_path)

    logging.info('Predicting classes')

    predictions = []
    for value in x:
        prediction = ovr.predict(value)
        predictions.append(np.argmax(prediction))

    logging.info('Prediction complete')

    truth = [np.argmax(i) for i in y]
    accuracy = accuracy_score(truth, predictions)

    logging.info('Accuracy: %.4f', accuracy)


if __name__ == '__main__':
    main()
