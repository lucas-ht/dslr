"""
This module is used to train a logistic regression model on the dataset.
"""
# pylint:disable=duplicate-code

import logging

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier


def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = Parser()

    parser.add_arg('--model', str, 'The model used',
                   required=False, choices=['batch', 'stochastic'])

    model = parser.read_model()
    logging.debug('Using model %s', model)

    df = parser.read_dataset()
    df = Parser.fill_dataset(df)

    y = Parser.get_y(df)
    x = Parser.get_x(df)

    ovr = OvrClassifier(model)

    logging.info('Training models')
    ovr.fit(x, y)
    logging.info('Training complete')

    ovr.save_models('model.json')


if __name__ == '__main__':
    main()
