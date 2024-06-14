"""
This module is used to train a logistic regression model on the dataset.
"""
# pylint:disable=duplicate-code

import logging

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_batch import LogRegBatch
from dslr.model.logreg_stochastic import LogRegStochastic



def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    df = Parser().read_dataset()
    df = Parser.fill_dataset(df)

    y = Parser.get_y(df)
    x = Parser.get_x(df)

    arg =Parser().get_batch()
    if arg != 'LogRegBatch':
            model = OvrClassifier(LogRegStochastic)
    model = OvrClassifier(LogRegBatch)

    logging.info('Training models')
    model.fit(x, y)
    logging.info('Training complete')

    model.save_models('model.json')


if __name__ == '__main__':
    main()
