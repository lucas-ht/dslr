"""
This module is used  Stochastic Gradient Descent .
"""
import logging

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_stochastic import LogRegStochastic

def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    logging.basicConfig(level=logging.DEBUG)

    df = Parser().read_dataset().dropna()
    x = Parser.get_x(df)
    y = Parser.get_y(df)

    model = OvrClassifier(LogRegStochastic)

    logging.info('Training models')
    model.fit(x, y)
    logging.info('Training complete')

    total_ok = model.accuracy(x, y)

    logging.info('Accuracy: %.4f', total_ok )


if __name__ == '__main__':
    main()
