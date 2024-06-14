"""
This module is used to predict the class of the input using the trained model.
"""
# pylint:disable=duplicate-code

import logging

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier


def main():
    """
    The main function to predict the class of the input using the trained model.
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

    x = Parser.get_x(df)

    ovr = OvrClassifier(model)
    ovr.load_models(model_path)

    logging.info('Predicting classes')

    predictions = []
    for value in x:
        prediction = ovr.predict(value)
        predictions.append(Parser.convert_label(prediction))

    logging.info('Prediction complete')
    Parser.save_houses(predictions)


if __name__ == '__main__':
    main()
