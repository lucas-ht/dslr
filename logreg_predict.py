"""
This module is used to predict the class of the input using the trained model.
"""

import logging

from dslr.parser import Parser
from dslr.model.ovr import OvrClassifier
from dslr.model.logreg_batch import LogRegBatch


def main():
    """
    The main function to predict the class of the input using the trained model.
    """

    logging.basicConfig(level=logging.DEBUG)

    parser = Parser()

    parser.add_arg('model', str, 'The path to the model file')
    model_path = parser.read_arg('model')

    df = parser.read_dataset()
    df = Parser.fill_dataset(df)

    x = Parser.get_x(df)

    ovr = OvrClassifier(LogRegBatch)
    ovr.load_models(model_path)

    logging.info('Predicting classes')

    predictions = []
    for (_, v) in enumerate(x):
        prediction = ovr.predict(v)
        predictions.append(Parser.convert_label(prediction))

    logging.info('Prediction complete')
    Parser.save_houses(predictions)


if __name__ == '__main__':
    main()
