"""
This module contains the Parser class which is used for
parsing command line arguments and reading the dataset.
"""

import argparse
import logging
import sys
from typing import Any, Type, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import normalize

from dslr.model.logreg import LogReg
from dslr.model.logreg_batch import LogRegBatch
from dslr.model.logreg_stochastic import LogRegStochastic
from dslr.hogwarts import HOGWARTS_COURSES, HOGWARTS_HOUSE, HOGWARTS_HOUSES


class Parser:
    """
    The Parser class is used for parsing command line arguments and reading the dataset.

    Attributes:
        _file (str): The path to the dataset file.
    """

    _parser: argparse.ArgumentParser

    def __init__(self) -> None:
        """
        The constructor for the Parser class.
        It initializes the parser.
        """

        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('file', type=str, help='Path to the dataset file')


    def add_arg(self, name: str, argument_type: Type, argument_help: str, **kwargs: Any) -> None:
        """
        This method adds an argument to the parser.

        Args:
            name (str): The name of the argument.
            argument_type (Type): The type of the argument.
            argument_help (str): The help message for the argument.
            **kwargs (Any): Additional keyword arguments for the argument.
        """

        self._parser.add_argument(name, type=argument_type, help=argument_help, **kwargs)


    def read_dataset(self) -> pd.DataFrame:
        """
        This method reads the dataset from the file specified in the command line arguments.

        Returns:
            pd.DataFrame: The dataset read from the file.

        Raises:
            SystemExit: If the dataset file could not be parsed.
        """

        args = self._parser.parse_args()
        file = args.file

        try:
            return pd.read_csv(file)
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            logging.error('Could not read the dataset from: `%s`: %s', file, e)
            sys.exit(1)
        # pylint: disable=broad-except
        except Exception as e:
            logging.error('Could not parse the dataset from: `%s`: %s', file, e)
            sys.exit(1)


    @staticmethod
    def fill_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """
        This method fills the missing values in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to fill.

        Returns:
            pd.DataFrame: The filled dataset.
        """

        try:
            for course in HOGWARTS_COURSES:
                df[course] = df[course].fillna(df[course].mean())
        except KeyError as e:
            logging.error('Could not fill the dataset: %s', e)
            sys.exit(1)

        return df


    def read_arg(self, name: str) -> str:
        """
        This method reads the argument from the command line arguments.

        Args:
            name (str): The name of the argument.

        Returns:
            str: The argument to read.
        """

        args = self._parser.parse_args()
        return getattr(args, name)


    def read_course(self, name: str) -> str:
        """
        This method reads the course from the command line arguments.

        Args:
            name (str): The name of the argument.

        Returns:
            str: The course to read.

        Raises:
            SystemExit: If the course is not valid.
        """

        course = self.read_arg(name)

        if course not in HOGWARTS_COURSES:
            logging.error('`%s` is not a valid course.', course)
            sys.exit(1)

        return course


    def read_model(self) -> Type[LogReg]:
        """
        This method reads the model from the command line arguments.

        Returns:
            Type: The model to read.

        Raises:
            SystemExit: If the model is not valid.
        """

        model = self.read_arg('model') or 'batch'

        match model.lower():
            case 'batch':
                return LogRegBatch
            case 'stochastic':
                return LogRegStochastic
            case _:
                sys.exit(1)


    @staticmethod
    def get_x(df: pd.DataFrame) -> np.ndarray:
        """
        This method is used to get the X values.
        """

        try:
            x = df.drop(columns=[
                'Index',
                'Hogwarts House',
                'First Name',
                'Last Name',
                'Birthday',
                'Best Hand',
            ]).to_numpy()

            return normalize(X=x, axis=0, norm='max') # type: ignore

        except (KeyError, ValueError) as e:
            logging.error('Could not get the X values: %s', e)
            sys.exit(1)


    @staticmethod
    def get_y(df: pd.DataFrame) -> np.ndarray:
        """
        This method is used to get the y values.
        """

        try:
            return label_binarize(df[HOGWARTS_HOUSE], classes=HOGWARTS_HOUSES) # type: ignore

        except (KeyError, ValueError) as e:
            logging.error('Could not get the y values: %s', e)
            sys.exit(1)


    @staticmethod
    def convert_label(predictions: np.ndarray) -> str:
        """
        This method is used to convert the predictions to the Hogwarts house.

        Args:
            predictions (np.ndarray): The predictions to convert.

        Returns:
            str: The Hogwarts house.
        """

        predicted_index = np.argmax(predictions)
        return HOGWARTS_HOUSES[predicted_index]


    @staticmethod
    def save_houses(predictions: List[str]) -> None:
        """
        This method is used to save the predictions to a file.

        Args:
            predictions (np.ndarray): The predictions to save.
        """

        logging.info('Writing predictions to `houses.csv`')

        try:
            df = pd.DataFrame(predictions, columns=['Hogwarts House'])
            df.to_csv('houses.csv', index_label='Index')
        # pylint: disable=broad-except
        except Exception as e:
            logging.error('An error occurred: %s', e)
            sys.exit(1)
