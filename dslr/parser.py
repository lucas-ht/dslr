"""
This module contains the Parser class which is used for
parsing command line arguments and reading the dataset.
"""

import argparse
import logging
import sys
from typing import Type

import pandas as pd

from dslr.hogwarts import HOGWARTS_COURSES


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


    def add_arg(self, name: str, argument_type: Type, argument_help: str) -> None:
        """
        This method adds an argument to the parser.

        Args:
            name (str): The name of the argument.
            argument_type (Type): The type of the argument.
            argument_help (str): The help message for the argument.
        """

        self._parser.add_argument(name, type=argument_type, help=argument_help)


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
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.error('Could not parse the dataset from: `%s`: %s', file, e)
            sys.exit(1)


    def read_model(self):
        """
        This method is used to read the model.
        """


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

        args = self._parser.parse_args()
        course = getattr(args, name)

        if course not in HOGWARTS_COURSES:
            logging.error('`%s` is not a valid course.', course)
            sys.exit(1)

        return course
