"""
This module contains the Parser class which is used for
parsing command line arguments and reading the dataset.
"""

import argparse
import logging
import sys
import pandas as pd

class Parser:
    """
    The Parser class is used for parsing command line arguments and reading the dataset.

    Attributes:
        _file (str): The path to the dataset file.
    """
    _file: str

    def __init__(self) -> None:
        """
        The constructor for the Parser class.
        It initializes the parser and parses the command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('file', type=str, help='Path to the dataset file')

        args = parser.parse_args()
        self._file = args.file

    def read_dataset(self) -> pd.DataFrame:
        """
        This method reads the dataset from the file specified in the command line arguments.

        Returns:
            pd.DataFrame: The dataset read from the file.

        Raises:
            SystemExit: If the dataset file could not be parsed.
        """
        try:
            return pd.read_csv(self._file)
        except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
            logging.error('Could not read the dataset from: `%s`: %s', self._file, e)
            sys.exit(1)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logging.error('Could not parse the dataset from: `%s`: %s', self._file, e)
            sys.exit(1)

    def read_model(self):
        """
        This method is used to read the model.
        """
