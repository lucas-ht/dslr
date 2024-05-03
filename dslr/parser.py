import argparse
import logging
import pandas as pd

class Parser:
    _file: str

    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument('file', type=str, help='Path to the dataset file')

        args = parser.parse_args()
        self._file = args.file

    def read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self._file)
        except Exception as e:
            logging.error(f'Could not parse the dataset from: `{self._file}`: {e}', exc_info=False)
            exit(1)
