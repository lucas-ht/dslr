"""
This module contains the tests for the Parser class.
"""
import unittest
from unittest.mock import patch, mock_open

import pandas as pd
import pandas.testing as pd_testing
from dslr.parser import Parser


class TestParser(unittest.TestCase):
    """
    This class contains the tests for the Parser class.
    """
    @patch('sys.argv', ['dslr', 'test.csv'])
    def setUp(self):
        self.parser = Parser()

    @patch('sys.argv', ['dslr'])
    def test_parse_arguments_none(self):
        """
        Test the case when no arguments are passed.
        """
        with self.assertRaises(SystemExit) as cm:
            _ = Parser()

        self.assertEqual(cm.exception.code, 2)

    @patch('sys.argv', ['dslr', '--foo'])
    def test_parse_arguments_incorrect(self):
        """
        Test the case when incorrect arguments are passed.
        """
        with self.assertRaises(SystemExit) as cm:
            _ = Parser()

        self.assertEqual(cm.exception.code, 2)

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_parse_not_found(self, _):
        """
        Test the case when the file is not found.
        """
        with self.assertRaises(SystemExit) as cm:
            self.parser.read_dataset()

        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', side_effect=PermissionError)
    def test_parse_permission_error(self, _):
        """
        Test the case when there is a permission error.
        """
        with self.assertRaises(SystemExit) as cm:
            self.parser.read_dataset()

        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open, read_data='a,b\n1.0,2.0\n')
    def test_parse_model(self, _):
        """
        Test the case when the model is read correctly.
        """
        data = self.parser.read_dataset()
        pd_testing.assert_frame_equal(data, pd.DataFrame({'a': [1.0], 'b': [2.0]}))

if __name__ == '__main__':
    unittest.main(verbosity=2)
