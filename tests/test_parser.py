"""
This module contains the tests for the Parser class.
"""
import unittest
from unittest.mock import patch, mock_open

import pandas as pd
import pandas.testing as pd_testing

from dslr.parser import Parser
from dslr.hogwarts import HOGWARTS_COURSES

class TestParser(unittest.TestCase):
    """
    This class contains the tests for the Parser class.
    """

    @patch('sys.argv', ['dslr', 'test.csv', 'batch'])
    def setUp(self):
        self.parser = Parser()


    @patch('sys.argv', ['dslr'])
    def test_parse_dataset_arguments_none(self):
        """
        Test the case when no arguments are passed.
        """

        with self.assertRaises(SystemExit) as cm:
            _ = Parser().read_dataset()

        self.assertEqual(cm.exception.code, 2)


    @patch('sys.argv', ['dslr', '--foo'])
    def test_parse_dataset_arguments_incorrect(self):
        """
        Test the case when incorrect arguments are passed.
        """

        with self.assertRaises(SystemExit) as cm:
            _ = Parser().read_dataset()

        self.assertEqual(cm.exception.code, 2)


    @patch('sys.argv', ['dslr', 'test.csv', 'batch'])
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_parse_dataset_not_found(self, _):
        """
        Test the case when the file is not found.
        """

        with self.assertRaises(SystemExit) as cm:
            self.parser.read_dataset()

        self.assertEqual(cm.exception.code, 1)


    @patch('sys.argv', ['dslr', 'test.csv', 'batch'])
    @patch('builtins.open', side_effect=PermissionError)
    def test_parse_dataset_permission_error(self, _):
        """
        Test the case when there is a permission error.
        """

        with self.assertRaises(SystemExit) as cm:
            self.parser.read_dataset()

        self.assertEqual(cm.exception.code, 1)


    @patch('sys.argv', ['dslr', 'test.csv', 'batch'])
    @patch('builtins.open', new_callable=mock_open, read_data='a,b\n1.0,2.0\n')
    def test_parse_dataset(self, _):
        """
        Test the case when the model is read correctly.
        """

        df = self.parser.read_dataset()
        pd_testing.assert_frame_equal(df, pd.DataFrame({'a': [1.0], 'b': [2.0]}))


    @patch('sys.argv', ['dslr', 'test.csv', 'batch'])
    def test_parse_course_arguments_none(self):
        """
        Test the parsing of a course when no arguments are passed.
        """

        self.parser.add_arg('course', str, 'The course required')

        with self.assertRaises(SystemExit) as cm:
            self.parser.read_course('course')

        self.assertEqual(cm.exception.code, 2)


    @patch('sys.argv', ['dslr', 'test.csv', 'batch', 'foo'])
    def test_parse_course_arguments_incorrect(self):
        """
        Test the parsing of a course when an incorrect argument is passed.
        """

        self.parser.add_arg('course', str, 'The course required')

        with self.assertRaises(SystemExit) as cm:
            self.parser.read_course('course')

        self.assertEqual(cm.exception.code, 1)


    @patch('sys.argv', ['dslr', 'test.csv', 'bach', HOGWARTS_COURSES[0]])
    def test_parse_course(self):
        """
        Test the parsing of a course when an incorrect argument is passed.
        """

        self.parser.add_arg('course', str, 'The course required')
        course = self.parser.read_course('course')

        self.assertEqual(course, HOGWARTS_COURSES[0])



    @patch('sys.argv', ['dslr', 'test.csv', 'LogRegBatch'])
    def test_get_batch_valid(self):
        """
        Test get_batch method with a valid batch argument.
        """
        batch = self.parser.get_batch()
        self.assertEqual(batch, 'LogRegBatch')

    @patch('sys.argv', ['dslr', 'test.csv', 'InvalidBatch'])
    def test_get_batch_invalid(self):
        """
        Test get_batch method with an invalid batch argument.
        """

        with self.assertRaises(SystemExit) as cm:
            _ = self.parser.get_batch()

        self.assertEqual(cm.exception.code, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
