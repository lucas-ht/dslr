import unittest
from unittest.mock import patch, mock_open

from dslr.parser import Parser
import pandas as pd
import pandas.testing as pd_testing


class TestParser(unittest.TestCase):
    @patch('sys.argv', ['dslr', 'test.csv'])
    def setUp(self):
        self.parser = Parser()

    @patch('sys.argv', ['dslr'])
    def test_parse_arguments_none(self):
        with self.assertRaises(SystemExit) as cm:
            _ = Parser()

        self.assertEqual(cm.exception.code, 2)

    @patch('sys.argv', ['dslr', '--foo'])
    def test_parse_arguments_incorrect(self):
        with self.assertRaises(SystemExit) as cm:
            _ = Parser()

        self.assertEqual(cm.exception.code, 2)

    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_parse_not_found(self, _):
        with self.assertRaises(SystemExit) as cm:
            self.parser.read()

        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', side_effect=PermissionError)
    def test_parse_permission_error(self, _):
        with self.assertRaises(SystemExit) as cm:
            self.parser.read()

        self.assertEqual(cm.exception.code, 1)

    @patch('builtins.open', new_callable=mock_open, read_data='a,b\n1.0,2.0\n')
    def test_parse_model(self, _):
        data = self.parser.read()
        pd_testing.assert_frame_equal(data, pd.DataFrame({'a': [1.0], 'b': [2.0]}))

if __name__ == '__main__':
    unittest.main(verbosity=2)
