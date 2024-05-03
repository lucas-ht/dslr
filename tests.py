"""
This module is used to run all the tests in the tests directory.
"""

import unittest

def main() -> None:
    """
    The main function for running the tests.
    """
    tests = unittest.TestLoader().discover('tests', pattern='test_*.py')
    unittest.TextTestRunner().run(tests)

if __name__ == '__main__':
    main()
