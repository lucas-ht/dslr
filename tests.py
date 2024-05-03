import unittest

def main() -> None:
    tests = unittest.TestLoader().discover('tests', pattern='test_*.py')
    unittest.TextTestRunner().run(tests)

if __name__ == '__main__':
    main()
