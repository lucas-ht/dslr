import pandas as pd
import numpy as np
from describe import Describe


def test_describe():
    expected_result = pd.DataFrame({
        'A': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5],
        'B': [5, 3.500000, 1.581139, 1.5, 2.5, 3.5, 4.5, 5.5, 5],
        'C': [5, 1.000000, 0.000000, 1, 1, 1, 1, 1, 1],
        'D': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5],
        'E': [5, 3.000000, 1.581139, 1, 2, 3, 4, 5, 5]
    }, index=['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'unique'])
    _describe_ = Describe(expected_result)
    _describe_.describe()


def test_describe_with_large_dataset():
    np.random.seed(0)  
    data = pd.DataFrame({
        'A': np.random.randint(0, 100, size=100000),
        'B': np.random.normal(0, 1, size=100000),
        'C': np.random.uniform(0, 1, size=100000),
        'D': np.random.lognormal(0, 1, size=100000)
    })

    data_ = Describe(data)
    data_.describe()


def test_describe_with_specific_percentiles():
    percentiles = [.1, .4, .9]
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [1.5, 2.5, 3.5, 4.5, 5.5],
        'C': [1, 1, 1, 1, 1],
        'D': [1, 2, 3, 4, 5],
        'E': [1, 2, 3, 4, 5]
    })
    describe_ = Describe(data)
    print('Data describe class method ')
    describe_.describe()
    print()
    print('Data describe method with pandas')
    print(data.describe())
    print()
    print('Data describe class method  with specific percentiles')
    describe_.describe(percentiles=percentiles)


if __name__ == '__main__':
    print('Test describe class method')
    test_describe()
    print()
    print('Test describe class method with large dataset')
    test_describe_with_large_dataset()
    print()
    print('Test describe class method with specific percentiles')
    test_describe_with_specific_percentiles()
