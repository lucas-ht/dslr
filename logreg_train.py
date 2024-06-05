"""
This module is used to train a logistic regression model on the dataset.
"""

from dslr.parser import Parser
from dslr.model.logistic_regression import LogisticRegression

def main():
    """
    The main function to train a logistic regression model on the dataset.
    """

    df = Parser().read_dataset().dropna()
    x = Parser.get_x(df)
    y = Parser.get_y(df)

    model = LogisticRegression()
    model.fit(x, y[:, 0])

    total_ok = 0

    for (k, v) in enumerate(x):

        result = model.predict(v)

        if result > 0.8 and y[k][0] == 1:
            total_ok += 1

        if result < 0.2 and y[k][0] == 0:
            total_ok += 1


    print(f'Accuracy: {total_ok / len(x)}')

if __name__ == '__main__':
    main()
