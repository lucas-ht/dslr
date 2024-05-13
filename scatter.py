"""
    This module is used to display a scatter of the dataset.
"""

import logging

import matplotlib.pyplot as plt

from dslr.hogwarts import HOGWARTS_HOUSE

from dslr.parser import Parser


def main() -> None:
    """ The main function to display a scatter of the dataset."""
    parser = Parser()

    parser.add_arg('first_course', str, 'The course to display the histogram for')
    parser.add_arg('second_course', str, 'The course to display the histogram for')

    df = parser.read_dataset()
    x = parser.read_course('first_course')
    y = parser.read_course('second_course')

    for house, group in df.groupby(HOGWARTS_HOUSE):
        plt.scatter(group[x], group[y], alpha=0.6, label=house)
    plt.legend(title="Hogwarts House")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Scatter of {x} and {y} for each house')

    try:
        plt.show()
    except (RuntimeError, ValueError) as e:
        logging.error('Could not render the plot: %s', e)


if __name__ == '__main__':
    main()
