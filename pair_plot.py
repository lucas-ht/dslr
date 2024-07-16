"""
This module is used to display a pair plot of the dataset.
"""

import logging

import matplotlib.pyplot as plt
import seaborn as sns

from dslr.hogwarts import HOGWARTS_HOUSE
from dslr.parser import Parser

def main() -> None:
    """
    The main function to display the pair plot.
    """

    parser = Parser()

    parser.add_arg("first_course", str, "The first course to display on the pair plot")
    parser.add_arg("second_course", str, "The second course to display on the pair plot")
    parser.add_arg("third_course", str, "The third course to display on the pair plot")
    parser.add_arg("fourth_course", str, "The fourth course to display on the pair plot")

    df = parser.read_dataset()

    courses = [
        parser.read_course("first_course"),
        parser.read_course("second_course"),
        parser.read_course("third_course"),
        parser.read_course("fourth_course"),
    ]

    sns.pairplot(df, hue=HOGWARTS_HOUSE, vars=courses, diag_kind='hist', height=2.0, aspect=1.5)

    try:
        plt.show()
    except (RuntimeError, ValueError) as e:
        logging.error('Could not render the plot: %s', e)

if __name__ == "__main__":
    main()
