"""
This module is used to display histograms of the dataset.
"""

import logging

import matplotlib.pyplot as plt

from dslr.parser import Parser
from dslr.hogwarts import HOGWARTS_HOUSE


def main() -> None:
    """
    The main function of the describe module.
    """

    parser = Parser()
    parser.add_arg('course', str, 'The course to display the histogram for')

    df = parser.read_dataset()
    course = parser.read_course('course')

    df_grouped = df.groupby(HOGWARTS_HOUSE)[course]
    df_grouped.plot(kind='hist', alpha=0.8, legend=True)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, ValueError) as e:
        logging.error('Could not render the plot: %s', e)


if __name__ == '__main__':
    main()
