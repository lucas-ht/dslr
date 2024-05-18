""" 
    this module is used to plot the pair plot of the data.
"""
import logging

import matplotlib.pyplot as plt

import seaborn as sns

from dslr.parser import Parser

def main() -> None:
    """
        the main function to display the pair plot.
        Args:
            file_path: the path to the csv file.
            courses (str): the course to display. 
    """

    parser = Parser()
    courses = []
    parser.add_arg("course1", str, "The courses to display.")
    parser.add_arg("course2", str, "The courses to display.")
    parser.add_arg("course3", str, "The courses to display.")
    parser.add_arg("course4", str, "The courses to display.")

    for i in range(1, 5):
        courses.append(parser.read_course(f'course{i}'))
    df = parser.read_dataset()

    sns.pairplot(df, hue='Hogwarts House', vars=courses, diag_kind='hist', height=2.0, aspect=1.5)

    try:
        plt.show()
    except (RuntimeError, ValueError) as e:
        logging.error('Could not render the plot: %s', e)

if __name__ == "__main__":
    main()
