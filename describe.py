"""
This module contains the describe implementation.
"""

from dslr.parser import Parser

def main():
    """
    The main function of the describe module.
    """
    df = Parser().read_dataset()
    print(df[df['Hogwarts House'] == 'Ravenclaw'])

if __name__ == '__main__':
    main()
