from dslr.parser import Parser
from describe.describe import Describe

def main():
    parser = Parser()
    data = parser.read()
    print(f'Describe from Class Describe')
    describe = Describe(data)
    describe.describe()
    print()
    print(f'Describe from Pandas')
    print(data.describe())

if __name__ == '__main__':
    main()
