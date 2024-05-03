from dslr.parser import Parser

def main():
    parser = Parser()
    data = parser.read()
    print(data.describe())

if __name__ == '__main__':
    main()
