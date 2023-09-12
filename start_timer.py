import pickle
from sys import argv
from datetime import datetime


def main():
    _, pkl_path = argv
    dt = datetime.now()
    with open(pkl_path, 'wb') as pkl:
        pickle.dump(dt, pkl)

if __name__ == '__main__':
    main()
        