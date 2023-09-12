import os
import pickle
from sys import argv
from datetime import datetime


def main():
    _, pkl_path, text = argv
    new_dt = datetime.now()
    with open(pkl_path, 'rb') as pkl:
        old_dt = pickle.load(pkl)
    print('\n' + text + str(new_dt - old_dt) + '\n')
    os.remove(pkl_path)

if __name__ == '__main__':
    main()