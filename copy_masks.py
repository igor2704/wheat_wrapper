import os
import shutil
import typing as tp
from sys import argv


def get_all_png_files(path: str) -> tp.Iterator[str]:
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith('png'):
                continue
            yield os.path.join(dirname, name)

def main():
    _, path_in, path_out_dir = argv
    for name in get_all_png_files(path_in):
        shutil.copy2(name, path_out_dir)

if __name__ == '__main__':
    main()
