import numpy as np
import cv2
import os
import typing as tp
from sys import argv


def get_all_png_jpg_files(path: str) -> tp.Iterator[str]:
    img_names: list[str] = list()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith(('png', 'PNG', 'jpg', 'JPG')):
                continue
            yield os.path.join(dirname, name)

def decrease_size_imgs(path: str,
                       size: tp.Tuple[int, int] = (150, 150)) -> None:
    names = get_all_png_jpg_files(path)
    for name in names:
        img = cv2.imread(name)
        os.remove(name)
        img = cv2.resize(img, size)
        cv2.imwrite(name, img)

def main():
    if len(argv) > 2:
        _, path, w, h = argv
    else:
        _, path = argv
        w, h = 150, 150
    decrease_size_imgs(path, (int(w), int(h)))

if __name__ == '__main__':    
    main()
        