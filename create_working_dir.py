import os
import shutil
import pandas as pd
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

def main():
    _, path, dir_count, out_path = argv
    img_names = list(get_all_png_jpg_files(path))
    path_out_dir = 'working_dir'
    os.mkdir(path_out_dir)
    img_in_dir_count = len(img_names) // int(dir_count)
    dir_number = 1
    old_names = list()
    path_name_lst = list()
    for i, name in enumerate(img_names):
        if not os.path.exists(os.path.join(path_out_dir, str(dir_number))):
            os.mkdir(os.path.join(path_out_dir, str(dir_number)))
        short_name = name.split('/')[-1].split('.')[0]
        if short_name in old_names:    
            new_name = name.split('.')[0] + f'_#{i}' + '.jpg'
        else:
            new_name = name
            old_names.append(short_name)
        new_name = new_name.split('/')[-1].split('.')[0] + '.' + 'jpg'
        path_name_lst.append((name, new_name.split('.')[0]))
        shutil.copy2(name, os.path.join('working_dir', str(dir_number), new_name))
        if i >= len(img_names) - 1:
            continue
        if i + 1 >= dir_number * img_in_dir_count and dir_number < int(dir_count):
            dir_number += 1
    pd.DataFrame(path_name_lst, columns=['Path', 'Name']).to_csv(os.path.join(out_path, 'path.csv'), index=False) 

            
if __name__ == '__main__':
    main()
        