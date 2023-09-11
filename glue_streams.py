import numpy as np
import pandas as pd
import cv2
import os
import typing as tp
import shutil
from sys import argv
from itertools import chain
from collections import defaultdict


def get_all_png_jpg_files(path: str) -> tp.Iterator[str]:
    img_names: list[str] = list()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith(('png', 'PNG', 'jpg', 'JPG')):
                continue
            yield os.path.join(dirname, name)

def processing(df: pd.DataFrame) -> None:
    columns = [c for c in list(df.columns) if 'Unnamed' not in c]
    df = df[columns]
    df = df.rename(columns={'File name': 'Name'})
    df['Name'] = df['Name'].apply(lambda x: x.split('.')[0])
    return df            
            
def main():
    _, path, dir_count = argv
    table_path_no_sep = ['detection/spikelets_count.csv', 'image_wheat_masks/biomass.csv']
    table_path_sep = ['features/colorDescriptors.csv', 'features/commonResults.csv', 
                      'features/quadrangleResults.csv']
    other_tables = ['features/profileResults.csv', 'features/radialResults.csv']
    dir_names = ['detection', 'features', 'image_wheat_masks', 'crops', 'detection/coordinates']
    for dir_name in dir_names:
        os.mkdir(os.path.join(path, dir_name))
    
    tables_dct = defaultdict(list)
    for key in chain(table_path_no_sep, table_path_sep, other_tables):
        for i in range(1, int(dir_count) + 1): 
            name = os.path.join(path, str(i), key)
            if key in table_path_no_sep:
                df = pd.read_csv(name)
            else:
                df = pd.read_csv(name, sep=';')
                df = processing(df)
            tables_dct[key].append(df)
            os.remove(name)
    for key in chain(table_path_no_sep, table_path_sep, other_tables):
        pd.concat(tables_dct[key]).to_csv(os.path.join(path, key), index=False)
    
    for dir_name in dir_names:
        for i in range(1, int(dir_count) + 1):
            for name in get_all_png_jpg_files(os.path.join(path, str(i), dir_name)):
                shutil.copy2(name, os.path.join(path, dir_name))
                os.remove(name)
                
    for i in range(1, int(dir_count) + 1): 
        shutil.rmtree(os.path.join(path, str(i)))
        
if __name__ == '__main__':
    main()
    