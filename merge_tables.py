import pandas as pd
import os
from sys import argv


def processing(df: pd.DataFrame) -> None:
    columns = [c for c in list(df.columns) if 'Unnamed' not in c]
    df = df[columns]
    df = df.rename(columns={'File name': 'Name'})
    df['Name'] = df['Name'].apply(lambda x: x.split('.')[0])
    return df

def main():
    _, path = argv
    tables = dict()
    tables[0] = pd.read_csv(os.path.join(path, 'detection/spikelets_count.csv'))
    tables[1] = pd.read_csv(os.path.join(path, 'features/colorDescriptors.csv'))
    # tables[1] = processing(tables[1])
    tables[2] = pd.read_csv(os.path.join(path, 'features/commonResults.csv'))
    # tables[2] = processing(tables[2])
    tables[3] = pd.read_csv(os.path.join(path, 'features/quadrangleResults.csv'))
    # tables[3] = processing(tables[3])
    tables[4] = pd.read_csv(os.path.join(path, 'path.csv'))
    tables[5] = pd.read_csv(os.path.join(path, 'image_wheat_masks/biomass.csv'))
    for i in tables.keys():
        tables[0] = tables[0].merge(tables[i], on='Name')
    tables[0].to_csv(os.path.join(path, 'all_featurs.csv'))

if __name__ == '__main__':
    main()
