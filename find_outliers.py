import pandas as pd
import numpy as np
import scipy.stats
from sys import argv


def main():
    level = 0.998
    if len(argv) > 3:
        _, table_path, out_table_path, level = argv
        level = float(level)
    else:
        _, table_path, out_table_path = argv
    alpha = scipy.stats.norm.ppf((1 - level)/2 + level)
        
    table = pd.read_csv(table_path)
    columns = [c for c in list(table.columns) if 'Unnamed' not in c]
    table = table[columns]
    drop_columns = ['Name', 'Path']
    for col in table.drop(drop_columns, axis=1).columns:
        table[col] = table[col].apply(lambda x: x if type(x) not in [object, str] 
                                                  else float(str(x).replace(',', '.')))
    
    left = table.drop(drop_columns, axis=1).mean() - level * table.drop(drop_columns, axis=1).std()
    right = table.drop(drop_columns, axis=1).mean() + level * table.drop(drop_columns, axis=1).std()
    left = left.values
    right = right.values
    
    outliers = np.logical_or(table.drop(drop_columns, axis=1).values < left, 
                             table.drop(drop_columns, axis=1).values > right)
    names, features = np.nonzero(outliers)
    rows = list()
    for i in range(len(table)):
        name = table.iloc[i]['Name']
        path = table.iloc[i]['Path']
        outliers = table.drop(drop_columns, axis=1).columns[features[names == i]]
        rows.append((name, path, list(outliers)))
    pd.DataFrame(rows, columns=['Name', 
                                'Path', 
                                f'outliers features (confident level = {level})']).to_csv(out_table_path)

if __name__ == '__main__': 
    main()