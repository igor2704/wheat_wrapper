import re
import os

def _readonly(self, *args, **kwargs):
    raise RuntimeError("Cannot modify ReadOnlyDict")


class ReadOnlyDict(dict):
    __setitem__ = _readonly
    __delitem__ = _readonly
    pop = _readonly
    popitem = _readonly
    clear = _readonly
    update = _readonly
    setdefault = _readonly
    
    
class ReadOnlyList(list):
    pop = _readonly
    remove = _readonly
    append = _readonly
    clear = _readonly
    extend = _readonly
    insert = _readonly
    reverse = _readonly
    
    
def find_experiment_index(save_path : str):
    pattern = r'run(\d+)'
    dirs = os.listdir(save_path)
    largest_index = 0
    for dirname in dirs:
        res = re.search(pattern, dirname)
        if res:
            index = int(res.group(1))
            if index >= largest_index:
                largest_index = index
    return largest_index + 1