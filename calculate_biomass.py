import os
import numpy as np
import pandas as pd
import cv2
import typing as tp
from sys import argv


def get_all_png_files(path: str) -> tp.Iterator[str]:
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith('png'):
                continue
            yield os.path.join(dirname, name)

def get_max_contour_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
    return cv2.fillPoly(np.zeros(mask.shape, dtype='uint8'), pts=[max_contour], color=1)

def get_biomass_max_contour(mask: np.ndarray) -> float:
    new_mask = mask * get_max_contour_mask(mask)
    new_mask = np.where(new_mask > 0, 1, 0)
    return np.sum(new_mask)

def get_wheat_spikelets_biomass(mask: np.ndarray) -> float:
    colorchecker_biomass = get_biomass_max_contour(mask[:, :, 0])
    spikelets_biomass = get_biomass_max_contour(mask[:, :, 1])
    return spikelets_biomass / colorchecker_biomass

def main():
    _, in_path, out_path = argv
    biomass_lst = []
    for name in get_all_png_files(in_path):
        mask = cv2.imread(name)
        biomass = get_wheat_spikelets_biomass(mask)
        biomass_lst.append((name.split('/')[-1].split('.')[0], biomass))
    pd.DataFrame(biomass_lst, columns=['Name', 'Biomass']).to_csv(os.path.join(out_path, 'biomass.csv'), index=False)
    
if __name__ == '__main__':
    main()
