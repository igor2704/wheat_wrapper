import numpy as np
import os
import typing as tp
import cv2
import warnings
from sys import argv
from pathlib import Path
from copy import deepcopy


def get_all_jpg_files(path: str) -> tp.Iterator[str]:
    img_names: list[str] = list()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith(('jpg', 'JPG')):
                continue
            yield os.path.join(dirname, name)

def get_crop(mask: np.ndarray, eps: int = 100) -> np.ndarray:
    mask_nonzero_x, mask_nonzero_y = np.nonzero(mask)
    d_cr = int(np.max((np.min(mask_nonzero_x) - eps, 0)))
    l_cr = int(np.max((np.min(mask_nonzero_y) - eps, 0)))
    u_cr = int(np.min((np.max(mask_nonzero_x) + eps, mask.shape[0] - 1)))
    r_cr = int(np.min((np.max(mask_nonzero_y) + eps, mask.shape[1] - 1)))
    return d_cr, u_cr, l_cr, r_cr

# def get_crops(img_folder_path: str,
#               mask_path: str) -> tp.Tuple[tp.List[np.ndarray], tp.List[str]]:
#     """
#     Args:
#         img_folder_path (str): folder with images.
#         mask_path (str): wheet mask path.
#     Returns:
#         list[np.ndarray]: cropping and resizing images.
#         list[str]: image names.
#     """
#     img_lst = []
#     name_lst = []
#     for path in get_all_jpg_files(img_folder_path):
#         name = str(path).split('/')[-1].split('.')[0]
#         print(os.path.join(str(mask_path), name))
#         img = cv2.imread(str(path))
#         mask = cv2.imread(os.path.join(str(mask_path), name + '.png'))[:, :, 1]
        
#         d_cr, u_cr, l_cr, r_cr = get_crop(mask)
#         img = img[d_cr:u_cr, l_cr:r_cr, :]
#         mask = mask[d_cr:u_cr, l_cr:r_cr]
#         mask = np.where(mask > 0, 1, 0)
        
#         for i in range(3):
#             img[:, :, i] = img[:, :, i] * mask 
            
#         contours, _ = cv2.findContours(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
#                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
#         img *= cv2.fillPoly(np.zeros(img.shape, dtype='uint8'), pts=[max_contour], color=(1,1,1))
        
#         d_cr, u_cr, l_cr, r_cr = get_crop(np.sum([img[:, :, i] for i in range(3)], axis=0))
#         img = img[d_cr:u_cr, l_cr:r_cr, :]

        
#         img_lst.append(img)
#         name_lst.append(name)
#     return img_lst, name_lst

def cropping_and_resizing(img_folder_path: str,
                          mask_path: str,
                          size: tp.Tuple[int, int] = (384, 384)) -> tp.Tuple[tp.List[np.ndarray], tp.List[str]]:
    """
    Args:
        img_folder_path (str): folder with images.
        mask_path (str): wheet mask path.
        size (tuple[int, int], optional): new size for images. Defaults to (384, 384).
    
    Returns:
        list[np.ndarray]: cropping and resizing images.
        list[str]: image names.
    """
    img_lst = []
    name_lst = []
    for path in get_all_jpg_files(img_folder_path):
        name = str(path).split('/')[-1].split('.')[0]
        print(os.path.join(str(mask_path), name))
        img = cv2.imread(str(path))
        mask = cv2.imread(os.path.join(str(mask_path), name + '.png'))
        mask = mask[:, :, 1]
        
        d_cr, u_cr, l_cr, r_cr = get_crop(mask)
        img = img[d_cr:u_cr, l_cr:r_cr, :]
        mask = mask[d_cr:u_cr, l_cr:r_cr]
        mask = np.where(mask > 0, 1, 0)
        
        new_img = deepcopy(img)
        for i in range(3):
            new_img[:, :, i] = img[:, :, i] * mask 
            
        contours, _ = cv2.findContours(cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = contours[np.argmax([cv2.contourArea(contour) for contour in contours])]
        max_contour_mask = cv2.fillPoly(np.zeros(img.shape, dtype='uint8'), pts=[max_contour], color=(1,1,1))
        
        d_cr, u_cr, l_cr, r_cr = get_crop(np.sum([max_contour_mask[:, :, i] for i in range(3)], axis=0))
        img = img[d_cr:u_cr, l_cr:r_cr, :]

        # img = cv2.resize(img, size)
        
        img_lst.append(img)
        name_lst.append(name)
    return img_lst, name_lst

def main():
    _, path_in, mask_path, path_out_dir = argv
    img_name_lst = cropping_and_resizing(path_in, mask_path)
    for img, name in zip(*img_name_lst):
        cv2.imwrite(os.path.join(path_out_dir, 'crop_' + name + '.png'), cv2.resize(img, (150, 150)))
        
if __name__ == '__main__':
    main()
    
    