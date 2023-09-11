import typing as tp
import warnings
import os
import albumentations as A
import torch, torch.nn as nn
import segmentation_models_pytorch as sm
import pandas as pd
import numpy as np
import cv2
from skimage import io
from tqdm.notebook import tqdm
from pathlib import Path
from copy import deepcopy
from albumentations.pytorch import ToTensorV2
from sys import argv

warnings.simplefilter("ignore")


def get_all_jpg_files(path: str) -> tp.Iterator[str]:
    img_names: list[str] = list()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            name = os.path.join(filename)
            if not name.endswith(('jpg', 'JPG')):
                continue
            yield os.path.join(dirname, name)
            
def get_crop(mask: np.ndarray) -> np.ndarray:
    mask_nonzero_x, mask_nonzero_y = np.nonzero(mask)
    d_cr = int(np.max((np.min(mask_nonzero_x), 0)))
    l_cr = int(np.max((np.min(mask_nonzero_y), 0)))
    u_cr = int(np.min((np.max(mask_nonzero_x), mask.shape[0] - 1)))
    r_cr = int(np.min((np.max(mask_nonzero_y), mask.shape[1] - 1)))
    return d_cr, u_cr, l_cr, r_cr

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

def count_spikelets(img: np.ndarray) -> int:
    if len(img.shape) > 2:
        img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = list()
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > 0 and area < img.shape[0] * img.shape[1] * 0.75:
            areas.append(area)
    mean_area = np.mean(areas)
    count = 0
    for area in areas:
        if area > mean_area * 0.25:
            count += 1
    return count

def get_central_points(mask: np.ndarray) -> tp.List[tp.Tuple[int, int]]:
    contours, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    central_points = list()
    areas = list()
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > 0 and area < mask.shape[0] * mask.shape[1] * 0.75:
            areas.append(area)
    mean_area = np.mean(areas)
    
    thr_contours = list()
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > mean_area * 0.25:
            thr_contours.append(contour) 

    for contour in thr_contours:
         try:
            moments = cv2.moments(contour)

            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

            central_points.append([cx, cy])
        except:
            pass

        central_points.append([cx, cy])

    return central_points

def get_mask(img_folder_path: str,
             out_img_folder_path: str,
             mask_path: str,
             model_path: str,
             device:str,
             ) -> None:
    """
    Create folder with masks.
    main code in https://github.com/kiteiru/nsu-diploma-wheat/blob/main/README.md
    Args:
        img_folder_path (str): folder with images.
        out_img_folder_path (str): folder for masks.
        mask_path (str): model for segmentation colorchecker path.
        model_path (str): model for detection path.
        device (str): device.
    """
    list_image_name = cropping_and_resizing(img_folder_path=img_folder_path, 
                                            mask_path=mask_path)
    data = {"Name": [],
            "Spikelets Num": []}
    model = sm.Unet(encoder_name='efficientnet-b4',
                    in_channels=3,
                    classes=1,
                    activation=None).to(device=device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    infer_transformations = A.Compose([A.Resize(384, 384), ToTensorV2()])
    
    for img, name in zip(*list_image_name):
        img = img.astype(np.float32)
        img /= 255
        augmentations = infer_transformations(image=img)
        img = augmentations["image"]
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float)

        with torch.no_grad():
            pred_mask = torch.sigmoid(model(img))
            pred_mask = (pred_mask > 0.5).float() * 255
            pred_mask = pred_mask[0][0].cpu().data.numpy()
            pred_mask = pred_mask.squeeze().astype(np.uint8)
        
        infer_directory = str(out_img_folder_path) + "/infer_masks"
        os.makedirs(infer_directory, exist_ok=True)
        io.imsave(os.path.join(infer_directory, "infer_" + name + ".jpg"), pred_mask)
        spikelets_num = count_spikelets(pred_mask)
        points = get_central_points(pred_mask)
        coordinates_directory = str(out_img_folder_path) + "/coordinates"
        os.makedirs(coordinates_directory, exist_ok=True)

        with open(Path(coordinates_directory + "/" + name + ".txt"), "w") as f:
            f.write("x;y;")
            f.write('\n')
            for point in points:
                f.write(f'{point[0]};{point[1]};')
                f.write('\n')


        data["Name"].append(name)
        data["Spikelets Num"].append(spikelets_num)

    df = pd.DataFrame(data, index=None)
    df.to_csv(out_img_folder_path + "/spikelets_count.csv", index=False)

def main():    
    _, in_path, out_path, mask_path, model_path = argv
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_mask(in_path, out_path, mask_path, model_path, device)

if __name__ == '__main__': 
    main()

