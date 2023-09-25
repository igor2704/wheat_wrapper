from training_config import augmentations
from config import PROJECT_DIR, DATA_DIR, PATTERNS

import numpy as np
import pandas as pd
import re

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

import cv2

# to retrive one feature and lable one time
class SpikeDataset(Dataset):
    def __init__(self, annotations_file, group='train', verbose=False):
        def data_target_split(split_filename, group):
            df = pd.read_csv(f'{PROJECT_DIR}/splits/{split_filename}', sep=';')
            df = df[df['Group'] == group]
            if df.empty:
                raise ValueError("Argument group is incorrect. It should be in ['train', 'valid', 'test']")
            return df['Path'].to_numpy(), df['Label'].to_numpy()
                                  
        self.annotations_file = annotations_file
        self.group = group
        self.data, self.targets = data_target_split(annotations_file, group)
        self.verbose = verbose
        
    def get_data(self) -> (np.array, np.array):
        return self.data, self.targets
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        # read image and get label using path to image    
        def get_crop(img_path):
            ploidness = re.search(PATTERNS['ploid'], img_path).group()
            species = re.search(PATTERNS['species'], img_path).group()
            subspecies = re.search(PATTERNS['subspecies'], img_path).group()
            
            img_type = 'pin'
            img_name = re.search(PATTERNS['pin'], img_path)
            if not img_name:
                img_type = 'table'
                img_name = re.search(PATTERNS['table'], img_path)
            if not img_name:
                img_type=None
                return
            img_name = img_name.group()
            
            # path to image in folder withsegemented boudning boxes with wheat spikes
            inp_dir = f'{PROJECT_DIR}/crops_without_spine/{ploidness}/{species}/{img_type}/{img_name}'
            inp_dir = inp_dir.replace('.jpg', '.png').replace('.JPG', '.png')
            
            inp_img = cv2.imread(inp_dir)
            preprocessed_img = None
            
            # different augmentations for images according to type of data
            if self.group == 'train':
                preprocessed_img = augmentations['train_transforms'](image=inp_img)['image']
                
            elif self.group == 'test' or self.group == 'valid':
                preprocessed_img = augmentations['inference_transforms'](image=inp_img)['image']
            else:
                raise ValueError('''Error occured. Check your data_type and change to one of these:
                      \"train\", \"test\"''')
            
            # np array in format requred by CNN architectures (c, w, h)
            preprocessed_img = np.moveaxis(preprocessed_img, -1, 0)
            if self.verbose == True:
                return preprocessed_img, {'ploidness': ploidness,
                                          'species': species, 
                                          'subspecies': subspecies, 
                                          'img_type': img_type,
                                          'img_path': img_path}
            return preprocessed_img
        
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        if self.verbose == True:
            final_img, kwargs = get_crop(current_sample)
            return final_img, current_target, kwargs
        else:
            final_img = get_crop(current_sample)
            return final_img, current_target
        
    def images(self):
        num_images = self.__len__()
        for i in range(num_images):
            img = None
            if self.verbose:
                img, _, _ = self.__getitem__(i)
            else:
                img, _ = self.__getitem__(i)
            yield img
            
    def labels(self):
        num_labels = self.__len__()
        for i in range(num_labels):
            img = None
            if self.verbose:
                _, label, _ = self.__getitem__(i)
            else:
                _, label = self.__getitem__(i)
            yield label
        
        
class HoldoutDataset(SpikeDataset):
    def __getitem__(self, idx): 
        # read image and get label using path to image    
        def get_crop(img_path):
            inp_img = cv2.imread(img_path)
            preprocessed_img = None
            
            # we assume that holdout data if only for test!
            if self.group == 'test':
                preprocessed_img = augmentations['inference_transforms'](image=inp_img)['image']
            else:
                raise ValueError('''Error occured. Check your data_type and change to "test". We assume, that holdout
                                 data is only for inference!''')
            
            # np array in format requred by CNN architectures (c, w, h)
            preprocessed_img = np.moveaxis(preprocessed_img, -1, 0)
            return preprocessed_img
        
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        final_img = get_crop(current_sample)
        if self.verbose:
            return final_img, current_target, {'img_path': current_sample}
        return final_img, current_target

        
def balance_dataset(dataset : SpikeDataset) -> WeightedRandomSampler:
    data, target = dataset.get_data()

    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    return WeightedRandomSampler(samples_weight, len(samples_weight))
