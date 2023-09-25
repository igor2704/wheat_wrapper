from utils import ReadOnlyDict, ReadOnlyList

import albumentations as A
import cv2

# shapes of crops
training_params = {'train_crop_size': 512,
                   'inference_crop_size': 480,
                   'epochs': 300,
                   'train_batch_size': 8,
                   'inference_batch_size': 1,
                   'lr': 1e-6,
                   'wd': 1e-7,
                   'num_workers': 4,
                   'balanced': True,
                   'loss': 'BCELoss',
                   'optimizer': 'Adam',
                   'scheduler' : 'ReduceLROnPlateau'
                  }

# used based augmentations (correspond to the best training strategy with average crops)
augmentations = {'train_transforms' : A.Compose([A.RandomScale((0.8, 1.7), p=1),
                                                    A.PadIfNeeded(training_params['train_crop_size'], 
                                                                  training_params['train_crop_size'], 
                                                                    border_mode=cv2.BORDER_CONSTANT),
                                                    A.CenterCrop(training_params['train_crop_size'], 
                                                                 training_params['train_crop_size'], p=1),
                                                    A.Flip(p=0.4),
                                                    A.Normalize(mean=(0.485, 0.456, 0.406), 
                                                                    std=(0.229, 0.224, 0.225), p=1)
                                                    ]),
                 'inference_transforms' : A.Compose([
                                                    A.PadIfNeeded(training_params['inference_crop_size'], 
                                                                  training_params['inference_crop_size'], 
                                                                    border_mode=cv2.BORDER_CONSTANT),
                                                    A.CenterCrop(training_params['inference_crop_size'], 
                                                                 training_params['inference_crop_size'], p=1),
                                                    A.Normalize(mean=(0.485, 0.456, 0.406), 
                                                                    std=(0.229, 0.224, 0.225), p=1)
                                                    ])
                }