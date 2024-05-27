import sys
sys.path.append('../')

import os
import argparse

from utils.imageprocessing import split, createPNG_Dataset
from utils.datasetvalidation import set_valid_CRS

from utils.dataloading import ImageNameDataset
from utils.traintestsplit import middle_split
from torch.utils.data import DataLoader, random_split

'''
pipeline:
    - split
    - set_CRS
    - createPNG_Dataset
    - train-test split
    - train-validation split
    - drop outlier pairs from any set
    - drop empty pairs from train & validation set
    - augmentation on train set
'''

if __name__ == '__main__':
    folder = '../data/2024-05-20-dataset/standard/original'
    out_folder = '../data/2024-05-20-dataset/standard/formatted'
    size = (200,200)

    #os.makedirs(folder, exist_ok=True)
    #os.makedirs(out_folder, exist_ok=True)

    ##split(
    #    '../data/all_data/2023-02-23_Bakonyszucs_actual.tif',
    #    '../data/all_data/Fa_pontok.shp',
    #    folder,
    #    size
    #)

    #set_valid_CRS(os.path.join(folder,'shps'), desired_crs_epsg=23700)
    #createPNG_Dataset(folder,out_folder,size, point_size=28, grayscale=False)
    

    dataset_folder = '../data/2024-04-21-dataset/standard/formatted'
    train_size = 0.90
    batch_size = 1

    dataset = ImageNameDataset(dataset_folder, sort = True)

    images_train, images_test, masks_train, masks_test = middle_split(
        [sample['image'] for sample in dataset],
        [sample['mask'] for sample in dataset],
        train_size
    )



