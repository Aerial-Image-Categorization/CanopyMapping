import sys
sys.path.append('../')

import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/create-dataset.log',#'../logs/image_processing.log',
                    filemode='a')
import argparse

from utils.imageprocessing import split, createPNG_Dataset
from utils.datasetvalidation import set_valid_CRS

from utils.dataloading import ImageNameDataset
from utils.traintestsplit import middle_split
import torch
from torch.utils.data import DataLoader, random_split

import shutil

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
    #set logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/create-dataset.log',
                    filemode='w')
    
    dataset_folder = '../data/2024-06-14-dataset'
    size = (200,200)
    tif_path = '../data/test_data/orig_test2.tif'
    shp_path = '../data/all_data/Fa_pontok.shp'#'../data/test_data/conc_biomed_full.shp'
    train_size = 0.8
    valid_size = 0.1
    batch_size = 1

    os.makedirs(os.path.join(dataset_folder, 'train','images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'train','masks'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'val','images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'val','masks'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'test','images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'test','masks'), exist_ok=True)
    
    os.makedirs(os.path.join(dataset_folder,'all','original'),exist_ok=True) #folder
    os.makedirs(os.path.join(dataset_folder,'all','formatted'),exist_ok=True) #out_folder

    #split into tiles
    split(
        tif_path,
        shp_path,
        os.path.join(dataset_folder,'all','original'),
        size
    )
    
    #set epsg
    set_valid_CRS(
        os.path.join(dataset_folder,'all','original','shps'),
        desired_crs_epsg=23700
    )
    
    #format
    if createPNG_Dataset(
        os.path.join(dataset_folder,'all','original'),
        os.path.join(dataset_folder,'all','formatted'),
        size,
        point_size=28,
        grayscale=False
    ) == False:
        raise Exception('ERROR: check the logs')

    #train - test
    dataset = ImageNameDataset(
        os.path.join(dataset_folder,'all','formatted'),
        sort = True
    )
    images_train, images_test, masks_train, masks_test = middle_split(
        [sample['image'] for sample in dataset],
        [sample['mask'] for sample in dataset],
        train_size
    )

    print(len([sample['image'] for sample in dataset]),len([sample['mask'] for sample in dataset]))
    print(len(images_train),len(images_test),len(masks_train),len(masks_test))

    #move
    for path in images_train:
        shutil.copy(path,os.path.join(dataset_folder, 'train','images'))
    for path in masks_train:
        shutil.copy(path,os.path.join(dataset_folder, 'train','masks'))
    for path in images_test:
        shutil.copy(path,os.path.join(dataset_folder, 'test','images'))
    for path in masks_test:
        shutil.copy(path,os.path.join(dataset_folder, 'test','masks'))

    #train - validation
    valid_size = valid_size / train_size
    dataset = ImageNameDataset(
        os.path.join(dataset_folder, 'train'),
        sort = True
    )
    train_dataset, valid_dataset = random_split(
        [sample for sample in dataset],
        [1-valid_size,valid_size],
        generator=torch.Generator().manual_seed(42)
    )
    images_train = [sample['image'] for sample in train_dataset]
    masks_train = [sample['mask'] for sample in train_dataset]
    images_valid = [sample['image'] for sample in valid_dataset]
    masks_valid = [sample['mask'] for sample in valid_dataset]

    print(len(images_train),len(images_valid),len(masks_train),len(masks_valid))

    #move
    for path in images_valid:
        shutil.copy(path,os.path.join(dataset_folder, 'val','images'))
    for path in masks_valid:
        shutil.copy(path,os.path.join(dataset_folder, 'val','masks'))





