import sys
#sys.path.append('../')

import os
os.environ['GTIFF_SRS_SOURCE'] = 'EPSG'

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/create-dataset.log',#'../logs/image_processing.log',
                    filemode='a')
import argparse

from utils.imageprocessing import split, createPNG_Dataset, remove_empty_images
from utils.datasetvalidation import set_valid_CRS, dropna_PNGs

from utils.dataloading import ImageNameDataset
from utils.traintestsplit import middle_split
import torch
from torch.utils.data import DataLoader, random_split

import shutil
import time
import pandas as pd

from utils.datasetvalidation import check_images_size
from utils.augmentation import rotate_train_pairs

'''
pipeline:
    - split
    - drop outlier pairs
    - set_CRS
    - createPNG_Dataset
    - train-test split
    - train-validation split
    - drop empty pairs from train & validation set
    - augmentation on train set
'''

if __name__ == '__main__':
    #set logging
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/create-dataset.log',
                    filemode='w')
    
    dataset_folder = '../data/2024-09-29-dataset'
    size = (200,200)
    tif_path = '../data/raw/2023-02-23_Bakonyszucs_actual.tif'
    shp_path = '../data/raw/Fa_pontok.shp'#'../data/test_data/conc_biomed_full.shp'
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
    
    #drop empty tifs
    logging.info(f"⚙️ Drop empty TIFs:\n\t- tifs path: {os.path.join(dataset_folder,'all','original','tifs')}")
    empty_tiles_count, elapsed_time = remove_empty_images(os.path.join(dataset_folder,'all','original','tifs'))
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Drop empties finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- empty tiles count: {empty_tiles_count}')
    
    
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

    #train-valid.-test split
    start_time = time.time()
    
    #train - test
    dataset = ImageNameDataset(
        os.path.join(dataset_folder,'all','formatted'),
        sort = True
    )
    images_train, images_test, masks_train, masks_test = middle_split(
        [sample['image'] for sample in dataset],
        [sample['mask'] for sample in dataset],
        round(train_size+valid_size,6)
    )

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
    valid_size_orig = valid_size
    valid_size = valid_size / (train_size+valid_size)
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

    #move
    for path in images_valid:
        shutil.move(path,os.path.join(dataset_folder, 'val','images'))
    for path in masks_valid:
        shutil.move(path,os.path.join(dataset_folder, 'val','masks'))

    valid_size = valid_size_orig
    hours, remainder = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    df = pd.DataFrame({
        'Set': ['Train', 'Validation', 'Test', 'SUM'],
        'Size': [round(train_size, 5), round(valid_size, 5), round(1 - train_size - valid_size, 5),round(train_size,5)+ round(valid_size, 5)+ round(1 - train_size - valid_size, 5)],
        'Amount': [len(images_train), len(images_valid), len(images_test),len(images_train)+ len(images_valid)+ len(images_test)]
    })
    print('Train-Validation-Test Splitage finished')
    logging.info(f'🏁 Train-Validation-Test Splitage finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n{df.to_string(index=False)}')

    #drop empty pairs from train & validation sets
    train_bin_list, _ = dropna_PNGs(os.path.join(dataset_folder, 'train'))
    val_bin_list, _ = dropna_PNGs(os.path.join(dataset_folder, 'val'))

    #check images size
    invalid_train_imgs, invalid_val_imgs = check_images_size(dataset_folder)
    for path in invalid_train_imgs:
        os.remove(path)
    #for path in invalid_val_imgs:
    #    os.remove(path)

    #augmentation
    os.makedirs(os.path.join(dataset_folder, 'aug_train','images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'aug_train','masks'), exist_ok=True)

    train_set = ImageNameDataset(os.path.join(dataset_folder, 'aug_train'), sort = True)
    
    train_images = [sample['image'] for sample in train_set]
    train_masks = [sample['mask'] for sample in train_set]

    for path in train_images:
        shutil.copy(path,train_images_folder)
    for path in train_masks:
        shutil.copy(path,train_masks_folder)

    
    rotate_train_pairs(
        train_images_folder = os.path.join(dataset_folder, 'aug_train','images'),
        train_masks_folder = os.path.join(dataset_folder, 'aug_train','masks')
    )    
