

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil
import cv2

def NDVI(image):
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)

    # Calculate NDVI
    ndvi = (green - red) / (green + red + 1e-5)  # Adding small value to avoid division by zero

    return cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalize NDVI to [0, 255] for better visibility

def interpolate(image, mask, size = (512, 512)):
    image_resized = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    
    return image_resized, mask_resized
    
def process(src_image_folder, src_mask_folder, dest_image_folder, dest_mask_folder, int_size = (512,512)):
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_mask_folder, exist_ok=True)

    mask_files = [f for f in os.listdir(src_mask_folder) if f.endswith(".png")]
    
    for mask_filename in tqdm(mask_files, desc="Processing images", leave=False):
        mask_path = os.path.join(src_mask_folder, mask_filename)
        image_path = os.path.join(src_image_folder, mask_filename.replace('_shp_','_tif_'))
        
        if not os.path.exists(image_path):
            print(f"Warning: Corresponding image for mask {mask_filename} not found. Skipping.")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, 0)
        
        image = NDVI(image)
        image, mask = interpolate(image, mask, int_size)
        
        cv2.imwrite(os.path.join(dest_image_folder, mask_filename.replace('_shp_','_tif_')), image)
        cv2.imwrite(os.path.join(dest_mask_folder, mask_filename), mask)
        
        
if __name__ == '__main__':
    dataset_path = '../data/2024-10-30-loc-dataset-1024'
    
    subdirs = [
        ('aug_train','man_aug_train'),
        ('aug_train_u10','man_aug_train_u10'),
        ('train','man_train'),
        ('val','man_val'),
        ('test','man_test')
    ]

    for subdir in tqdm(subdirs,desc='Processing folders'):
        subdir, new_subdir = subdir
        process(
            src_image_folder = os.path.join(dataset_path,subdir,'images'),
            src_mask_folder = os.path.join(dataset_path,subdir,'masks'),
            dest_image_folder = os.path.join(dataset_path,new_subdir,'images'),
            dest_mask_folder = os.path.join(dataset_path,new_subdir,'masks'),
            int_size = (512,512)
        )