

import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil
import cv2

def desaturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])  #adjust needed
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    #desaturate areas outside of the green mask
    hsv[:, :, 1] = cv2.bitwise_and(hsv[:, :, 1], green_mask)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def shadow_boosting(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if len(image.shape) == 3:
        return cv2.merge([clahe.apply(channel) for channel in cv2.split(image)])
    else:
        return clahe.apply(image) #grayscale
        
        

def NDVI(image):
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)

    # Calculate NDVI
    ndvi = (green - red) / (green + red + 1e-5)  # Adding small value to avoid division by zero

    return cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Normalize NDVI to [0, 255] for better visibility

def VARI(image):
    blue = image[:, :, 0].astype(float)
    green = image[:, :, 1].astype(float)
    red = image[:, :, 2].astype(float)

    # Calculate NDVI
    ndvi = (green - red) / (green + red - blue + 1e-5)  # Adding small value to avoid division by zero

    return cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

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
        
        
        ### image man.
        #image = VARI(image)
        image = NDVI(image)
        #image = desaturation(image)
        #image = shadow_boosting(image)
        ###
        
        ### resize (interpolate)
        #image, mask = interpolate(image, mask, int_size)
        ###
        
        cv2.imwrite(os.path.join(dest_image_folder, mask_filename.replace('_shp_','_tif_')), image)
        cv2.imwrite(os.path.join(dest_mask_folder, mask_filename), mask)
        
        
if __name__ == '__main__':
    #dataset_path = '../data/2024-10-30-loc-dataset-1024'
    #dataset_path = '../data/2024-11-13-seg-dataset-1024'
    dataset_path = '../data/2024-11-13-seg-dataset-2048'
    
    subdirs = [
        #('u_aug_train','u_aug_train_vari'),
        #('u_val','u_val_vari'),
        #('u_test','u_test_vari')
        #('train','u_train')
        #('aug_train','u_aug_train'),
        #('val','u_val'),
        #('test','u_test')
        #('aug_train','ndvi_aug_train'),
        #('val','ndvi_val'),
        #('test','ndvi_test')
        #('aug_train','u_aug_train'),
        #('train','u_train')
        #('train','man_train'),
        #('val','u_val'),
        #('test','u_test')
        #('u_aug_train_f10','prep_u_aug_train_f10'),
        #('u_val_f10','prep_u_val_f10'),
        #('u_test_f10','prep_u_test_f10')
        #('u_aug_train_u10','sb_u_aug_train_u10'),
        #('u_val','sb_u_val'),
        #('u_test','sb_u_test')
        #
        #("u_aug_train_u10","clahe_u_aug_train_u10"),
        #("u_val","clahe_u_val"),
        #("u_test","clahe_u_test")
        #
        ('u_aug_train','u_aug_train_clahe'),
        ('u_val','u_val_clahe'),
        ('u_test','u_test_clahe')
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