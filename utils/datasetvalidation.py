# %%
import os
import re
from tqdm.auto import tqdm
import time
import gc
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
import cv2

import time
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/image_processing.log',
                    filemode='a')

# %%
def dropna_SHPs(folder, pattern = r'^tile_shp.*\.shp$'):
    '''
    returns: 
        -   number of removed image-mask pairs
        -   runtime
    '''
    logging.info(f'⚙️ DROP SHPS with zero points started:\n\t- shps folder: {folder}')
    start_time = time.time()
    ad_extensions = ['.cpg','.dbf','.prj','.shx']
    
    bin_list = []
    
    filenames = os.listdir(folder)
    total_length = len(filenames)
    with tqdm(total=total_length) as pbar:
        for filename in filenames:
            if re.match(pattern, filename):
                shapefile = gpd.read_file(os.path.join(folder,filename))
                if len(shapefile.geometry)==0:
                    splitted = filename.split('.')[0].split('_')
                    bin_list.append((splitted[2],splitted[3]))
                    os.remove(os.path.join(folder,filename))
                    os.remove(os.path.join(folder,filename.replace('shp','tif')))
                    for ext in ad_extensions:
                        if os.path.exists(os.path.join(folder,filename[:-4]+ext)):
                            os.remove(os.path.join(folder,filename[:-4]+ext))
            pbar.update(1)
            
    gc.collect()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ DROP SHPS ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- drop count: {len(bin_list)}')
    return bin_list, elapsed_time

# %%
def set_valid_CRS(folder, pattern = r'^tile_tif.*\.tif$', desired_crs_epsg=23700):
    logging.info(f'⚙️ SET VALID CRS started:\n\t- shps folder: {folder}\n\t- desired crs epsg: {desired_crs_epsg}')
    start_time = time.time()
    out_list = []
    filenames = os.listdir(folder)
    total_length = len(filenames)
    with tqdm(total=total_length) as pbar:
        for filename in filenames:
            if re.match(pattern, filename):
                with rasterio.open(os.path.join(folder,filename)) as src:
                    if src.crs == None:
                        src.crs = CRS.from_epsg(desired_crs_epsg)
                        print(f'{filename} set to EPSG:{src.crs}')
                        splitted = filename.split('.')[0].split('_')
                        out_list.append((splitted[2],splitted[3]))
            pbar.update(1)
    gc.collect()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ SET VALID CRS ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- set count: {len(out_list)}')
    return out_list, elapsed_time

def check_images_size(dataset_path):
    '''
    dataset structure:
       - dataset
           - train
               - images
               - masks
           - test
               - images
               - masks            
    '''
    train_images_path = os.path.join(dataset_path, 'train','images')
    test_images_path = os.path.join(dataset_path, 'test','images')
    
    for image_name in os.listdir(train_images_path):
        image = cv2.imread(os.path.join(train_images_path,image_name))
        image_mask = cv2.imread(os.path.join(train_images_path.replace('images','masks'),image_name.replace('_tif_','_shp_')))
        try:
            if image.shape != image_mask.shape:
                print(f'train: {image_name} \n\timage shape: {image.shape}\n\tmask shape: {image_mask.shape}')
        except:
            print(f'error: {image_name}')
            
    for image_name in os.listdir(test_images_path):
        image = cv2.imread(os.path.join(test_images_path,image_name))
        image_mask = cv2.imread(os.path.join(test_images_path.replace('images','masks'),image_name.replace('_tif_','_shp_')))
        try:
            if image.shape != image_mask.shape:
                print(f'test: {image_name} \n\timage shape: {image.shape}\n\tmask shape: {image_mask.shape}')
        except:
            print(f'error: {image_name}')
