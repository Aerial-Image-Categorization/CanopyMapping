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

# %%
def dropna_SHPs(folder, pattern = r'^tile_shp.*\.shp$'):
    '''
    returns: 
        -   runtime
        -   number of removed image-mask pairs
    '''
    
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
    return bin_list

# %%
def set_valid_CRS(folder, pattern = r'^tile_tif.*\.tif$', desired_crs_epsg=23700):
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
    return out_list

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
        if image.shape != image_mask.shape:
            print(f'train: {image_name} \n\timage shape: {image.shape}\n\tmask shape: {image_mask.shape}')
            
    for image_name in os.listdir(test_images_path):
        image = cv2.imread(os.path.join(test_images_path,image_name))
        image_mask = cv2.imread(os.path.join(test_images_path.replace('images','masks'),image_name.replace('_tif_','_shp_')))
        if image.shape != image_mask.shape:
            print(f'test: {image_name} \n\timage shape: {image.shape}\n\tmask shape: {image_mask.shape}')
    
