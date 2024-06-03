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
import shutil

import time
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/image_processing.log',
                    filemode='a')

# %%
def dropna_SHPs(folder,destination_folder='dropnas', pattern = r'^tile_shp.*\.shp$'):
    '''
    returns: 
        -   number of removed image-mask pairs
        -   runtime
    '''
    logging.info(f'⚙️ DROP SHPS with zero points started:\n\t- shps folder: {folder}')
    start_time = time.time()
    ad_extensions = ['.cpg','.dbf','.prj','.shx']

    destination_folder = os.path.join(folder,destination_folder)
    
    bin_list = []
    os.makedirs(os.path.join(destination_folder,'images'), exist_ok=True)
    os.makedirs(os.path.join(destination_folder,'masks'), exist_ok=True)
    
    filenames = os.listdir(folder)
    total_length = len(filenames)
    with tqdm(total=total_length) as pbar:
        for filename in filenames:
            if re.match(pattern, filename):
                shapefile = gpd.read_file(os.path.join(folder,filename))
                if len(shapefile.geometry)==0:
                    splitted = filename.split('.')[0].split('_')
                    bin_list.append((splitted[2],splitted[3]))
                    #os.remove(os.path.join(folder,filename))
                    #os.remove(os.path.join(folder.replace('shps','tifs'),filename.replace('shp','tif')))
                    #for ext in ad_extensions:
                    #    if os.path.exists(os.path.join(folder,filename[:-4]+ext)):
                    #        os.remove(os.path.join(folder,filename[:-4]+ext))
                    shutil.move(os.path.join(folder, filename), os.path.join(destination_folder,'masks', filename))
                    shutil.move(os.path.join(folder.replace('shps', 'tifs'), filename.replace('shp', 'tif')),
                        os.path.join(destination_folder,'images', filename.replace('shp', 'tif')))
                    for ext in ad_extensions:
                        if os.path.exists(os.path.join(folder, filename[:-4] + ext)):
                            shutil.move(os.path.join(folder, filename[:-4] + ext), os.path.join(destination_folder,'masks', filename[:-4] + ext))
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

import rasterio
from rasterio.enums import Resampling

def ResampleTIF(upscale_factor, tif_path, resampled_tif_path):
    tif_image = rasterio.open(tif_path)
    # resample
    data = tif_image.read(
        out_shape=(
            tif_image.count,
            int(tif_image.height * upscale_factor),
            int(tif_image.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )
    
    #pixel_width, pixel_height = tif_image.res
    #print(f"Transformed pixel size: {pixel_width} x {pixel_height}")
    
    # scale image transform
    new_transform = tif_image.transform * tif_image.transform.scale(
        (tif_image.width / data.shape[-1]),
        (tif_image.height / data.shape[-2])
    )
    #pixel_width = transform.a
    #pixel_height = -transform.e
    
    #print(f"Transformed pixel size: {pixel_width} x {pixel_height}")
    #pixel_area = pixel_width * pixel_height
    #print(tif_image.transform,data.shape,transform)

    profile = tif_image.profile
    profile.update({
        'height': data.shape[1],
        'width': data.shape[2],
        'transform': new_transform
    })
    rasterio.open(resampled_tif_path, 'w', **profile).write(data)

def check_resolution(desired_res, tif_path, round_accuracy):
    '''
    Projected Coordinate System (PCS): the units are typically in meters or feet.
    Geographic Coordinate System (GCS): the units are typically in degrees.
    The CRS (Coordinate Reference System) EPSG:23700 corresponds to the "HD72 / EOV" projection.
    This is a projected coordinate system used in Hungary, and the units for this CRS are meters.
    '''
    tif_image = rasterio.open(tif_path)
    p_width, p_height = tif_image.res
    p_width_normalized = round(p_width,round_accuracy)
    p_height_normalized = round(p_height,round_accuracy)
    #print(p_width_normalized,p_height_normalized)
    
    actual_res = p_width_normalized
    ratio = actual_res/desired_res
    return ratio