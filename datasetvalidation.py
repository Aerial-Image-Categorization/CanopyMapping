# %%
import os
import re
from tqdm.auto import tqdm
import time
import gc
import geopandas as gpd
import rasterio
from rasterio.crs import CRS

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


