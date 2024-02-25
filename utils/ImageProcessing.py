import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image, ImageDraw
from osgeo import gdal, ogr, osr
import rasterio
from rasterio.warp import transform_bounds
from pyproj import Transformer
from shapely.geometry import Point, box
import geopandas as gpd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/image_processing.log',
                    filemode='w')

'''
pipeline:
    - split
    - createPNG_Dataset / convert_TIFtoPNG
    - train / pred.
    - convert_PNGtoSHP
'''

def split_tif(tif_path, out_folder, tile_size=(250, 250)):
    """
    splitting .tif files 
    w/gdal
    pattern: out_folder/tile_tif_{i}_{j}.tif
    """
    logging.info('tif splitting started')
    ds = gdal.Open(tif_path)
    if ds is None:
        print(f"Could not open input TIF file: {tif_path}")
        return
    
    #get raster width and height
    width = ds.RasterXSize
    height = ds.RasterYSize

    #calc. the number of rows and columns
    num_cols = (width + tile_size[0] - 1) // tile_size[0]
    num_rows = (height + tile_size[1] - 1) // tile_size[1]
    
    total_tiles = num_rows * num_cols
    logging.info(f'tif splitting ended\n\tsplitted tif: {tif_path}\n\tto: {out_folder}\n\tsplit size: {num_rows}x{num_cols}\n\ttile size: {tile_size}')
    with tqdm(total=total_tiles, desc='splitting tifs') as pbar:
        for i in range(num_rows):
            for j in range(num_cols):
                #bbox coordinates
                xmin = j * tile_size[0]
                ymin = i * tile_size[1]
                xmax = min((j + 1) * tile_size[0], width)
                ymax = min((i + 1) * tile_size[1], height)
                
                #crop & save
                output_tif = os.path.join(out_folder, f'tile_tif_{i}_{j}.tif')
                gdal.Translate(output_tif, ds, srcWin=[xmin, ymin, xmax - xmin, ymax - ymin])
                pbar.update(1)
                
def split_shp(bigger_shapefile_path, tif_file_path, out_folder):
    """
    splitting .shp files
    w/ geopandas, rasterio
    pattern: out_folder/tile_shp_{i}_{j}.shp
    """
    
    #read the shp
    gdf_bigger = gpd.read_file(bigger_shapefile_path)
    
    #read the tif and get the bbox
    with rasterio.open(tif_file_path) as src:
        bbox = box(*src.bounds)
    
    #clip the bigger shapefile to the extent of the TIF file
    gdf_intersection = gdf_bigger[gdf_bigger.intersects(bbox)]
    
    splitted_name = os.path.basename(tif_file_path).split('_')
    out_path = os.path.join(out_folder,
                            f'tile_shp_{splitted_name[2]}_{os.path.splitext(splitted_name[3])[0]}.shp')
    gdf_intersection.to_file(out_path)
    
def split(tif_path, shp_path, output_folder, tile_size=(250, 250)): 
    """
    splitting .shp & .tif files 
    w/ split_tif() & split_shp() functions
    pattern: out_folder/tile_{ext.}_{i}_{j}.{ext.}
    """
    split_tif(tif_path,output_folder,tile_size)

    logging.info('shp splitting started')
    extension = '.tif'
    files = os.listdir(output_folder)
    total_files = len(files)
    count= 0
    with tqdm(total=total_files, desc='splitting shps') as pbar:
        for file in files:
            if file.endswith(extension):
                split_shp(shp_path, os.path.join(output_folder, file), output_folder)
                count=count+1
            pbar.update(1)
    logging.info(f'shp splitting ended\n\tsplitted shp: {shp_path}\n\tto: {output_folder}\n\tsplit size: {count}\n\ttile size: {tile_size}')

def convert_SHPtoPNG(tif_path, shp_path, png_path, tile_size=(250,250), point_size=1, bg_color='black', fg_color='white'):
    '''
    saves a .png image to the desired folder with the help of the .tif image 
    w/ geopandas, rasterio, pillow 
    Returns: the point list as a list of tuples 
    '''
    
    gdf = gpd.read_file(shp_path)
    
    target_epsg = 23700
    with rasterio.open(tif_path) as src:
        src_crs = src.crs
        # Transform the bounding box coordinates to the target EPSG code
        transformer = Transformer.from_crs(src_crs, f'EPSG:{target_epsg}', always_xy=True)
        bbox_transformed = transform_bounds(src_crs, f'EPSG:{target_epsg}', *src.bounds)

    #xmin, ymin, xmax, ymax = gdf.total_bounds
    #print(gdf.total_bounds)
    xmin, ymin, xmax, ymax = bbox_transformed

    img_width, img_height = tile_size
    
    #new blank image
    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    for geom in gdf.geometry:
        x = int((geom.x - xmin) / (xmax - xmin) * img_width)
        y = int((ymax - geom.y) / (ymax - ymin) * img_height)  # Invert
        draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill=fg_color, outline=fg_color)

    img.save(png_path)
    img.close()

    #extract points
    points = list(zip(gdf.geometry.x, gdf.geometry.y))
    return points
    
def getPoints_fromPNG(image_path):
    """
    Returns: the located points from the .png image 
    w/ opencv
    """
    img = cv2.imread(image_path)
    
    #grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #threshold
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    #find contours
    contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    white_dot_centers = []
    for contour in contours:
        #calculate the moments
        M = cv2.moments(contour)
        
        #calculate centroid coordinates
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            white_dot_centers.append((cX, cY))
    
    return white_dot_centers

def create_shapefile(output_shp, points):
    """
    creates a shapefile 
    w/ osgeo library
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    output_ds = driver.CreateDataSource(output_shp)
    
    spatial_ref = osr.SpatialReference()
    #spatial_ref.ImportFromEPSG(4326)  # WGS84
    spatial_ref.ImportFromEPSG(23700) # new
    
    #new layer
    output_layer = output_ds.CreateLayer("points", spatial_ref, ogr.wkbPoint)
    
    #define a field for the point ID
    id_field = ogr.FieldDefn("ID", ogr.OFTInteger)
    output_layer.CreateField(id_field)
    
    #create points and add them to the layer
    for i, (x, y) in enumerate(points):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(x,y)
        
        feature = ogr.Feature(output_layer.GetLayerDefn())
        feature.SetGeometry(point)
        feature.SetField("ID", i+1)
        output_layer.CreateFeature(feature)
        
        feature = None
    
    output_ds = None
    
def extend_image_shape(image_array: np.array, size=(250,250)):
    """
    extends the image shape to the desired shape 
    w/ numpy
    """
    new_image_array = np.zeros(size, dtype=image_array.dtype)
    #start_row = (new_image_array.shape[0] - image_array.shape[0]) // 2
    #start_col = (new_image_array.shape[1] - image_array.shape[1]) // 2
    start_col=0
    start_row=0

    end_row = start_row + image_array.shape[0]
    end_col = start_col + image_array.shape[1]
    new_image_array[start_row:end_row, start_col:end_col] = image_array

    #cv2.imwrite("output_image_filled2.png", new_image_array)
    return new_image_array

def scale_pixel_values(input_dataset):
    """
    scaling the pixel values 
    w/ numpy
    """
    
    band = input_dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    
    if np.any(data < 0):
        mask = (data > 200) & (data < 250)
        non_neg_min_val = np.min(data[mask])
        data[data < 0] = non_neg_min_val
        
    #scale pixel values to fit within the range of uint16
    min_val = np.min(data)
    max_val = np.max(data)
    
    scale = 65535
    scaled_data = ((data - min_val) / (max_val - min_val) * scale).astype(np.uint16)

    return scaled_data

def createPNG_Dataset(folder, out_folder, tile_size=(250,250),point_size=1,bg_color='black',fg_color='white'):
    """
    creates a dataset from a folder of splitted .tif & .shp files into another folder 
    w/ convert_SHPtoPNG(), gdal
    pattern: out_folder/tile_{ext.}_{i}_{j}.{ext.}
    """
    logging.info('creating dataset')
    files = os.listdir(folder)
    total_count= len(files)
    with tqdm(total=total_count, desc='creating png dataset (tif,shp -> png)') as pbar:
        for file in files:
            if file.endswith('.shp'):
                # .shp
                tif_path = os.path.join(folder,str(file).replace('_shp_','_tif_').replace('.shp','.tif'))
                shp_path = os.path.join(folder,file)
                out_path = os.path.join(out_folder, os.path.splitext(file)[0]+'.png')
                convert_SHPtoPNG(tif_path, shp_path, out_path, tile_size, point_size,bg_color,fg_color)
            
                # .tif
                tif_file = gdal.Open(tif_path)
                scaled = scale_pixel_values(tif_file)
                if scaled.shape != tile_size:
                    scaled = extend_image_shape(scaled,tile_size)
                cv2.imwrite(out_path.replace('shp','tif'), scaled)
            pbar.update(1)
    logging.info(f'dataset created\n\tfrom: {folder}\n\tto: {out_folder}\n\ttile_size: {tile_size}\n\tpoint size: {point_size}\n\tbackground color: {bg_color}\n\tforeground color: {fg_color}')

def convert_TIFtoPNG(folder, out_folder,tile_size=(250,250)):
    """
    converts .tif files to .png files from a folder into another.
    w/ gdal
    """
    logging.info('conversion from tif to png started')
    files = os.listdir(folder)
    total_count= len(files)
    with tqdm(total=total_count, desc='converting tifs to pngs') as pbar:
        for file in files:
            if file.endswith('.shp'):
                tif_path = os.path.join(folder,str(file).replace('_shp_','_tif_').replace('.shp','.tif'))
                out_path = os.path.join(out_folder, os.path.splitext(file)[0]+'.png')
            
                # .tif
                tif_file = gdal.Open(tif_path)
                scaled = scale_pixel_values(tif_file)
                if scaled.shape != tile_size:
                    scaled = extend_image_shape(scaled,tile_size)
                cv2.imwrite(out_path.replace('shp','tif'), scaled)
            pbar.update(1)
    logging.info(f'conversion finished\n\tfrom: {folder}\n\tto: {out_folder}\n\ttile_size: {tile_size}')
    
def convert_PNGtoSHP(folder,out_folder, result_folder):
    """
    converts .png files to .shp files from a folder into another folder 
    w/ getPoints_fromPNG(), rasterio, geopandas
    """
    logging.info('conversion from png to shp started')
    filenames = os.listdir(out_folder)
    total_count = len(filenames)
    count=0
    with tqdm(total=total_count, desc='converting pngs to shps') as pbar:
        for filename in filenames:
            if filename.split('_')[1]=='shp':
                image_coords = getPoints_fromPNG(os.path.join(out_folder,filename))
            
                tif_path = os.path.join(folder,filename.replace('_shp_','_tif_').replace('.png','.tif'))
                with rasterio.open(tif_path) as src:
                    transform = src.transform  #affine transformation object
                    crs = src.crs  #coordinate Reference System
                
                geo_coords = [transform * (x, y) for x, y in image_coords]
            
                #geo_coords -> Points (obj.)
                point_geoms = [Point(coord) for coord in geo_coords]
    
                #create a GeoDataFrame with the Points
                gdf = gpd.GeoDataFrame(geometry=point_geoms, crs=crs)  # Adjust CRS as needed
    
                #save the GeoDataFrame as a shapefile
                gdf.to_file(os.path.join(result_folder,'pred_'+filename.replace('.png','.shp')))
                count=count+1
            pbar.update(1)
    logging.info(f'conversion finished\n\tfrom: {out_folder}\n\tto: {result_folder}\n\tamount: {count}')
    
def get_tif_location(tif_path, target_epsg):
    """
    Shows: "Bounding box coordinates in EPSG:{target_epsg}: {bbox_transformed}"
    """
    with rasterio.open(tif_path) as src:
        src_crs = src.crs
        transformer = Transformer.from_crs(src_crs, f'EPSG:{target_epsg}', always_xy=True)
        bbox_transformed = transform_bounds(src_crs, f'EPSG:{target_epsg}', *src.bounds)
        print(f"Bounding box coordinates in EPSG:{target_epsg}: {bbox_transformed}")
        
def get_tif_informations(tif_path):
    """
    Shows:  "Affine transformation:", transform
            "Coordinate Reference System (CRS):", crs
    """
    with rasterio.open(tif_path) as src:
        transform = src.transform  #affine transformation object
        crs = src.crs  #coordinate Reference System
    print("Affine transformation:", transform)
    print("Coordinate Reference System (CRS):", crs)
    
def get_tif_size(tif_file_path):
    """
    Returns: width, height
    w/rasterio
    """
    with rasterio.open(tif_file_path) as src:
        width = src.width
        height = src.height
    return width, height

def show_image(path, divisor=10):
    """
    plots an image
    w/ matplotlib
    """
    image_array = plt.imread(path)
    height, width = image_array.shape[:2]
    fig, ax = plt.subplots(figsize=(width / divisor, height / divisor), tight_layout=True)
    ax.imshow(image_array)
    ax.axis('off')

    plt.show()