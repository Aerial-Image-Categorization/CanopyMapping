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
from shapely.geometry import box, Point, MultiPoint

from tqdm.auto import tqdm
import time

import logging
#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s - %(levelname)s - %(message)s',
#                    filename='../logs/test.log',#'../logs/image_processing.log',
#                    filemode='a')

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
    return: num_cols, num_rows, elapsed_time
    """
    start_time = time.time()
    
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
    
    with tqdm(total=total_tiles, desc='splitting TIF') as pbar:
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
                
    elapsed_time = time.time() - start_time
    return num_cols, num_rows, elapsed_time
    
def split_shp(shp_path, tifs_folder, out_folder):
    """
    splitting .shp files
    w/ geopandas, rasterio
    pattern: out_folder/tile_shp_{i}_{j}.shp
    return: count, elapsed_time
    """
    start_time = time.time()
    
    extension = '.tif'
    files = os.listdir(tifs_folder)
    total_files = len(files)
    count= 0
    with tqdm(total=total_files, desc='splitting shps') as pbar:
        for file in files:
            if file.endswith(extension):
                tif_file_path = os.path.join(tifs_folder, file)
                
                #read the shp
                gdf_bigger = gpd.read_file(shp_path)
                #read the tif and get the bbox
                with rasterio.open(tif_file_path) as src:
                    bbox = box(*src.bounds)
                #clip the bigger shapefile to the extent of the TIF file
                gdf_intersection = gdf_bigger[gdf_bigger.intersects(bbox)]
                splitted_name = os.path.basename(tif_file_path).split('_')
                out_path = os.path.join(out_folder, f'tile_shp_{splitted_name[2]}_{os.path.splitext(splitted_name[3])[0]}.shp')
                gdf_intersection.to_file(out_path)

                
                count=count+1
            pbar.update(1)
            
    elapsed_time = time.time() - start_time
    return count, elapsed_time

def split_shp_seg(shp_path, tifs_folder, out_folder):
    """
    splitting .shp files
    w/ geopandas, rasterio
    pattern: out_folder/tile_shp_{i}_{j}.shp
    return: count, elapsed_time
    """
    start_time = time.time()
    
    extension = '.tif'
    files = os.listdir(tifs_folder)
    total_files = len(files)
    count= 0
    with tqdm(total=total_files, desc='splitting shps') as pbar:
        for file in files:
            if file.endswith(extension):
                tif_file_path = os.path.join(tifs_folder, file)
                
                with rasterio.open(tif_file_path) as src:
                    bbox = box(*src.bounds)
                tile_gdf = gpd.read_file(shp_path, mask=bbox)

                splitted_name = os.path.basename(tif_file_path).split('_')
                out_path = os.path.join(out_folder, f'tile_shp_{splitted_name[2]}_{os.path.splitext(splitted_name[3])[0]}.shp')
                tile_gdf.to_file(out_path)

                
                count=count+1




            pbar.update(1)

        elapsed_time = time.time() - start_time
        return count, elapsed_time

def split(tif_path, shp_path, output_folder, tile_size=(250, 250)): 
    """
    splitting .shp & .tif files 
    w/ split_tif() & split_shp() functions
    pattern: out_folder/{tifs / shps}/tile_{ext.}_{i}_{j}.{ext.}
    """

    tifs_path_folder = os.path.join(output_folder, 'tifs')
    shps_path_folder = os.path.join(output_folder, 'shps')
    
    if not os.path.exists(tifs_path_folder):
        os.makedirs(tifs_path_folder)
    if not os.path.exists(shps_path_folder):
        os.makedirs(shps_path_folder)
    
    logging.info(f'⚙️ TIF splitage started:\n\t- splitted shp: {tif_path}\n\t- to: {tifs_path_folder}')
    
    num_cols, num_rows, elapsed_time = split_tif(tif_path,tifs_path_folder,tile_size)

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ TIF splitting ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {num_rows}x{num_cols}\n\t- tile size: {tile_size}')

    logging.info(f'⚙️ SHP splitage started:\n\t- splitted shp: {shp_path}\n\t- to: {shps_path_folder}')
    
    count, elapsed_time = split_shp(shp_path, tifs_path_folder, shps_path_folder)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ SHP splitting ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {count}')
    if num_rows*num_cols != count:
        logging.warning(f'⚠️ TIFs - SHPs count mismatch: {num_rows*num_cols} - {count}')

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
        #transformer = Transformer.from_crs(src_crs, f'EPSG:{target_epsg}', always_xy=True)
        bbox_transformed = transform_bounds(src_crs, f'EPSG:{target_epsg}', *src.bounds)

    #xmin, ymin, xmax, ymax = gdf.total_bounds
    #print(gdf.total_bounds)
    xmin, ymin, xmax, ymax = bbox_transformed

    img_width, img_height = tile_size
    
    #new blank image
    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    points_x = []
    points_y = []

    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            points_x.append(point.x)
            points_y.append(point.y)
            x = int((geom.x - xmin) / (xmax - xmin) * img_width)
            y = int((ymax - geom.y) / (ymax - ymin) * img_height)  # Invert
            draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill=fg_color, outline=fg_color)
        elif geom.geom_type == 'MultiPoint':
            points = gpd.GeoSeries(geom).explode()
            for point in points:
                points_x.append(point.x)
                points_y.append(point.y)
                x = int((point.x - xmin) / (xmax - xmin) * img_width)
                y = int((ymax - point.y) / (ymax - ymin) * img_height)  # Invert
                draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill=fg_color, outline=fg_color)
        #x = int((geom.x - xmin) / (xmax - xmin) * img_width)
        #y = int((ymax - geom.y) / (ymax - ymin) * img_height)  # Invert
        #draw.ellipse([x - point_size, y - point_size, x + point_size, y + point_size], fill=fg_color, outline=fg_color)

    img.save(png_path)
    img.close()

    #extract points
    #points = list(zip(gdf.geometry.x, gdf.geometry.y))
    points = list(zip(points_x, points_y))
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

def save_color_image(out_path, scaled_data, tif_file):
    """
    Save color image maintaining the original color mapping
    """
    red_band = tif_file.GetRasterBand(3).ReadAsArray()
    green_band = tif_file.GetRasterBand(2).ReadAsArray()
    blue_band = tif_file.GetRasterBand(1).ReadAsArray()

    scaled_color_image = np.stack((red_band, green_band, blue_band), axis=-1)
    scaled_color_image = cv2.normalize(scaled_color_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(out_path, scaled_color_image)

def createPNG_Dataset(folder, out_folder, tile_size=(250,250),point_size=1,bg_color='black',fg_color='white', grayscale=False):
    """
    creates a dataset from a folder of splitted .tif & .shp files into another folder 
    w/ convert_SHPtoPNG(), gdal
    pattern: out_folder/{images / masks}/tile_{ext.}_{i}_{j}.{ext.}
    """
    logging.info(f'⚙️ Creating DATASET:\n\t- from: {folder}\n\t- to: {out_folder}\n\t- tile_size: {tile_size}\n\t- point size: {point_size}\n\t- foreground color: {fg_color}\n\t- background color: {bg_color}')
    start_time = time.time()
    
    tifs_folder = os.path.join(folder, 'tifs')
    shps_folder = os.path.join(folder, 'shps')

    out_tifs_folder = os.path.join(out_folder, 'images')
    out_shps_folder = os.path.join(out_folder, 'masks')
    
    if not os.path.exists(out_tifs_folder):
        os.makedirs(out_tifs_folder)
    if not os.path.exists(out_shps_folder):
        os.makedirs(out_shps_folder)

    files = [f for f in os.listdir(tifs_folder) if f.endswith('.tif')]
    
    total_count = len(files)

    with tqdm(total=total_count, desc='Processing tif files') as pbar:
        for file in files:
            tif_path = os.path.join(tifs_folder, file)
            out_path = os.path.join(out_tifs_folder, os.path.splitext(file)[0] + '.png')
            tif_file = gdal.Open(tif_path)
            scaled = scale_pixel_values(tif_file)
            if scaled.shape != tile_size:
                scaled = extend_image_shape(scaled, tile_size)
            if grayscale:
                cv2.imwrite(out_path, scaled)
            else:
                save_color_image(out_path, scaled, tif_file)
            pbar.update(1)
            
    files = [f for f in os.listdir(shps_folder) if f.endswith('.shp')]
    total_count = len(files)
    try:
        with tqdm(total=total_count, desc='Processing shp files') as pbar:
            for file in files:
                tif_path = os.path.join(tifs_folder, file.replace('_shp_', '_tif_').replace('.shp', '.tif'))
                out_path = os.path.join(out_shps_folder, os.path.splitext(file)[0] + '.png')
                try:
                    convert_SHPtoPNG(tif_path, os.path.join(shps_folder, file), out_path, tile_size, point_size, bg_color, fg_color)
                except Exception as e:
                    logging.warning(f"Conversion error at \n\t{tif_path}\n\t{out_path}\nerror message: {e}")
                pbar.update(1)
    except rasterio._err.CPLE_NotSupportedError as e:
        logging.error(f'🚨 ERROR: bad EPSG\n\tmessage: {e}')
    else:
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info(f'✅ DATASET created in: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')
        return elapsed_time
    return False
def createPNG_Dataset_old(folder, out_folder, tile_size=(250,250),point_size=1,bg_color='black',fg_color='white', grayscale=False):
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
                if grayscale:
                    cv2.imwrite(out_path.replace('shp','tif'), scaled)
                else:
                    save_color_image(out_path.replace('shp','tif'), scaled,tif_file)
            pbar.update(1)
    logging.info(f'dataset created\n\tfrom: {folder}\n\tto: {out_folder}\n\ttile_size: {tile_size}\n\tpoint size: {point_size}\n\tbackground color: {bg_color}\n\tforeground color: {fg_color}')

def convert_TIFtoPNG(folder, out_folder,tile_size=(250,250),grayscale=False):
    """
    converts .tif files to .png files from a folder into another.
    w/ gdal
    """
    start_time = time.time()
    files = os.listdir(folder)
    total_count= len(files)
    os.makedirs(out_folder, exist_ok=True)
    with tqdm(total=total_count, desc='converting TIFs to PNGs') as pbar:
        for file in files:
            tif_path = os.path.join(folder,file)
            out_path = os.path.join(out_folder, os.path.splitext(file)[0]+'.png')
        
            # .tif
            tif_file = gdal.Open(tif_path)
            scaled = scale_pixel_values(tif_file)
            if scaled.shape != tile_size:
                scaled = extend_image_shape(scaled,tile_size)
            if grayscale:
                cv2.imwrite(out_path, scaled)
            else:
                save_color_image(out_path, scaled,tif_file)
            pbar.update(1)
    return time.time()-start_time
    
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
    
def is_image_all_white(image_path):
    """Check if the image at the given path is all white."""
    with Image.open(image_path) as img:
        pixels = list(img.getdata())
        return all(pixel > (200, 200, 200) for pixel in pixels)

def remove_empty_images(folder_path):
    """Remove all images in the given folder that contain only white pixels."""
    start_time = time.time()
    counter = 0
    with tqdm(total=len(os.listdir(folder_path)), desc='drop empty images') as pbar:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')):
                if is_image_all_white(file_path):
                    #print(f"Removing {file_path} (all white)")
                    os.remove(file_path)
                    counter+=1
            pbar.update(1)
    return counter, time.time()-start_time
                    
def create_SHP(folder,out_folder, result_path):
    """
    w/ getPoints_fromPNG(), rasterio, geopandas
    """
    start_time=time.time()
    filenames = os.listdir(out_folder)
    total_count = len(filenames)
    count=0
    
    out_points = [] 
    with tqdm(total=total_count, desc='converting MASKs to SHP') as pbar:
        for filename in filenames:
            if not str(filename).startswith('.'):
                image_coords = getPoints_fromPNG(os.path.join(out_folder,filename))

                tif_path = os.path.join(folder,filename.replace('mask','tile_tif').replace('_shp_','_tif_').replace('.png','.tif'))
                with rasterio.open(tif_path) as src:
                    transform = src.transform  #affine transformation object
                    crs = src.crs  #coordinate Reference System

                geo_coords = [transform * (x, y) for x, y in image_coords]

                #geo_coords -> Points (obj.)
                point_geoms = [Point(coord) for coord in geo_coords]

                count=count+1
                out_points.append(point_geoms)
                pbar.update(1)
    flattened_points = flatten_list(out_points)
    print(len(flattened_points))
    gdf = gpd.GeoDataFrame(geometry=flattened_points, crs=23700)  # Adjust CRS as needed
    gdf.to_file(result_path)
    elapsed_time = time.time() - start_time
    return count, elapsed_time

def flatten_list(lst:list):
    flattened_list = []
    for sublist in lst:
        if isinstance(sublist, list):
            flattened_list.extend(flatten_list(sublist))
        else:
            flattened_list.append(sublist)
    return flattened_list
    
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
    
    
def split_SEG(tif_path, points_shp_path, poly_shp_path, output_folder, tile_size=(200, 200), only_closest = False):
    """
    Split .shp & .tif files using points from a point shapefile to center each split tile (200x200).
    The function cuts both TIF and polygon shapefile over the same areas centered around the points.
    """
    tifs_path_folder = os.path.join(output_folder, 'tifs')
    os.makedirs(tifs_path_folder, exist_ok=True)
    points_gdf = gpd.read_file(points_shp_path)
    tif_dataset = gdal.Open(tif_path)
    tif_transform = tif_dataset.GetGeoTransform()
    pixel_width = tif_transform[1]
    pixel_height = abs(tif_transform[5])

    out_dict = {}
    
    if poly_shp_path:
        shps_path_folder = os.path.join(output_folder, 'shps')
        os.makedirs(shps_path_folder, exist_ok=True)
        logging.info(f'⚙️ TIF and SHP splitting started.')
    else:
        logging.info(f'⚙️ TIF splitting started.')
        
    for idx, point in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0], desc="Processing trees"):
        if isinstance(point.geometry, Point):
            if only_closest:
                process_seg_tile(idx, None, point.geometry.x, point.geometry.y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height)
            else:
                process_seg_tile_with_all_polygons(idx, None, point.geometry.x, point.geometry.y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height)

            out_dict[idx] = point.geometry.x, point.geometry.y
        elif isinstance(point.geometry, MultiPoint):
            for sub_idx, sub_point in enumerate(point.geometry.geoms):
                out_dict[idx] = sub_point.x, sub_point.y
                if only_closest:
                    process_seg_tile(idx, sub_idx, sub_point.x, sub_point.y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height)
                else:
                    process_seg_tile_with_all_polygons(idx, sub_idx, sub_point.x, sub_point.y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height)

        else:
            logging.warning(f'Unsupported geometry type {type(point.geometry)} at index {idx}')
            continue

    if poly_shp_path:
        logging.info(f'✅ TIF and SHP splitting ended.')
    else:
        logging.info(f'✅ TIF splitting ended.')

    return out_dict

#process_seg_tile_with_all_polygons(idx, None, point.geometry.x, point.geometry.y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height)
def split_SEG2(tif_path, shp_path, output_folder, tile_size=(250, 250)): 
    """
    splitting .shp & .tif files 
    w/ split_tif() & split_shp() functions
    pattern: out_folder/{tifs / shps}/tile_{ext.}_{i}_{j}.{ext.}
    """

    tifs_path_folder = os.path.join(output_folder, 'tifs')
    shps_path_folder = os.path.join(output_folder, 'shps')
    
    if not os.path.exists(tifs_path_folder):
        os.makedirs(tifs_path_folder)
    if not os.path.exists(shps_path_folder):
        os.makedirs(shps_path_folder)
    
    logging.info(f'⚙️ TIF splitage started:\n\t- splitted shp: {tif_path}\n\t- to: {tifs_path_folder}')
    
    num_cols, num_rows, elapsed_time = split_tif(tif_path,tifs_path_folder,tile_size)

    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ TIF splitting ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {num_rows}x{num_cols}\n\t- tile size: {tile_size}')

    logging.info(f'⚙️ SHP splitage started:\n\t- splitted shp: {shp_path}\n\t- to: {shps_path_folder}')
    
    count, elapsed_time = split_shp_seg(shp_path, tifs_path_folder, shps_path_folder)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'✅ SHP splitting ended in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {count}')
    if num_rows*num_cols != count:
        logging.warning(f'⚠️ TIFs - SHPs count mismatch: {num_rows*num_cols} - {count}')


def process_seg_tile(idx, sub_idx, center_x, center_y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height):
    """
    Helper function to process a single tile around a point, using only the closest polygon to the center.
    """

    half_width = (tile_size[0] // 2) * pixel_width
    half_height = (tile_size[1] // 2) * pixel_height

    minx, miny = center_x - half_width, center_y - half_height
    maxx, maxy = center_x + half_width, center_y + half_height

    tile_bbox = box(minx, miny, maxx, maxy)

    offset_x = int((minx - tif_transform[0]) / pixel_width)
    offset_y = int((tif_transform[3] - maxy) / pixel_height)

    if poly_shp_path:
        #load polygons within the bounding box
        tile_gdf = gpd.read_file(poly_shp_path, mask=tile_bbox)

        if not tile_gdf.empty:
            #find the closest polygon to the center point
            center_point = Point(center_x, center_y)
            tile_gdf['distance'] = tile_gdf.geometry.distance(center_point)
            closest_polygon = tile_gdf.loc[tile_gdf['distance'].idxmin()]  # Select polygon with minimum distance

            #new GeoDataFrame
            closest_polygon_gdf = gpd.GeoDataFrame([closest_polygon], crs=tile_gdf.crs)

            if sub_idx is not None and sub_idx != 0:
                tile_shp_path = os.path.join(shps_path_folder, f"tile_{idx}_{sub_idx}.shp")
            else:
                tile_shp_path = os.path.join(shps_path_folder, f"tile_{idx}.shp")
            closest_polygon_gdf.to_file(tile_shp_path)
            logging.info(f'Saved tile shapefile for point {idx} (sub-point {sub_idx}) at {tile_shp_path}')

            if sub_idx is not None and sub_idx != 0:
                tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}_{sub_idx}.tif")
            else:
                tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}.tif")
            
            gdal.Translate(
                tif_output_path,
                tif_dataset,
                srcWin=[offset_x, offset_y, tile_size[0], tile_size[1]]
            )
            logging.info(f'Saved tile TIF for point {idx} (sub-point {sub_idx})')
    else:
        #process only the TIF
        if sub_idx is not None and sub_idx != 0:
            tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}_{sub_idx}.tif")
        else:
            tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}.tif")
        
        gdal.Translate(
            tif_output_path,
            tif_dataset,
            srcWin=[offset_x, offset_y, tile_size[0], tile_size[1]]
        )
        logging.info(f'Saved tile TIF for point {idx} (sub-point {sub_idx})')
        
import os
import logging
import geopandas as gpd
from shapely.geometry import box, Point
from osgeo import gdal

def process_seg_tile_with_all_polygons(idx, sub_idx, center_x, center_y, tif_dataset, tif_transform, poly_shp_path, tifs_path_folder, shps_path_folder, tile_size, pixel_width, pixel_height):
    """
    Helper function to process a single tile around a point, saving all polygons within the bounding box.
    """
    half_width = (tile_size[0] // 2) * pixel_width
    half_height = (tile_size[1] // 2) * pixel_height

    minx, miny = center_x - half_width, center_y - half_height
    maxx, maxy = center_x + half_width, center_y + half_height

    tile_bbox = box(minx, miny, maxx, maxy)

    offset_x = int((minx - tif_transform[0]) / pixel_width)
    offset_y = int((tif_transform[3] - maxy) / pixel_height)

    if poly_shp_path:
        # Load polygons within the bounding box
        tile_gdf = gpd.read_file(poly_shp_path, mask=tile_bbox)

        if not tile_gdf.empty:
            # Save all polygons within the bounding box
            if sub_idx is not None and sub_idx != 0:
                tile_shp_path = os.path.join(shps_path_folder, f"tile_{idx}_{sub_idx}.shp")
            else:
                tile_shp_path = os.path.join(shps_path_folder, f"tile_{idx}.shp")
            tile_gdf.to_file(tile_shp_path)
            logging.info(f'Saved tile shapefile for point {idx} (sub-point {sub_idx}) at {tile_shp_path}')

            if sub_idx is not None and sub_idx != 0:
                tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}_{sub_idx}.tif")
            else:
                tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}.tif")
            
            gdal.Translate(
                tif_output_path,
                tif_dataset,
                srcWin=[offset_x, offset_y, tile_size[0], tile_size[1]]
            )
            logging.info(f'Saved tile TIF for point {idx} (sub-point {sub_idx})')
    else:
        # Process only the TIF
        if sub_idx is not None and sub_idx != 0:
            tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}_{sub_idx}.tif")
        else:
            tif_output_path = os.path.join(tifs_path_folder, f"tile_{idx}.tif")
        
        gdal.Translate(
            tif_output_path,
            tif_dataset,
            srcWin=[offset_x, offset_y, tile_size[0], tile_size[1]]
        )
        logging.info(f'Saved tile TIF for point {idx} (sub-point {sub_idx})')



def convert_SHPtoPNG_SEG(tif_path, shp_path, png_path, tile_size=(250, 250), bg_color='black', fg_color='white'):
    '''
    Converts a polygon shapefile to a PNG image based on the extent of a reference TIFF file.
    
    Parameters:
    - tif_path: Path to the reference TIFF file (for CRS and bounds).
    - shp_path: Path to the shapefile containing polygons.
    - png_path: Output path for the resulting PNG image.
    - tile_size: Size of the PNG image (width, height) in pixels.
    - bg_color: Background color of the image.
    - fg_color: Foreground color for polygons.

    Returns:
    - List of polygon vertices as a list of tuples.
    '''
    
    gdf = gpd.read_file(shp_path)
    target_epsg = 23700
    
    with rasterio.open(tif_path) as src:
        src_crs = src.crs
        bbox_transformed = transform_bounds(src_crs, f'EPSG:{target_epsg}', *src.bounds)

    xmin, ymin, xmax, ymax = bbox_transformed
    img_width, img_height = tile_size

    #blank image with background color
    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)

    polygons_points = []

    #iterate through polygons
    for geom in gdf.geometry:
        if geom.geom_type == 'Polygon':
            # Extract the exterior coordinates of the polygon
            polygon_coords = [(int((x - xmin) / (xmax - xmin) * img_width),
                               int((ymax - y) / (ymax - ymin) * img_height))  # Invert y
                              for x, y in geom.exterior.coords]
            draw.polygon(polygon_coords, fill=fg_color, outline=fg_color)
            polygons_points.append(polygon_coords)
            
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                polygon_coords = [(int((x - xmin) / (xmax - xmin) * img_width),
                                   int((ymax - y) / (ymax - ymin) * img_height))  # Invert y
                                  for x, y in poly.exterior.coords]
                draw.polygon(polygon_coords, fill=fg_color, outline=fg_color)
                polygons_points.append(polygon_coords)

    img.save(png_path)
    img.close()

    return polygons_points


def createPNG_Dataset_SEG(folder, out_folder, tile_size=(250,250),point_size=1,bg_color='black',fg_color='white', grayscale=False):
    """
    creates a dataset from a folder of splitted .tif & .shp files into another folder 
    w/ convert_SHPtoPNG(), gdal
    pattern: out_folder/{images / masks}/tile_{index}.{ext.}
    """
    logging.info(f'⚙️ Creating DATASET:\n\t- from: {folder}\n\t- to: {out_folder}\n\t- tile_size: {tile_size}\n\t- point size: {point_size}\n\t- foreground color: {fg_color}\n\t- background color: {bg_color}')
    start_time = time.time()
    
    tifs_folder = os.path.join(folder, 'tifs')
    shps_folder = os.path.join(folder, 'shps')

    out_tifs_folder = os.path.join(out_folder, 'images')
    out_shps_folder = os.path.join(out_folder, 'masks')
    
    if not os.path.exists(out_tifs_folder):
        os.makedirs(out_tifs_folder)
    if not os.path.exists(out_shps_folder):
        os.makedirs(out_shps_folder)

    files = [f for f in os.listdir(tifs_folder) if f.endswith('.tif')]
    
    total_count = len(files)

    with tqdm(total=total_count, desc='Processing tif files') as pbar:
        for file in files:
            tif_path = os.path.join(tifs_folder, file)
            out_path = os.path.join(out_tifs_folder, os.path.splitext(file)[0] + '.png')
            tif_file = gdal.Open(tif_path)
            scaled = scale_pixel_values(tif_file)
            if scaled.shape != tile_size:
                scaled = extend_image_shape(scaled, tile_size)
            if grayscale:
                cv2.imwrite(out_path, scaled)
            else:
                save_color_image(out_path, scaled, tif_file)
            pbar.update(1)
            
    files = [f for f in os.listdir(shps_folder) if f.endswith('.shp')]
    total_count = len(files)
    try:
        with tqdm(total=total_count, desc='Processing shp files') as pbar:
            for file in files:
                tif_path = os.path.join(tifs_folder, file.replace('shp', 'tif'))
                out_path = os.path.join(out_shps_folder, os.path.splitext(file)[0] + '.png')
                try:
                    convert_SHPtoPNG_SEG(tif_path, os.path.join(shps_folder, file), out_path, tile_size, bg_color, fg_color)
                except Exception as e:
                    logging.warning(f"Conversion error at \n\t{tif_path}\n\t{out_path}\nerror message: {e}")
                pbar.update(1)
    except rasterio._err.CPLE_NotSupportedError as e:
        logging.error(f'🚨 ERROR: bad EPSG\n\tmessage: {e}')
    else:
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logging.info(f'✅ DATASET created in: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')
        return elapsed_time
    return False




from shapely.geometry import Polygon
from skimage import measure

def create_SHP_SEG(folder, out_folder, result_path, target_crs=23700):
    """
    Converts mask PNG files into a shapefile with polygons.
    
    Parameters:
    - folder: Directory containing the original TIFF files.
    - out_folder: Directory containing mask PNG files.
    - result_path: Path to save the output shapefile.
    - target_crs: Target coordinate reference system (CRS) for the output shapefile.
    
    Returns:
    - count: Number of polygons created.
    - elapsed_time: Time taken for the conversion process.
    """
    start_time = time.time()
    filenames = os.listdir(out_folder)
    total_count = len(filenames)
    count = 0
    crs = target_crs
    polygons = [] 
    with tqdm(total=total_count, desc='Converting MASKs to SHP') as pbar:
        for filename in filenames:
            if not filename.startswith('.'):
                tif_path = os.path.join(folder, filename.replace('.png', '.tif').replace('mask','tile_tif'))
                png_path = os.path.join(out_folder, filename)
                
                mask_img = Image.open(png_path).convert("1")
                mask_array = np.array(mask_img)
                
                with rasterio.open(tif_path) as src:
                    transform = src.transform
                    crs = src.crs
                
                contours = measure.find_contours(mask_array, level=0.5)
                
                for contour in contours:
                    geo_coords = [transform * (x, y) for y, x in contour]
                
                    polygon_geom = Polygon(geo_coords)
                    polygons.append(polygon_geom)
                    count += 1
                
                pbar.update(1)
    
    #geodataframe
    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    gdf.to_file(result_path)
    
    elapsed_time = time.time() - start_time
    return count, elapsed_time