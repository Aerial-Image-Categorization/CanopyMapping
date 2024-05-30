import sys
sys.path.append('../')
import os
from utils.imageprocessing import split, createPNG_Dataset,createPNG_Dataset_old, convert_TIFtoPNG, convert_PNGtoSHP, show_image
from utils.datasetvalidation import dropna_SHPs, set_valid_CRS

if __name__ == '__main__':
    folder = '../data/2024-05-20-dataset/standard/original'
    out_folder = '../data/2024-05-20-dataset/standard/formatted'
    size = (200,200)

    os.makedirs(folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # 1. data prep.
    #   - splitting geo. images
    split('../data/all_data/2023-02-23_Bakonyszucs_actual.tif',
          '../data/all_data/Fa_pontok.shp',
          folder,
          size)

    #   - convert to .png for the model
    # convert_TIFtoPNG(folder,out_folder,size)
    #     or
    # create dataset
    #dropna_SHPs(os.path.join(folder,'shps'))
    set_valid_CRS(os.path.join(folder,'shps'), desired_crs_epsg=23700)
    createPNG_Dataset(folder,out_folder,size, point_size=28, grayscale=False)


    # 2. prediction

    # ****
    # model pred.
    #print('predicted')
    # ****

    # 4. Convert the predictions back
    #convert_PNGtoSHP(folder,out_folder,'../results')