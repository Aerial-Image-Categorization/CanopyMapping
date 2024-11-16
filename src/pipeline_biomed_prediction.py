import sys
sys.path.append('../')

import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/biomed_prediction.log',
                    filemode='w')
import argparse

from utils.imageprocessing import split_tif, remove_empty_images, convert_TIFtoPNG, create_SHP

from models.biomed_UNet import model as UNetModel
from models.biomed_UNet.predict import predict_img, mask_to_image
from torch.utils.data import DataLoader
import torch
from PIL import Image


from tqdm.auto import tqdm

import time
import shutil

'''
pipeline:
    - split tif into tiles
    - drop empty tifs
    - convert tifs to pngs
    - prediction (pngs -> masks)
    - convert masks to one shp
'''

if __name__ == '__main__':
    #read arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default = 'aerial-image-categorization/tree-detection_biomed-benchmark')
    parser.add_argument('--tif_path', type=str)
    parser.add_argument('--result_path', type=str, default = 'result_shp.shp')
    parser.add_argument('--tile_size', type=tuple, default = (200,200))
    parser.add_argument('--cuda', type=bool, default = None)
    parser.add_argument('--num_workers', type=int, default = os.cpu_count())
    parser.add_argument('--batch_size', type=int, default = 1)
    parser.add_argument('--threshold', type=float, default = 0.9)
    parser.add_argument('--scale_factor', type=int, default = 1)
    parser.add_argument('--working_folder_path', type=str, default = './working_folder')
    parser.add_argument('--delete_working_folder', type=bool, default = True)
    
    args = parser.parse_args()

    args.tif_path = '../data/all_data/2023-02-23_Bakonyszucs_actual.tif'
    args.delete_working_folder = False


    os.makedirs(os.path.join(args.working_folder_path,'tifs'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'images'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'masks'), exist_ok=True)

    
    #split tif
    logging.info(f"⚙️ TIF splitage started:\n\t- tif path: {args.tif_path}\n\t- into: {os.path.join(args.working_folder_path,'tifs')}")
    
    num_cols, num_rows, elapsed_time = split_tif(args.tif_path,os.path.join(args.working_folder_path,'tifs'),args.tile_size)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 TIF splitage finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {num_rows}x{num_cols}\n\t- tile size: {args.tile_size}')


    #drop empty tifs
    logging.info(f"⚙️ Drop empty TIFs:\n\t- tifs path: {os.path.join(args.working_folder_path,'tifs')}")
    empty_tiles_count, elapsed_time = remove_empty_images(os.path.join(args.working_folder_path,'tifs'))
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 Drop empties finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- empty tiles count: {empty_tiles_count}')
    
    #tifs to pngs
    logging.info(f"⚙️ Convert TIFs to PNGs:\n\t- tifs path: {os.path.join(args.working_folder_path,'tifs')}\n\t- pngs path: {os.path.join(args.working_folder_path,'images')}")
    elapsed_time = convert_TIFtoPNG(
        os.path.join(args.working_folder_path,'tifs'),
        os.path.join(args.working_folder_path,'images'), 
        tile_size=args.tile_size,
        grayscale=False
    )
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Conversion finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')
    
    #checking for device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.cuda==True and device == torch.device('cuda'):
        logging.info('💨 Using CUDA')
    elif args.cuda==False and device == torch.device('cpu'):
        logging.info('🐌 Using CPU')
    elif args.cuda==False and device == torch.device('cuda'):
        logging.warning('🐌 Using CPU\n\t⚠️ CUDA is available')
        device = torch.device('cpu')
    elif args.cuda==True and device == torch.device('cpu'):
        logging.warning('🐌 Using CPU\n\t⚠️ CUDA is not available')
    elif args.cuda is None:
        logging.info(f"Automatic set to {'💨 CUDA' if device==torch.device('cuda') else '🐌 CPU'}")

    
    #load model
    try:
        Model = UNetModel.from_pretrained(args.model_name, device)
    except Exception as e:
        logging.error('🚨: %s', repr(e))
    else:
        logging.info(f'🔮 Model successfully loaded from 🤗 HuggingFace 🤗\n\t- model name: {args.model_name}')

    Model.to(device)
    
    #prediction
    logging.info(f"⚙️ Prediction started on\n\t{os.path.join(args.working_folder_path,'masks')}")
    start_time = time.time()
    with tqdm(total=len(os.listdir(os.path.join(args.working_folder_path,'images'))), desc='prediction') as pbar:
        for image_path in os.listdir(os.path.join(args.working_folder_path,'images')):
            image = Image.open(os.path.join(args.working_folder_path,'images',image_path))
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            try:
                mask = predict_img(net=Model,
                               full_img=image,
                               scale_factor=args.scale_factor,
                               out_threshold=args.threshold,
                               device=device)
                out_img = mask_to_image(mask,[0, 1])
                x,y = os.path.basename(image_path).split('.')[0].split('_')[2:4]
                out_img.save(os.path.join(args.working_folder_path,'masks',f'mask_{x}_{y}.png'))
            except Exception as e:
                logging.error(f"{os.path.join(args.working_folder_path,'images',image_path)} 🚨: %s", repr(e))
            pbar.update(1)
            
    hours, remainder = divmod(time.time()-start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Prediction finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')

    
    #convert pngs to one shp
    logging.info(f"⚙️ Convert PNGs to SHP:\n\t- pngs path: {os.path.join(args.working_folder_path,'images')}\n\t- result path: {args.result_path}")
    detected_trees_count, elapsed_time = create_SHP(os.path.join(args.working_folder_path,'tifs'),os.path.join(args.working_folder_path,'masks'),args.result_path)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Conversion finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- {detected_trees_count} trees detected')


    #delete tmp dir
    if args.delete_working_folder:
        shutil.rmtree(args.working_folder_path)
        logging.info(f'🧹 Working folder deleted successfully')
    
    
    