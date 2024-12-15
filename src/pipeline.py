import sys
sys.path.append('../')

import os
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='../logs/prediction_pipeline.log',
                    filemode='w')
import argparse

from utils.imageprocessing import split_tif, remove_empty_images, convert_TIFtoPNG, create_SHP, create_SHP_SEG

#from models.biomed_UNet import model as UNetModel
from models import biomed_UNet as UNetModel
from models.biomed_UNet.predict import predict_img, mask_to_image
from ultralytics import YOLO
from torch.utils.data import DataLoader
import torch
from PIL import Image

import numpy as np

from tqdm.auto import tqdm

import time
import shutil
import cv2

import torch.nn.functional as F

'''
pipeline:
    - split tif into tiles
    - drop empty tifs
    - convert tifs to pngs
    - prediction (pngs -> masks)
    - convert masks to one shp
'''
def process_yolo_image(net, image, device):
    if not isinstance(image, torch.Tensor):
        # Convert to tensor if not already a tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            raise TypeError("Input to process_yolo_image must be a torch.Tensor or a numpy.ndarray")
    results = net(image, save=True, imgsz=512, conf=0.2)
    try:
        for result in results:
            masks = result.masks.data
            boxes = result.boxes.data
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            mask_pred = out = torch.any(people_masks, dim=0).int()
            cv2.imwrite('test_yolo_loc.jpg', (mask_pred * 255).cpu().numpy())
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
    except Exception as e:
        logging.error(f"Error processing YOLO segmentation: {repr(e)}")
        mask_pred = torch.zeros((512, 512), device=device)
    mask_pred = mask_pred.unsqueeze(0)
    return mask_pred, (out * 255).cpu().numpy()

if __name__ == '__main__':
    #read arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--tif_path', type=str)
    parser.add_argument('--loc_model_name', type=str, default = 'YOLOv11n-seg/yolo11n_loc/yolo11n_512/weights/best.pt')
    parser.add_argument('--loc_result_path', type=str, default = 'loc_result_shp.shp')
    parser.add_argument('--loc_tile_size', type=tuple, default = (1024,1024))
    parser.add_argument('--seg_model_name', type=str, default = 'checkpoints_seg-unet-biomed-2048_2024-12-08_16-20_sota/checkpoint_epoch34.pth')
    parser.add_argument('--seg_result_path', type=str, default = 'seg_result_shp.shp')
    parser.add_argument('--seg_tile_size', type=tuple, default = (2048,2048))
    parser.add_argument('--cuda', type=bool, default = None)
    parser.add_argument('--num_workers', type=int, default = os.cpu_count())
    parser.add_argument('--batch_size', type=int, default = 1)
    parser.add_argument('--threshold', type=float, default = 0.9)
    parser.add_argument('--scale_factor', type=int, default = 1)
    parser.add_argument('--working_folder_path', type=str, default = './working_folder')
    parser.add_argument('--delete_working_folder', type=bool, default = True)
    
    args = parser.parse_args()

    args.tif_path = '../data/all_data/2023-02-23_Bakonyszucs_actual.tif'
    args.tif_path = 'test.tif'
    args.delete_working_folder = False


    os.makedirs(os.path.join(args.working_folder_path,'loc_tifs'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'loc_images'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'loc_masks'), exist_ok=True)

    
    #split tif
    logging.info(f"⚙️ TIF splitage started:\n\t- tif path: {args.tif_path}\n\t- into: {os.path.join(args.working_folder_path,'loc_tifs')}")
    
    num_cols, num_rows, elapsed_time = split_tif(args.tif_path,os.path.join(args.working_folder_path,'loc_tifs'),args.loc_tile_size)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 TIF splitage finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {num_rows}x{num_cols}\n\t- tile size: {args.loc_tile_size}')


    #drop empty tifs
    logging.info(f"⚙️ Drop empty TIFs:\n\t- tifs path: {os.path.join(args.working_folder_path,'loc_tifs')}")
    empty_tiles_count, elapsed_time = remove_empty_images(os.path.join(args.working_folder_path,'loc_tifs'))
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 Drop empties finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- empty tiles count: {empty_tiles_count}')
    
    #tifs to pngs
    logging.info(f"⚙️ Convert TIFs to PNGs:\n\t- tifs path: {os.path.join(args.working_folder_path,'loc_tifs')}\n\t- pngs path: {os.path.join(args.working_folder_path,'loc-images')}")
    elapsed_time = convert_TIFtoPNG(
        os.path.join(args.working_folder_path,'loc_tifs'),
        os.path.join(args.working_folder_path,'loc_images'), 
        tile_size=args.loc_tile_size,
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
    #try:
    #    Model = UNetModel.from_pretrained(args.loc_model_name, device)
    #except Exception as e:
    #    logging.error('🚨: %s', repr(e))
    #else:
    #    logging.info(f'🔮 Model successfully loaded from 🤗 HuggingFace 🤗\n\t- model name: {args.loc_model_name}')
    #
    #Model.to(device)
    Model = YOLO(args.loc_model_name)
    
    #prediction
    logging.info(f"⚙️ Prediction started on\n\t{os.path.join(args.working_folder_path,'loc_masks')}")
    start_time = time.time()
    with tqdm(total=len(os.listdir(os.path.join(args.working_folder_path,'loc_images'))), desc='prediction') as pbar:
        for image_path in os.listdir(os.path.join(args.working_folder_path,'loc_images')):
            image = Image.open(os.path.join(args.working_folder_path,'loc_images',image_path))
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image = image.resize((512, 512), Image.ANTIALIAS)
            try:
                image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to 0-1 range
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
                mask, out_img = process_yolo_image(net=Model, image=image_tensor, device=device)
                print(type(out_img))
                print(out_img.shape)
                #out_img = out_img.squeeze(0).cpu().numpy()
                #out_img = F.interpolate(out_img, size=(1024, 1024), mode='bilinear', align_corners=False)
                out_img = out_img.astype(np.float32)
                out_img = cv2.resize(out_img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                #out_img = mask_to_image(mask,[0, 1])
                x,y = os.path.basename(image_path).split('.')[0].split('_')[2:4]
                cv2.imwrite(os.path.join(args.working_folder_path,'loc_masks',f'mask_{x}_{y}.png'), out_img)
            except Exception as e:
                logging.error(f"{os.path.join(args.working_folder_path,'loc_images',image_path)} 🚨: %s", repr(e))
            pbar.update(1)
            
    hours, remainder = divmod(time.time()-start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Prediction finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')

    
    #convert pngs to one shp
    logging.info(f"⚙️ Convert PNGs to SHP:\n\t- pngs path: {os.path.join(args.working_folder_path,'loc_images')}\n\t- result path: {args.loc_result_path}")
    detected_trees_count, elapsed_time = create_SHP(os.path.join(args.working_folder_path,'loc_tifs'),os.path.join(args.working_folder_path,'loc_masks'),args.loc_result_path)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Conversion finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- {detected_trees_count} trees detected')


    
    ###############
    #-------------#
    ###############
    
    
    
    os.makedirs(os.path.join(args.working_folder_path,'seg_tifs'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'seg_images'), exist_ok=True)
    os.makedirs(os.path.join(args.working_folder_path,'seg_masks'), exist_ok=True)

    
    #split tif
    logging.info(f"⚙️ TIF splitage started:\n\t- tif path: {args.tif_path}\n\t- into: {os.path.join(args.working_folder_path,'seg_tifs')}")
    
    num_cols, num_rows, elapsed_time = split_tif(args.tif_path,os.path.join(args.working_folder_path,'seg_tifs'),args.seg_tile_size)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 TIF splitage finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- split size: {num_rows}x{num_cols}\n\t- tile size: {args.seg_tile_size}')


    #drop empty tifs
    logging.info(f"⚙️ Drop empty TIFs:\n\t- tifs path: {os.path.join(args.working_folder_path,'seg_tifs')}")
    empty_tiles_count, elapsed_time = remove_empty_images(os.path.join(args.working_folder_path,'seg_tifs'))
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f'🏁 Drop empties finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- empty tiles count: {empty_tiles_count}')
    
    #tifs to pngs
    logging.info(f"⚙️ Convert TIFs to PNGs:\n\t- tifs path: {os.path.join(args.working_folder_path,'seg_tifs')}\n\t- pngs path: {os.path.join(args.working_folder_path,'seg-images')}")
    elapsed_time = convert_TIFtoPNG(
        os.path.join(args.working_folder_path,'seg_tifs'),
        os.path.join(args.working_folder_path,'seg_images'), 
        tile_size=args.seg_tile_size,
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
    #try:
    #    Model = UNetModel.from_pretrained(args.loc_model_name, device)
    #except Exception as e:
    #    logging.error('🚨: %s', repr(e))
    #else:
    #    logging.info(f'🔮 Model successfully loaded from 🤗 HuggingFace 🤗\n\t- model name: {args.loc_model_name}')
    #
    #Model.to(device)
    Model = UNetModel.model(UNetModel.config(n_channels=3, n_classes=1, bilinear=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.seg_model_name}')
    logging.info(f'Using device {device}')

    Model.to(device=device)
    state_dict = torch.load(args.seg_model_name, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    Model.load_state_dict(state_dict)
    
    
    #prediction
    logging.info(f"⚙️ Prediction started on\n\t{os.path.join(args.working_folder_path,'seg_masks')}")
    start_time = time.time()
    with tqdm(total=len(os.listdir(os.path.join(args.working_folder_path,'seg_images'))), desc='prediction') as pbar:
        for image_path in os.listdir(os.path.join(args.working_folder_path,'seg_images')):
            image = Image.open(os.path.join(args.working_folder_path,'seg_images',image_path))
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image = image.resize((512, 512), Image.ANTIALIAS)
            try:
                mask = predict_img(net=Model,
                               full_img=image,
                               scale_factor=args.scale_factor,
                               out_threshold=args.threshold,
                               device=device)
                #mask = F.interpolate(mask, size=(2048, 2048), mode='bilinear', align_corners=False)
                print(type(out_img))
                print(out_img.shape)
                
                out_img = mask_to_image(mask,[0, 1])
                
                out_img = np.array(out_img)
                out_img = out_img.astype(np.float32)
                out_img = cv2.resize(out_img, (2048, 2048), interpolation=cv2.INTER_LINEAR)
                
                x,y = os.path.basename(image_path).split('.')[0].split('_')[2:4]
                cv2.imwrite(os.path.join(args.working_folder_path,'seg_masks',f'mask_{x}_{y}.png'), out_img)
                #out_img.save(os.path.join(args.working_folder_path,'seg_masks',f'mask_{x}_{y}.png'))
            except Exception as e:
                logging.error(f"{os.path.join(args.working_folder_path,'seg_images',image_path)} 🚨: %s", repr(e))
            pbar.update(1)
            
    hours, remainder = divmod(time.time()-start_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Prediction finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}')

    
    #convert pngs to one shp
    logging.info(f"⚙️ Convert PNGs to SHP:\n\t- pngs path: {os.path.join(args.working_folder_path,'seg_images')}\n\t- result path: {args.seg_result_path}")
    detected_trees_count, elapsed_time = create_SHP_SEG(os.path.join(args.working_folder_path,'seg_tifs'),os.path.join(args.working_folder_path,'seg_masks'),args.loc_result_path)
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f'🏁 Conversion finished in {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}\n\t- {detected_trees_count} trees detected')



    #delete tmp dir
    if args.delete_working_folder:
        shutil.rmtree(args.working_folder_path)
        logging.info(f'🧹 Working folder deleted successfully')
    
    
    