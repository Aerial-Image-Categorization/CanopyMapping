import sys
sys.path.append('../')

import os
import argparse

from utils.imageprocessing import *

from models.biomed_UNet import *

from torch.utils.data import DataLoader
import torch
from PIL import Image

import logging

from tqdm.auto import tqdm

import time
import shutil

'''
pipeline:
    - Load model from HuggingFace
    - TEST - old score
    - TRAIN
    - TEST - new score
    - Save model to HuggingFace
'''

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='./logs/biomed_train.log',
                    filemode='w')
    #read arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name', type=str, default = 'aerial-image-categorization/tree-detection_biomed-benchmark')
    parser.add_argument('--tif_path', type=str)
    parser.add_argument('--shp_path', type=str)
    parser.add_argument('--working_folder_path', type=str, default = './working_folder')
    parser.add_argument('--delete_working_folder', type=bool, default = True)
    
    args = parser.parse_args()

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





