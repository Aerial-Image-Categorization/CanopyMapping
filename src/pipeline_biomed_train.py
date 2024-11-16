import sys
sys.path.append('../')

import os
import argparse

from utils.imageprocessing import *

from models.biomed_UNet import model as UNetModel

from torch.utils.data import DataLoader
import torch
from PIL import Image

import logging

from tqdm.auto import tqdm

import time
import shutil

import subprocess
from huggingface_hub import login, HfApi, HfFolder, Repository


'''
pipeline:
    - Load model from HuggingFace
    - TEST - old score
    - TRAIN
    - TEST - new score
    - Save model to HuggingFace
'''
# sudo apt-get install git-lfs


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
    parser.add_argument('--cuda', type=bool, default = None)
    parser.add_argument('--working_folder_path', type=str, default = 'working_folder')
    parser.add_argument('--delete_working_folder', type=bool, default = True)
    
    args = parser.parse_args()

    args.model_name = 'aerial-image-categorization/treedetection-test-model'
    args.test=''
    args.working_folder_path='train-pipeline-folder'
    hf_token='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'

    os.makedirs(args.working_folder_path,exist_ok = True)
    os.makedirs(os.path.join(args.working_folder_path,"model"),exist_ok = True)

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

    #hf login
    #try:
    #    subprocess.run(['chmod', '+x', 'hf_login.sh'], check=True)
    #    subprocess.run(['./hf_login.sh'], check=True)
    #    logging.info(f'🔮 Logged in successfully to 🤗 HuggingFace 🤗')
    #except subprocess.CalledProcessError as e:
    #    print(f"An error occurred during in 🤗 HuggingFace 😊 login: {e}")
    #    logging.info(f'🔮 An error occured while logging into 🤗 HuggingFace 🤗')

    try:
        login(token=hf_token)
        logging.info(f'🔮 Logged in successfully to 🤗 HuggingFace 🤗')
    except Exception as e:
        print(f"An error occurred while logging in: {e}")
        logging.info(f'🔮 An error occured while logging into 🤗 HuggingFace 🤗')
        
    #load model
    try:
        Model = UNetModel.from_pretrained(args.model_name, device)
    except Exception as e:
        logging.error('🚨: %s', repr(e))
    else:
        logging.info(f'🔮 Model successfully loaded from 🤗 HuggingFace 🤗\n\t- model name: {args.model_name}')
        
    Model.to(device)
    
    #save model
    try:
        Model.save_pretrained(os.path.join(args.working_folder_path,"model"))
        api = HfApi()
        api.create_repo(repo_id=args.model_name, exist_ok=True)
        subprocess.run(['git', '-C', save_directory, 'init'], check=True)
        subprocess.run(['git', '-C', save_directory, 'remote', 'add', 'origin', f'https://huggingface.co/{repository_id}'], check=True)
        subprocess.run(['git', '-C', save_directory, 'pull', 'origin', 'main'], check=True)
        #api.create_repo(repo_id=args.model_name)
        repo = Repository(local_dir=os.path.join(args.working_folder_path,"model"), clone_from=args.model_name)
        repo.push_to_hub(commit_message="train-pipeline-update")
        logging.info(f'🔮 Model successfully saved to 🤗 HuggingFace 🤗\n\t- repository: {args.model_name}')
    except Exception as e:
        print(f"An error occurred during saving the model to 🤗 HuggingFace 😊: {e}")
        logging.info(f'🔮 An error occurred during saving the model to 🤗 HuggingFace 🤗')
        






