import os
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from models.UNet.datasets import ImageNameDataset
import shutil
#from utils.imageprocessing import show_image
#from utils.datasets import load_image

from tqdm import tqdm

def rotate_train_pairs(train_images_folder,train_masks_folder):
    t_rot_90= T.RandomRotation((90,90))
    t_rot_180 = T.RandomRotation((180,180))
    t_rot_270 = T.RandomRotation((270,270))
    
    image_filenames = os.listdir(train_images_folder)
    total_len = len(image_filenames)
    with tqdm(total=total_len, desc='generating rotated images') as pbar:
        for image_filename in image_filenames:
            image = Image.open(os.path.join(train_images_folder,image_filename))
            filename_splitext = image_filename.split('.')
            t_rot_90(image).save(os.path.join(train_images_folder,filename_splitext[0]+'_90.'+filename_splitext[1]))
            t_rot_180(image).save(os.path.join(train_images_folder,filename_splitext[0]+'_180.'+filename_splitext[1]))
            t_rot_270(image).save(os.path.join(train_images_folder,filename_splitext[0]+'_270.'+filename_splitext[1]))

            mask_filename = image_filename.replace('tif','shp')

            mask = Image.open(os.path.join(train_masks_folder, mask_filename))
            filename_splitext = mask_filename.split('.')
            t_rot_90(mask).save(os.path.join(train_masks_folder,filename_splitext[0]+'_90.'+filename_splitext[1]))
            t_rot_180(mask).save(os.path.join(train_masks_folder,filename_splitext[0]+'_180.'+filename_splitext[1]))
            t_rot_270(mask).save(os.path.join(train_masks_folder,filename_splitext[0]+'_270.'+filename_splitext[1]))
            pbar.update(1)