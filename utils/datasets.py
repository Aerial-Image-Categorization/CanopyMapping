import os
from os.path import splitext

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

import cv2


def get_size(dataset:Dataset,train_size_ratio):
    dataset_size = len(dataset)
    train_size = round(dataset_size*train_size_ratio)
    test_size = dataset_size-train_size
    actual_ratio = train_size / dataset_size
    return train_size, test_size, actual_ratio

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        #return Image.open(filename)
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)#.convert('L')

class ImageDataset(Dataset):
    def __init__(self, dataset_path) -> None:
        images = os.listdir(os.path.join(dataset_path,'images'))
        #files = os.listdir(dataset_path)
        self.images_path = []
        self.masks_path = []
        self.image_coords = []
        self.mask_coords = []
        
        for file in images:
            self.images_path.append(os.path.join(os.path.join(dataset_path,'images'),file))
            self.masks_path.append(os.path.join(os.path.join(dataset_path,'masks'),file.replace('tif','shp')))
            xy = splitext(file)[0].split('_')[2:4]
            self.image_coords.append((int(xy[0]),int(xy[1])))
            xy = splitext(file.replace('tif','shp'))[0].split('_')[2:4]
            self.mask_coords.append((int(xy[0]),int(xy[1])))
                
    def __len__(self):
        return len(self.images_path)
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(range(mask_values)):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
        
    def __getitem__(self, idx):
        #return {
        #    'image': self.images_path[idx],
        #    'mask': self.masks_path[idx]
        #}
        mask = load_image(self.images_path[idx])
        img = load_image(self.masks_path[idx])
        #img = img.convert('L')
        
        assert img.size == mask.size, \
            f'Image and mask should be the same size, but are {img.size} and {mask.size}'

        self.scale = 1.0
        #self.scale = 0.2
        
        img = self.preprocess(1, img, self.scale, is_mask=False)
        mask = self.preprocess(1, mask, self.scale, is_mask=True)
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'image_coords': self.image_coords[idx],
            'mask_coords': self.mask_coords[idx],
        }
    
class ImageNameDataset(Dataset):
    def __init__(self, dataset_path, sort=False) -> None:
        image_filenames = os.listdir(os.path.join(dataset_path,'images'))
        mask_filenames = os.listdir(os.path.join(dataset_path,'masks'))
        self.images_path = []
        self.masks_path = []
        for file in image_filenames:
            if file.split('_')[1]=='tif':
                self.images_path.append(os.path.join(os.path.join(dataset_path,'images'),file))
        for file in mask_filenames:
            if file.split('_')[1]=='shp':
                self.masks_path.append(os.path.join(os.path.join(dataset_path,'masks'),file))
        if sort:
            self.images_path = sorted(self.images_path, key=lambda x: (int(os.path.basename(x).split('_')[2]), int(os.path.basename(x).split('.')[0].split('_')[3])))
            self.masks_path = sorted(self.masks_path, key=lambda x: (int(os.path.basename(x).split('_')[2]), int(os.path.basename(x).split('.')[0].split('_')[3])))

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        return {
            'image': self.images_path[idx],
            'mask': self.masks_path[idx]
        }