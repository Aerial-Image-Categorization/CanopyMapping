import os
from torch.utils.data import Dataset

class ImageNameDataset(Dataset):
    def __init__(self, dataset_path, sort=False) -> None:
        image_filenames = os.listdir(os.path.join(dataset_path,'images'))
        mask_filenames = os.listdir(os.path.join(dataset_path,'masks'))
        self.images_path = []
        self.masks_path = []
        
        for file in image_filenames:
            if file.startswith('tile'):
                self.images_path.append(os.path.join(os.path.join(dataset_path,'images'),file))
        for file in mask_filenames:
            if file.startswith('tile'):
                self.masks_path.append(os.path.join(os.path.join(dataset_path,'masks'),file))
                
        if sort:
            if len(file.split('_')) == 4:
                self.images_path = sorted(self.images_path, key=lambda x: (int(os.path.basename(x).split('_')[2]), int(os.path.basename(x).split('.')[0].split('_')[3])))
                self.masks_path = sorted(self.masks_path, key=lambda x: (int(os.path.basename(x).split('_')[2]), int(os.path.basename(x).split('.')[0].split('_')[3])))   
            elif len(file.split('_')) == 2:
                self.images_path = sorted(self.images_path, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
                self.masks_path = sorted(self.masks_path, key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))


                
                
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        return {
            'image': self.images_path[idx],
            'mask': self.masks_path[idx]
        }