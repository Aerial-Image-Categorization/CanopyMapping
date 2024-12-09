#import sys
#sys.path.append('../')
import os
import argparse
import torch
from models import centroid_UNet as UNet
from models.biomed_UNet.datasets import ImageDataset

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    img_size = args.size
    train_set_path = f'../data/2024-10-30-loc-dataset-{img_size}/u_aug_train_u10'
    valid_set_path = f'../data/2024-10-30-loc-dataset-{img_size}/u_val'

    epochs = args.epochs #25
    batch_size = args.batchsize #6
    lr = 1e-4
    scale = 1
    val = 0.1
    amp = False
    bilinear = True
    
    model = UNet.model(n_channels=3, n_classes=1, bilinear=bilinear)
    dataset = None #UNet.ImageDataset(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_set = ImageDataset(train_set_path)
    valid_set = ImageDataset(valid_set_path)
    
    try:
        UNet.train_net_loc(
            train_set = train_set,
            valid_set = valid_set,
            model=model,
            epochs=epochs,
            img_size = img_size,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            img_scale=scale,
            val_percent=val / 100,
            amp=amp
        )
    except Exception:#torch.cuda.OutOfMemoryError:
        #logging.error('Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        UNet.train_net(
            train_set = train_set,
            valid_set = valid_set,
            model=model,
            epochs=epochs,
            img_size = img_size,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            img_scale=scale,
            val_percent=val / 100,
            amp=amp
        )