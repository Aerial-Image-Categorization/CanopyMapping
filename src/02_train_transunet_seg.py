#import sys
#sys.path.append('../')
import sys
import os
import argparse
import logging
import torch
import torch.nn as nn

#from models import centroid_UNet as UNet
from models.DSTransUNet.train_seg import train_net
from models.DSTransUNet.lib.DS_TransUNet import UNet
        
def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=50,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='choosing optimizer Adam or SGD')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    #train_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/u_aug_train_u10/images/'
    #train_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/u_aug_train_u10/masks/'
    #val_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/u_val/images/'
    #val_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/u_val/masks/'
    train_img_dir = f'../data/2024-11-13-seg-dataset-{args.size}/u_train'
    train_mask_dir = f'../data/2024-11-13-seg-dataset-{args.size}/u_aug_train/masks/'
    val_img_dir = f'../data/2024-11-13-seg-dataset-{args.size}/u_val'
    val_mask_dir = f'../data/2024-11-13-seg-dataset-{args.size}/u_val/masks/'
    dir_checkpoint = f'ds_transunet_seg_checkpoints_{args.size}/'
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #net = UNet(128, 1)
    net = UNet(384, 1)
    net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net,
                  train_img_dir = train_img_dir,
                  train_mask_dir = train_mask_dir,
                  val_img_dir = val_img_dir,
                  val_mask_dir = val_mask_dir,
                  dir_checkpoint = dir_checkpoint,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr= 1e-5, #args.lr,
                  device=device,
                  img_size=512#args.size#512 ## TODO !!! args.size
                  
        ) 
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
