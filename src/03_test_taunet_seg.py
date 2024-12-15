import sys
import os

import argparse
import torch
import logging
from models.TAUNet.loc import test_net_seg
from models import TAUNet
from models.biomed_UNet.datasets import SegImageDataset

def get_args():
    parser = argparse.ArgumentParser(description='Train the model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str, default='checkpoints/kvasir.pth',
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--img_size', dest='size', type=int, default=512,
                        help='The size of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    
    test_dir = f'../data/2024-12-08-seg2-dataset-{args.size}/u_test_filtered'
    test_set = SegImageDataset(test_dir)
    model = TAUNet.loc.model(TAUNet.loc.config(n_channels=3, n_classes=1, bilinear=True))
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.load}')
    logging.info(f'Using device {device}')

    model.to(device=device)
    state_dict = torch.load(args.load, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    logging.info('Model loaded!')
    #net = UNet(128, 1)
    #net = nn.DataParallel(net, device_ids=[0])
    #net.to(device=device)

    #if args.load:
    #    net.load_state_dict(
    #        torch.load(args.load, map_location=device), False
    #    )
    #    logging.info(f'Model loaded from {args.load}')

    try:
        test_net_seg(
            net = model,
            test_set = test_set,
            batch_size=args.batchsize,
            device=device,
            img_size=args.size
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
