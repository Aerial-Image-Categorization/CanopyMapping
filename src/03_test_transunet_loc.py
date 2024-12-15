import sys
import os

import argparse
import torch
import logging
from models.DSTransUNet.lib.DS_TransUNet import UNet
from models.DSTransUNet.test_loc import test_net


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
    
    
    test_dir = f'../data/2024-10-30-loc-dataset-{args.size}/u_test'
    
    #net = UNet(128, 1)

    net = UNet(64, 1)
    #net = nn.DataParallel(net, device_ids=[0])
    net = net.to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
    logging.info(f'Model loaded from {args.load}')

    #net = UNet(128, 1)
    #net = nn.DataParallel(net, device_ids=[0])
    #net.to(device=device)

    #if args.load:
    #    net.load_state_dict(
    #        torch.load(args.load, map_location=device), False
    #    )
    #    logging.info(f'Model loaded from {args.load}')

    try:
        test_net(
            net = net,
            batch_size=args.batchsize,
            device=device,
            img_size=args.size
        )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
