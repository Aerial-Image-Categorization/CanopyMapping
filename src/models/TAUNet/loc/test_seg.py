import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm


from torch.utils.data import DataLoader, random_split
from PIL import Image

from ...utils.evaluation_seg import evaluate_seg

import wandb

def test_net_seg(
    net,
    test_set,
    device,
    batch_size=1,
    n_class=1,
    img_size=512):

    wandb.init(project="CanopyMapping", resume='allow', anonymous='must',name=f'test_taunet_{img_size}', magic=True)

    #val_img_dir = f'../data/2024-10-30-loc-dataset-{img_size}/test/images/'#'data/Kvasir_SEG/val/images/'
    #val_mask_dir = f'../data/2024-10-30-loc-dataset-{img_size}/test/masks/'#'data/Kvasir_SEG/val/masks/'
    
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)


    #val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    net.eval()

    #eval_net(net, val_loader, device)
    results = evaluate_seg(net, test_loader, device, 6, False)
    wandb.log({
        'Segmentation metrics': {
            'Dice': results['dice_score'],
            'IoU':results['iou'],
        },
        'test': {
            'images': wandb.Image(results['image'].cpu()),
            'masks': {
                'true': wandb.Image(results['mask_true'].float().cpu()),
                'pred': wandb.Image(results['mask_pred'].float().cpu()),
            }
        },
    })