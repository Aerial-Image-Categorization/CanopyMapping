import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
#from .evaluate import evaluate
#from .datasets import ImageDataset
#from .scores import dice_loss, jaccard_loss
from ..utils import evaluate
from ..utils import dice_loss
from ..utils import EarlyStopping

import datetime

import numpy as np

def train_model_Jaccard(
        dataset,
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    name='unet-centroid-standard-iou-b6'
    dir_checkpoint = Path(f'./checkpoints_{name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model = model.to(memory_format=torch.channels_last)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='TreeDetection', resume='allow', anonymous='must',name=name, magic=True)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        #loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        #loss += jaccard_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        loss = jaccard_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        #loss = criterion(masks_pred, true_masks)
                        #loss += jaccard_loss(
                        #    F.softmax(masks_pred, dim=1).float(),
                        #    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #    multiclass=True
                        #)
                        loss = jaccard_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation IoU score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation IoU': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            
            
def train_net_seg(
        train_set,
        valid_set,
        model,
        device,
        img_size=512,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    name=f'seg-unet-centroid-{img_size}'
    dir_checkpoint = Path(f'./checkpoints_{name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')
    
    #if img_size > 512:
    #    img_size = 512
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    model = model.to(memory_format=torch.channels_last)
    
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 2. Split into train / validation partitions
    #n_val = int(len(dataset) * val_percent)
    #n_train = len(dataset) - n_val
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    n_train = len(train_set)
    n_val = len(valid_set)

    # 3. Create data loaders
    train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    val_loader = DataLoader(valid_set, shuffle=True, drop_last=False, **val_loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='TreeDetection', resume='allow', anonymous='must',name=name, magic=True)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    best_dice = 0
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode="max")
    
    # 5. Begin training
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        b_cp = False
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        #loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        loss += dice_loss(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        #loss = criterion(masks_pred, true_masks)
                        #loss += jaccard_loss(
                        #    F.softmax(masks_pred, dim=1).float(),
                        #    F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #    multiclass=True
                        #)
                        loss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, epoch, amp)

                        scheduler.step(val_score['dice_score'].item())

                        #logging.info(f"""Validation scores:
                        #- IoU: {val_score['dice_score']:.4f}
                        #pixel-wise:
                        #- Accuracy: {val_score['px_accuracy']:.4f}
                        #- Precision: {val_score['px_precision']:.4f}
                        #- Recall: {val_score['px_recall']:.4f}
                        #- F1: {val_score['px_f1']:.4f}
                        #object-wise:
                        #- Accuracy: {val_score['ob_accuracy']:.4f}
                        #- Precision: {val_score['ob_precision']:.4f}
                        #- Recall: {val_score['ob_recall']:.4f}
                        #- F1: {val_score['ob_f1']:.4f}""")
                        
                        val_dice = val_score['dice_score']
                        if val_dice > best_dice:
                           best_dice = val_dice
                           b_cp = True
                        
                        try:
                            wandb_log_data = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'Segmentation metrics': {
                                    'Dice': val_dice, #val_score['dice_score'],
                                    'IoU':val_score['iou'],
                                    'Weighted IoU': val_score['w_iou'],
                                    'Weighted Dice': val_score['w_dice']
                                },
                                #'validation Dice': val_score['dice_score'],
                                #'validation IoU': val_score['iou'],
                                'train': {
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image((torch.sigmoid(masks_pred[0]) > 0.5).float().cpu()),
                                    }
                                },
                                'validation': {
                                    'images': wandb.Image(val_score['image'].cpu()),
                                    'masks': {
                                        'true': wandb.Image(val_score['mask_true'].float().cpu()),
                                        'pred': wandb.Image(val_score['mask_pred'].float().cpu()),
                                    }
                                },
                                'step': global_step,
                                'epoch': epoch,
                                #**histograms
                            }
                            experiment.log(wandb_log_data)
                        except Exception as e:
                            print(f'error {e}')
                            pass
        
        early_stopping(val_score["dice_score"])
        
        if (save_checkpoint and b_cp) or early_stopping.early_stop:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

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


#if __name__ == '__main__':
#    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#    args = get_args()
#    
#    train_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train_u10/images/'
#    train_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train_u10/masks/'
#    val_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/images/'
#    val_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/masks/'
#    dir_checkpoint = f'ds_transunet_checkpoints_{args.size}_u10_man/'
#    
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    logging.info(f'Using device {device}')
#
#    net = UNet(128, 1)
#    net = nn.DataParallel(net, device_ids=[0])
#    net = net.to(device)
#
#    if args.load:
#        net.load_state_dict(
#            torch.load(args.load, map_location=device)
#        )
#    logging.info(f'Model loaded from {args.load}')
#
#    try:
#        train_net(net=net,
#                  train_img_dir = train_img_dir,
#                  train_mask_dir = train_mask_dir,
#                  val_img_dir = val_img_dir,
#                  val_mask_dir = val_mask_dir,
#                  dir_checkpoint = dir_checkpoint,
#                  epochs=args.epochs,
#                  batch_size=args.batchsize,
#                  lr=args.lr,
#                  device=device,
#                  img_size=args.size#512 ## TODO !!! args.size
#                  
#        ) 
#    except KeyboardInterrupt:
#        torch.save(net.state_dict(), 'INTERRUPTED.pth')
#        logging.info('Saved interrupt')
#        try:
#            sys.exit(0)
#        except SystemExit:
#            os._exit(0)
#