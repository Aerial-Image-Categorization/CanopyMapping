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

from utils.eval import eval_net
from lib.DS_TransUNet import UNet

from torch.utils.data import DataLoader, random_split
from utils.dataloader import get_loader,test_dataset

from utils.earlystopping import EarlyStopping

import wandb
from utils.eval import evaluate
from utils.scores import jaccard_loss

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def cal(loader):
    tot = 0
    for batch in loader:
        imgs, _ = batch
        tot += imgs.shape[0]
    return tot

def structure_loss_with_0(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def structure_loss(pred, mask, epsilon=1e-6):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + epsilon) / (union - inter + epsilon)

    has_foreground = mask.sum(dim=(1, 2, 3)) > 0

    if has_foreground.any():
        loss = (wbce + wiou)[has_foreground].mean()
    else:
        loss = (wbce + wiou).mean() * 0 #torch.tensor(0.0, device=torch.device('cuda'))#pred.device)
    return loss


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def train_net(net,
              train_img_dir,
              train_mask_dir,
              val_img_dir,
              val_mask_dir,
              dir_checkpoint,
              device,
              epochs=500,
              batch_size=16,
              lr=0.01,
              save_cp=True,
              n_class=1,
              img_size=512):

    train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=1, trainsize=img_size, augmentation = False)

    n_train = cal(train_loader)
    n_val = cal(val_loader)
    logger = get_logger(f'ds_transunet_base_{img_size}.log')

    # (Initialize logging)
    name=f'ds_transunet_base_{img_size}'#opt_sch'
    save_checkpoint=True
    img_scale=1
    amp=False
    experiment = wandb.init(project='TreeDetection', resume='allow', anonymous='must',name=name, magic=True)
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=lr,
             val_percent=n_val, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logger.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Vailding size:   {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images size:  {img_size}
    ''')

    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//5, lr/10)
    
    #optimizer = optim.RMSprop(net.parameters(),
    #                          lr=lr, weight_decay=1e-4, momentum=0.9, foreach=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()


    best_dice = 0
    #size_rates = [384, 512, 640]
    size_rates = [img_size]
    global_step = 0
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode="max")
    
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        b_cp = False
        Batch = len(train_loader)
        with tqdm(total=n_train*len(size_rates), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates:
                    imgs, true_masks = batch
                    trainsize = rate
                    if rate != img_size:
                        imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)


                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)

                    masks_pred, l2, l3 = net(imgs)
                    loss1 = structure_loss(masks_pred, true_masks)
                    loss2 = structure_loss(l2, true_masks)
                    loss3 = structure_loss(l3, true_masks)
                    loss = 0.6*loss1 + 0.2*loss2 + 0.2*loss3
                    
                    #loss = jaccard_loss(masks_pred.squeeze(1), true_masks.squeeze(1).float(), multiclass=False)
                    
                    epoch_loss += loss.item()
                    
                    
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(net.parameters(), 0.1)
                    optimizer.step()

                    pbar.update(imgs.shape[0])
                    global_step += 1

                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        #val_dice = eval_net(net, val_loader, device)
                        val_score = evaluate(net, val_loader, device, False)
                        
                        scheduler.step(val_score['dice_score'])
                        val_dice = val_score['dice_score']
                        if val_dice > best_dice:
                           best_dice = val_dice
                           b_cp = True
                        epoch_loss = epoch_loss / Batch
                        logger.info('epoch: {} train_loss: {:.3f} epoch_dice: {:.3f}, best_dice: {:.3f}'.format(epoch + 1, epoch_loss, val_dice, best_dice))

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'Segmentation metrics': {
                                    'Dice': val_dice, #val_score['dice_score'],
                                    'IoU':val_score['iou']
                                },
                                #'validation Dice': val_score['dice_score'],
                                #'validation IoU': val_score['iou'],
                                'Classification metrics': {
                                    'Accuracy': val_score['ob_accuracy'],
                                    'Precision': val_score['ob_precision'],
                                    'Recall': val_score['ob_recall'],
                                    'F1-score': val_score['ob_f1']
                                },
                                'train': {
                                    'images': wandb.Image(imgs[0].cpu()),
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
                            })
                        except Exception as e:
                            print(f'error {e}')
                            pass
        
        early_stopping(val_score["dice_score"])
        
        if (save_cp and b_cp) or early_stopping.early_stop:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'epoch:{}_dice:{:.3f}.pth'.format(epoch + 1, val_dice*100))
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
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
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
    
    train_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train/images/'
    train_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train/masks/'
    val_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/images/'
    val_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/masks/'
    dir_checkpoint = f'ds_transunet_checkpoints_{args.size}/'
    
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(128, 1)
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
                  lr=args.lr,
                  device=device,
                  img_size=args.size)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
