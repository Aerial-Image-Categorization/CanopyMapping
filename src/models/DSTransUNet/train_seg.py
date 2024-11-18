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

#from utils.eval import eval_net
from .lib.DS_TransUNet import UNet

from torch.utils.data import DataLoader, random_split
from .utils.dataloader import get_loader,test_dataset

from ..utils.earlystopping import EarlyStopping

from ..biomed_UNet.datasets import SegImageDataset
#train_img_dir = 'data/Kvasir_SEG/train/image/'x
#train_mask_dir = 'data/Kvasir_SEG/train/mask/'x
#val_img_dir = 'data/Kvasir_SEG/val/images/'x
#val_mask_dir = 'data/Kvasir_SEG/val/masks/'x
#dir_checkpoint = 'checkpoints/'x


import wandb
#from ..utils import evaluate_transunet
from ..utils.evaluation_transunet import evaluate_transunet_seg
from ..utils import dice_loss
from ..utils import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import wandb

from ..utils.losses import TreeSpotLocalizationLoss


def conf_matrix(
    TP,
    FP,
    FN,
    TN,
    class_labels = ["Negative", "Positive"],
    xlabel = "Predicted Label",
    ylabel = "True Label",
    title = 'Binary Confusion Matrix'
):
    cm = np.array([[TN, FP],
                   [FN, TP]])
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")

    plt.colorbar(cax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    return fig

def precision_recall_curve(precision, recall):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker='.', label='PR Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.title('Precision-Recall Curve')
    ax.legend()
    ax.grid(True)

    return fig



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
    #weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    
    #from border to center weights
    weit = 1 + F.avg_pool2d(mask, kernel_size=13, stride=1, padding=6)
    weit = torch.where(weit < 1.2, torch.tensor(1.0, dtype=weit.dtype, device=weit.device), weit/1.2)
    #
    
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

    has_foreground_mask = mask.sum(dim=(1, 2, 3)) > 0
    has_foreground_pred = pred.sum(dim=(1, 2, 3)) > 0

    if not has_foreground_mask.any() and not has_foreground_pred.any():
        loss = (wbce + wiou).mean() * 0
    else:
        loss = (wbce + wiou)[has_foreground_mask].mean()
    return loss


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def structure_loss_with_0_no_weigths(pred, mask):
    #wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    #return (wbce + wiou).mean()
    return wiou.mean()

def train_net(net,
              train_img_dir,
              train_mask_dir,
              val_img_dir,
              val_mask_dir,
              dir_checkpoint,
              device,
              epochs=500,
              batch_size=16,
              lr=1e-5,
              save_cp=True,
              n_class=1,
              img_size=512):

    #train_loader = get_loader(train_img_dir, train_mask_dir, batchsize=batch_size, trainsize=img_size, augmentation = False)
    #val_loader = get_loader(val_img_dir, val_mask_dir, batchsize=1, trainsize=img_size, augmentation = False)

    train_set = SegImageDataset(train_img_dir)
    valid_set = SegImageDataset(val_img_dir)
    train_loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    val_loader = DataLoader(valid_set, shuffle=True, drop_last=False, **val_loader_args)
    
    #n_train = cal(train_loader)
    #n_val = cal(val_loader)
    n_train = len(train_set)
    n_val = len(valid_set)
    logger = get_logger(f'ds_transunet_seg_base_{img_size}.log')
    #logger = get_logger(f'ds_transunet_base_1024.log')

    # (Initialize logging)
    name=f'ds_transunet_seg_base_{img_size}_u10'#opt_sch'
    if img_size>512:
        img_size = 512
    save_checkpoint=True
    img_scale=1
    amp=False
    experiment = wandb.init(project='CanopyMapping', resume='allow', anonymous='must',name=name, magic=True)
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

    #optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs//5, lr/10)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, lr/10, last_epoch=-1)

    optimizer = optim.RMSprop(net.parameters(),
                              lr=lr, weight_decay=1e-4, momentum=0.9, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    if n_class > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()


    best_dice = 0
    #size_rates = [384, 512, 640]
    size_rates = [img_size]
    global_step = 0
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode="max")
    #loss_function = TreeSpotLocalizationLoss(w_dice_weight=0, tversky_weight=1.0, soft_l2_weight=0, tversky_alpha=0.3)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        b_cp = False
        Batch = len(train_loader)
        with tqdm(total=n_train*len(size_rates), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                for rate in size_rates:
                    #imgs, true_masks = batch
                    imgs, true_masks = batch['image'], batch['mask']
                    trainsize = rate
                    if rate != img_size:
                        imgs = F.upsample(imgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                        true_masks = F.upsample(true_masks, size=(trainsize, trainsize), mode='bilinear', align_corners=True)


                    imgs = imgs.to(device=device, dtype=torch.float32)
                    mask_type = torch.float32 if n_class == 1 else torch.long
                    true_masks = true_masks.to(device=device, dtype=mask_type)
                    #print('mask_pred max: '+str(mask_pred.max().item()))
                    #print(mask_true.shape,mask_pred.shape)
                    #dice_score.append(dice(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))
                
                    masks_pred, l2, l3 = net(imgs)
                    #masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
                    ##masks_pred = masks_pred.squeeze(1)
                    true_masks = true_masks.unsqueeze(0)
                    loss1 = structure_loss_with_0_no_weigths(masks_pred, true_masks)
                    loss2 = structure_loss_with_0_no_weigths(l2, true_masks)
                    loss3 = structure_loss_with_0_no_weigths(l3, true_masks)
                    #loss = 0.6*loss1 + 0.2*loss2 + 0.2*loss3# + loss_function(torch.sigmoid(masks_pred.squeeze(0)), true_masks.float())
                    loss = 0.6*loss1 + 0.2*loss2 + 0.2*loss3# + 0.2*loss_function(torch.sigmoid(masks_pred.squeeze(0)), true_masks.float())
                    
                    #loss1 = 0.1 * criterion(masks_pred.squeeze(1), true_masks.float()) + 0.9 * loss_function(torch.sigmoid(masks_pred.squeeze(1)), true_masks.float())
                    #loss2 = 0.1 * criterion(masks_pred.squeeze(1), true_masks.float()) + 0.9 * loss_function(torch.sigmoid(l2.squeeze(1)), true_masks.float())
                    #loss3 = 0.1 * criterion(masks_pred.squeeze(1), true_masks.float()) + 0.9 * loss_function(torch.sigmoid(l3.squeeze(1)), true_masks.float())
                    #print(masks_pred.squeeze(0).size(), true_masks.size())
                    #loss1 = loss_function(torch.sigmoid(masks_pred.squeeze(0)), true_masks.float())
                    #loss2 = loss_function(torch.sigmoid(l2.squeeze(0)), true_masks.float())
                    #loss3 = loss_function(torch.sigmoid(l3.squeeze(0)), true_masks.float())
                    
                    #DEBUG
                    #print(masks_pred.requires_grad, l2.requires_grad, l3.requires_grad) 
                    #loss = 0.1 * criterion(masks_pred, true_masks.float()) + 0.9*loss2# + 0.1*loss2 + 0.1*loss3
                    
                    assert loss.requires_grad, "Loss tensor does not require gradients."
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

                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        #val_dice = eval_net(net, val_loader, device)
                        val_score = evaluate_transunet_seg(net, val_loader, device, epoch, False)

                        #if isinstance(val_score['dice_score'], torch.Tensor) and val_score['dice_score'].requires_grad:
                        #    raise RuntimeError("In-place operations or tensors that require gradients are not allowed in scheduler.step().")
                        #
                        #if not isinstance(val_score['dice_score'], (float, int)):
                        #    raise ValueError(f"Invalid value for dice_score: {val_score['dice_score']} -> {type(val_score['dice_score'])}")

                        scheduler.step(val_score['dice_score'].item())
                        
                        
                        val_dice = val_score['dice_score']
                        if val_dice > best_dice:
                           best_dice = val_dice
                           b_cp = True
                        epoch_loss = epoch_loss / Batch
                        logger.info('epoch: {} train_loss: {:.3f} epoch_dice: {:.3f}, best_dice: {:.3f}'.format(epoch + 1, epoch_loss, val_dice, best_dice))
                        #print(y_true)
                        #print(preds)
                        try:
                            wandb_log_data = {
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'Segmentation metrics': {
                                    'Dice': val_dice, #val_score['dice_score'],
                                    'IoU':val_score['iou'],
                                    #'Weighted IoU': val_score['w_iou'],
                                    #'Weighted Dice': val_score['w_dice']
                                },
                                #'validation Dice': val_score['dice_score'],
                                #'validation IoU': val_score['iou'],
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
                            }
                            experiment.log(wandb_log_data)
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
    
    train_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train_u10/images/'
    train_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/aug_train_u10/masks/'
    val_img_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/images/'
    val_mask_dir = f'../data/2024-10-30-loc-dataset-{args.size}/val/masks/'
    dir_checkpoint = f'ds_transunet_checkpoints_{args.size}_u10_man/'
    
    
    
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
                  img_size=args.size#512 ## TODO !!! args.size
                  
        ) 
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
