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

#from .evaluate import evaluate
from .datasets import ImageDataset
#from .scores import dice_loss, jaccard_loss

import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../')

from ..utils import evaluate
from ..utils import dice_loss
from ..utils import EarlyStopping

import numpy as np

import datetime

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
#from lib.DS_TransUNet import UNet
#
#from torch.utils.data import DataLoader, random_split
#from utils.dataloader import get_loader,test_dataset

#train_img_dir = 'data/Kvasir_SEG/train/image/'x
#train_mask_dir = 'data/Kvasir_SEG/train/mask/'x
#val_img_dir = 'data/Kvasir_SEG/val/images/'x
#val_mask_dir = 'data/Kvasir_SEG/val/masks/'x
#dir_checkpoint = 'checkpoints/'x

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

import wandb
#from utils.eval import evaluate

import numpy as np
import matplotlib.pyplot as plt
import wandb

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

def train_net_loc(
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
    name=f'loc-unet-biomed-{img_size}'
    dir_checkpoint = Path(f'./checkpoints_{name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")}')
    
    if img_size > 512:
        img_size = 512
    
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
    experiment = wandb.init(project='CanopyMapping', resume='allow', anonymous='must',name=name, magic=True)
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
                           
                        val_predictions = np.zeros((len(np.array(val_score['predictions']['prob'])), 2))
                        val_predictions[:, 1] = np.array(val_score['predictions']['prob'])
                        val_predictions[:, 0] = 1 - val_predictions[:, 1]
                        
                        y_true = val_score['ground_truth'].cpu().numpy().tolist() if isinstance(val_score['ground_truth'], torch.Tensor) else val_score['ground_truth']
                        preds = val_score['predictions']['label'].cpu().numpy().tolist() if isinstance(val_score['predictions']['label'], torch.Tensor) else val_score['predictions']['label']

                        
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
                            if epoch > 5:
                                wandb_log_data.update({
                                'Classification metrics': {
                                    'obj. IoU 50': val_score['obj_iou_50'],
                                    'Accuracy 50': val_score['ob_accuracy_50'],
                                    'Precision 50': val_score['ob_precision_50'],
                                    'Recall 50': val_score['ob_recall_50'],
                                    'F1-score 50': val_score['ob_f1_50'],
                                    'Weighted obj. IoU 50': val_score['obj_w_iou_50'],
                                    'Weighted Accuracy 50': val_score['ob_w_accuracy_50'],
                                    'Weighted Precision 50': val_score['ob_w_precision_50'],
                                    'Weighted Recall 50': val_score['ob_w_recall_50'],
                                    'Weighted F1-score 50': val_score['ob_w_f1_50'],
                                    'Weighted obj. IoU 25': val_score['obj_w_iou_25'],
                                    'Weighted Accuracy 25': val_score['ob_w_accuracy_25'],
                                    'Weighted Precision 25': val_score['ob_w_precision_25'],
                                    'Weighted Recall 25': val_score['ob_w_recall_25'],
                                    'Weighted F1-score 25': val_score['ob_w_f1_25'],
                                    #
                                    #"ROC": roc_curve(y_true,preds),
                                    #"ROC-AUC": roc_auc_score(y_true,preds),
                                    #"Precision-Recall": precision_recall_curve(y_true,preds),
                                    #"Average Precision": average_precision_score(y_true,preds)
                                    #
                                    #'Weighted obj. IoU 50-25': [val_score['obj_w_iou_50'],val_score['obj_w_iou_25']],
                                    #'Weighted Accuracy 50-25': [val_score['ob_w_accuracy_50'],val_score['ob_w_accuracy_25']],
                                    #'Weighted Precision 50-25': [val_score['ob_w_precision_50'],val_score['ob_w_precision_25']],
                                    #'Weighted Recall 50-25': [val_score['ob_w_recall_50'],val_score['ob_w_recall_25']],
                                    #'Weighted F1-score 50-25': [val_score['ob_w_f1_50'],val_score['ob_w_f1_25']],
                                },
                                "Confusion Matrix": wandb.plot.confusion_matrix(
                                    probs=None,
                                    y_true=y_true,
                                    preds=preds,
                                    class_names=['background', 'tree']
                                ),
                                "Precision-Recall Curve": wandb.plot.pr_curve(
                                    np.array(val_score['ground_truth']),
                                    val_predictions,
                                    labels=["background", "tree"]
                                ),
                                "ROC curve": wandb.plot.roc_curve(
                                    np.array(val_score['ground_truth']),
                                    val_predictions,
                                    labels=["background", "tree"]
                                ),
                                })
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