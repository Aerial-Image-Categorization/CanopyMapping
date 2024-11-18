import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .scores import dice, weighted_dice, iou, weighted_iou, objectwise_classification_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@torch.inference_mode()
def evaluate_seg(net, dataloader, device, epoch, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    dice_score = []
    total_iou = []

    out_mask_pred = None
    out_mask_true = None
    out_image = None
    #total_weighted_dice_score = []
    #total_weighted_iou_score = []
    
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            #image, mask_true = batch
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            #size = 512
            #image = F.upsample(image, size=(size, size), mode='bilinear', align_corners=True)
            #mask_true = F.upsample(mask_true, size=(size, size), mode='bilinear', align_corners=True)
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            #mask_pred, _, _ = net(image)
            mask_pred = net(image)
            
            if True:#net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #print('mask_pred max: '+str(mask_pred.max().item()))
                #print(mask_true.shape,mask_pred.shape)
                dice_score.append(dice(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))
                
                #total_weighted_dice_score.append(weighted_dice(mask_pred.squeeze(), mask_true.squeeze()))
                #total_weighted_iou_score.append(weighted_iou(mask_pred.squeeze(), mask_true.squeeze()))

                # 'accuracy'
                # 'precision'
                # 'recall'
                # 'f1'
                # 'mean_iou'
                # 'true_positive_count'
                # 'false_positive_count'
                # 'false_negative_count'
                # 'tp_mask_batch'
                # 'fp_mask_batch'
                # 'fn_mask_batch'
                # 'predictions'
                # 'ground_truth'
                #
                

                total_iou.append(iou(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))
                
                out_mask_pred = mask_pred.squeeze(0)
                out_mask_true = mask_true.squeeze(0)
                out_image = image.squeeze(0)

    net.train()
    #raise RuntimeError('asdf')
    avg_dice = np.mean([x.cpu().numpy() for x in dice_score if x is not None]) #/ max(num_val_batches, 1)
    avg_iou = np.mean([x.cpu().numpy() for x in total_iou if x is not None]) #/ max(num_val_batches, 1)
    #avg_w_dice = np.mean([x.cpu().numpy() for x in total_weighted_dice_score if x is not None]) #/ max(num_val_batches, 1)
    #avg_w_iou = np.mean([x.cpu().numpy() for x in total_weighted_iou_score if x is not None]) #/ max(num_val_batches, 1)

    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        #'w_dice': avg_w_dice,
        #'w_iou': avg_w_iou,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }