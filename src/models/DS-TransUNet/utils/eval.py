import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


def dice_coeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")

    #pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    
    gt_flat = gt_flat > 0 # foreground only!!

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
    return loss.mean(), N #loss.sum(), N



def eval_net(net, loader, device, n_class=1):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if n_class == 1 else torch.long
    tot = 0
    n_val = len(loader) 
    N = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred, _, _ = net(imgs)

            if n_class > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                l, n = dice_coeff(pred, true_masks)
                tot += l
                N += n
            pbar.update()

    return tot / N


import torch
import torch.nn.functional as F
from tqdm import tqdm

from .scores import multiclass_dice_coeff, dice_coeff, multiclass_jaccard_coeff, calculate_pixelwise_classification_metrics, calculate_objectwise_classification_metrics, calculate_objectwise_classification_metrics2, calculate_objectwise_classification_metrics3, iou, jaccard_coeff, calculate_objectwise_classification_metrics4

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_iou = 0
    total_px_accuracy = 0
    total_px_precision = 0
    total_px_recall = 0
    total_px_f1 = 0
    total_ob_accuracy = 0
    total_ob_precision = 0
    total_ob_recall = 0
    total_ob_f1 = 0
    out_mask_pred = None
    out_mask_true = None
    out_image = None
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch #batch['image'], batch['mask']
            # move images and labels to correct device and type
            #size = 512
            #image = F.upsample(image, size=(size, size), mode='bilinear', align_corners=True)
            #mask_true = F.upsample(mask_true, size=(size, size), mode='bilinear', align_corners=True)
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, _, _ = net(image)
            
            if True:#net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #print('mask_pred max: '+str(mask_pred.max().item()))
                #print(mask_true.shape,mask_pred.shape)
                dice_score += dice_coeff(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False)
                #mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                ## compute the Dice score
                #mask_true = mask_true.unsqueeze(1)
                ##print(mask_pred.shape)
                ##print(mask_true.shape)
                
                
                #dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                #
                #accuracy, precision, recall, f1 = calculate_pixelwise_classification_metrics(
                #    mask_pred,
                #    mask_true
                #)
                #total_px_accuracy += accuracy
                #total_px_precision += precision
                #total_px_recall += recall
                #total_px_f1 += f1
#
                #accuracy, precision, recall, f1 = calculate_objectwise_classification_metrics(
                #    mask_pred.squeeze(1),
                #    mask_true.squeeze(1),
                #    threshold = 0.5
                #)
                #total_ob_accuracy += accuracy
                #total_ob_precision += precision
                #total_ob_recall += recall
                #total_ob_f1 += f1
                #print(mask_true.shape, mask_pred.long().squeeze().shape)
                #print(mask_true.max(), mask_pred.max())
                
                accuracy, precision, recall, f1, tp_mask_batch, fp_mask_batch, fn_mask_batch = calculate_objectwise_classification_metrics4(
                    mask_pred.squeeze(0).long(),#.numpy(),#mask_pred,#.squeeze(1), 
                    mask_true.squeeze(0),
                    threshold = 0.5
                )
                
                total_ob_accuracy += accuracy
                total_ob_precision += precision
                total_ob_recall += recall
                total_ob_f1 += f1
                # compute the Dice score, ignoring background
                #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #dice_score += dice_coeff(
                #    mask_pred.squeeze(1), 
                #    mask_true, 
                #    reduce_batch_first=False
                #)
                #print(mask_true.squeeze(0).size(),mask_pred.squeeze(0).size())
                total_iou += iou(
                    mask_pred.squeeze(0), 
                    mask_true.squeeze(0)
                )
                
                out_mask_pred = mask_pred.squeeze(0)
                out_mask_true = mask_true.squeeze(0)
                out_image = image.squeeze(0)
            #else:
            #    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            #    # convert to one-hot format
            #    #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #    #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #    #mask_pred = mask_pred.argmax(dim=1)[0].long().squeeze()#.numpy()
            #    #mask_true = mask_true.squeeze(0)#.numpy()
            #    
            #    #accuracy, precision, recall, f1 = calculate_pixelwise_classification_metrics(
            #    #    mask_pred,
            #    #    mask_true
            #    #)
            #    #total_px_accuracy += accuracy
            #    #total_px_precision += precision
            #    #total_px_recall += recall
            #    #total_px_f1 += f1
#
            #    accuracy, precision, recall, f1 = calculate_objectwise_classification_metrics3(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(),
            #        mask_true.squeeze(0),
            #        threshold = 0.5
            #    )
            #    total_ob_accuracy += accuracy
            #    total_ob_precision += precision
            #    total_ob_recall += recall
            #    total_ob_f1 += f1
            #    # compute the Dice score, ignoring background
            #    #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            #    dice_score += jaccard_coeff(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(), 
            #        mask_true.squeeze(0), 
            #        reduce_batch_first=False
            #    )
            #    total_iou += iou(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(), 
            #        mask_true.squeeze(0)
            #    )

    net.train()

    avg_dice = dice_score / max(num_val_batches, 1)
    avg_iou = total_iou / max(num_val_batches, 1)
    
    px_avg_accuracy = total_px_accuracy / max(num_val_batches, 1)
    px_avg_precision = total_px_precision / max(num_val_batches, 1)
    px_avg_recall = total_px_recall / max(num_val_batches, 1)
    px_avg_f1 = total_px_f1 / max(num_val_batches, 1)
    
    ob_avg_accuracy = total_ob_accuracy / max(num_val_batches, 1)
    ob_avg_precision = total_ob_precision / max(num_val_batches, 1)
    ob_avg_recall = total_ob_recall / max(num_val_batches, 1)
    ob_avg_f1 = total_ob_f1 / max(num_val_batches, 1)
    
    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'px_accuracy': px_avg_accuracy,
        'px_precision': px_avg_precision,
        'px_recall': px_avg_recall,
        'px_f1': px_avg_f1,
        'ob_accuracy': ob_avg_accuracy,
        'ob_precision': ob_avg_precision,
        'ob_recall': ob_avg_recall,
        'ob_f1': ob_avg_f1,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }
    
    

@torch.inference_mode()
def test_evaluate(net, dataloader, img_size, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    total_iou = 0
    total_px_accuracy = 0
    total_px_precision = 0
    total_px_recall = 0
    total_px_f1 = 0
    total_ob_accuracy = 0
    total_ob_precision = 0
    total_ob_recall = 0
    total_ob_f1 = 0
    out_mask_pred = None
    out_mask_true = None
    out_image = None
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch #batch['image'], batch['mask']
            # move images and labels to correct device and type
            size = img_size
            image = F.upsample(image, size=(size, size), mode='bilinear', align_corners=True)
            mask_true = F.upsample(mask_true, size=(size, size), mode='bilinear', align_corners=True)
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, _, _ = net(image)
            
            if True:#net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                #print('mask_pred max: '+str(mask_pred.max().item()))
                #print(mask_true.shape,mask_pred.shape)
                dice_score += dice_coeff(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False)
                #mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
                ## compute the Dice score
                #mask_true = mask_true.unsqueeze(1)
                ##print(mask_pred.shape)
                ##print(mask_true.shape)
                
                
                #dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                #
                #accuracy, precision, recall, f1 = calculate_pixelwise_classification_metrics(
                #    mask_pred,
                #    mask_true
                #)
                #total_px_accuracy += accuracy
                #total_px_precision += precision
                #total_px_recall += recall
                #total_px_f1 += f1
#
                #accuracy, precision, recall, f1 = calculate_objectwise_classification_metrics(
                #    mask_pred.squeeze(1),
                #    mask_true.squeeze(1),
                #    threshold = 0.5
                #)
                #total_ob_accuracy += accuracy
                #total_ob_precision += precision
                #total_ob_recall += recall
                #total_ob_f1 += f1
                #print(mask_true.shape, mask_pred.long().squeeze().shape)
                #print(mask_true.max(), mask_pred.max())
                
                accuracy, precision, recall, f1, tp_mask_batch, fp_mask_batch, fn_mask_batch = calculate_objectwise_classification_metrics4(
                    mask_pred.squeeze(0).long(),#.numpy(),#mask_pred,#.squeeze(1), 
                    mask_true.squeeze(0),
                    threshold = 0.5
                )
                
                total_ob_accuracy += accuracy
                total_ob_precision += precision
                total_ob_recall += recall
                total_ob_f1 += f1
                # compute the Dice score, ignoring background
                #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                #dice_score += dice_coeff(
                #    mask_pred.squeeze(1), 
                #    mask_true, 
                #    reduce_batch_first=False
                #)
                #print(mask_true.squeeze(0).size(),mask_pred.squeeze(0).size())
                total_iou += iou(
                    mask_pred.squeeze(0), 
                    mask_true.squeeze(0)
                )
                
                out_mask_pred = mask_pred.squeeze(0)
                out_mask_true = mask_true.squeeze(0)
                out_image = image.squeeze(0)
            #else:
            #    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            #    # convert to one-hot format
            #    #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            #    #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            #    #mask_pred = mask_pred.argmax(dim=1)[0].long().squeeze()#.numpy()
            #    #mask_true = mask_true.squeeze(0)#.numpy()
            #    
            #    #accuracy, precision, recall, f1 = calculate_pixelwise_classification_metrics(
            #    #    mask_pred,
            #    #    mask_true
            #    #)
            #    #total_px_accuracy += accuracy
            #    #total_px_precision += precision
            #    #total_px_recall += recall
            #    #total_px_f1 += f1
#
            #    accuracy, precision, recall, f1 = calculate_objectwise_classification_metrics3(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(),
            #        mask_true.squeeze(0),
            #        threshold = 0.5
            #    )
            #    total_ob_accuracy += accuracy
            #    total_ob_precision += precision
            #    total_ob_recall += recall
            #    total_ob_f1 += f1
            #    # compute the Dice score, ignoring background
            #    #dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
            #    dice_score += jaccard_coeff(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(), 
            #        mask_true.squeeze(0), 
            #        reduce_batch_first=False
            #    )
            #    total_iou += iou(
            #        mask_pred.argmax(dim=1)[0].long().squeeze(), 
            #        mask_true.squeeze(0)
            #    )

    net.train()

    avg_dice = dice_score / max(num_val_batches, 1)
    avg_iou = total_iou / max(num_val_batches, 1)
    
    px_avg_accuracy = total_px_accuracy / max(num_val_batches, 1)
    px_avg_precision = total_px_precision / max(num_val_batches, 1)
    px_avg_recall = total_px_recall / max(num_val_batches, 1)
    px_avg_f1 = total_px_f1 / max(num_val_batches, 1)
    
    ob_avg_accuracy = total_ob_accuracy / max(num_val_batches, 1)
    ob_avg_precision = total_ob_precision / max(num_val_batches, 1)
    ob_avg_recall = total_ob_recall / max(num_val_batches, 1)
    ob_avg_f1 = total_ob_f1 / max(num_val_batches, 1)
    
    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'px_accuracy': px_avg_accuracy,
        'px_precision': px_avg_precision,
        'px_recall': px_avg_recall,
        'px_f1': px_avg_f1,
        'ob_accuracy': ob_avg_accuracy,
        'ob_precision': ob_avg_precision,
        'ob_recall': ob_avg_recall,
        'ob_f1': ob_avg_f1,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }