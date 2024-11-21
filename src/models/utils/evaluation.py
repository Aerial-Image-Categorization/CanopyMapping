import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .scores import dice, weighted_dice, iou, weighted_iou, objectwise_classification_metrics, weighted_dice_opt, weighted_iou_opt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@torch.inference_mode()
def evaluate(net, dataloader, device, epoch, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    dice_score = []
    total_iou = []

    total_ob_50_preds = {
        'label': [],
        'prob': []
    }
    total_ob_50_ground_truth=[]

    out_mask_pred = None
    out_mask_true = None
    out_image = None
    total_weighted_dice_score = []
    total_weighted_iou_score = []
    
    total_ob_iou = []
    total_ob_w_iou_50 = []
    total_ob_w_iou_25 = []
    
    total_ob_w_50_preds = {
        'label': [],
        'prob': []
    }
    total_ob_w_50_ground_truth=[]
    
    total_ob_w_25_preds = {
        'label': [],
        'prob': []
    }
    total_ob_w_25_ground_truth=[]
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    total_preds = {
        'label': [],
        'prob': []
    }
    total_ground_truth=[]
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
                #print(mask_true.size(), mask_pred.size())
                total_weighted_dice_score.append(weighted_dice_opt(mask_pred.squeeze(1), mask_true.float()))
                total_weighted_iou_score.append(weighted_iou_opt(mask_pred.squeeze(1), mask_true.float()))
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
                if epoch > 5:
                    obj_cl_results_50 = objectwise_classification_metrics(
                        mask_pred.squeeze(0).long(),#.numpy(),#mask_pred,#.squeeze(1), 
                        mask_true,#.squeeze(0),
                        threshold = 0.5,
                        weighted=False
                    )

                    obj_w_cl_results_50 = objectwise_classification_metrics(
                        mask_pred.squeeze(0).long(),#.numpy(),#mask_pred,#.squeeze(1), 
                        mask_true,#.squeeze(0),
                        threshold = 0.5,
                        weighted = True,
                        sigma = 10
                    )

                    obj_w_cl_results_25 = objectwise_classification_metrics(
                        mask_pred.squeeze(0).long(),#.numpy(),#mask_pred,#.squeeze(1), 
                        mask_true,#.squeeze(0),
                        threshold = 0.25,
                        weighted = True,
                        sigma = 10
                    )

                    total_ob_50_preds['label'].extend(obj_cl_results_50['predictions']['label'])
                    total_ob_50_preds['prob'].extend(obj_cl_results_50['predictions']['prob'])
                    total_ob_50_ground_truth += obj_cl_results_50['ground_truth']
                    total_ob_iou.append(obj_cl_results_50['mean_iou'])

                    total_ob_w_50_preds['label'].extend(obj_w_cl_results_50['predictions']['label'])
                    total_ob_w_50_preds['prob'].extend(obj_w_cl_results_50['predictions']['prob'])
                    total_ob_w_50_ground_truth += obj_w_cl_results_50['ground_truth']
                    total_ob_w_iou_50.append(obj_w_cl_results_50['mean_iou'])

                    #total_tp += obj_w_cl_results_50['true_positive_count']
                    #total_fp += obj_w_cl_results_50['false_positive_count']
                    #total_fn += obj_w_cl_results_50['false_negative_count']
                    
                    total_ob_w_25_preds['label'].extend(obj_w_cl_results_25['predictions']['label'])
                    total_ob_w_25_preds['prob'].extend(obj_w_cl_results_25['predictions']['prob'])
                    total_ob_w_25_ground_truth += obj_w_cl_results_25['ground_truth']
                    total_ob_w_iou_25.append(obj_w_cl_results_25['mean_iou'])

                total_iou.append(iou(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))
                
                out_mask_pred = mask_pred.squeeze(0)
                out_mask_true = mask_true.squeeze(0)
                out_image = image.squeeze(0)

    net.train()
    #raise RuntimeError('asdf')
    avg_dice = np.mean([x.cpu().numpy() for x in dice_score if x is not None]) #/ max(num_val_batches, 1)
    avg_iou = np.mean([x.cpu().numpy() for x in total_iou if x is not None]) #/ max(num_val_batches, 1)
    avg_w_dice = np.mean([x.cpu().numpy() for x in total_weighted_dice_score if x is not None]) #/ max(num_val_batches, 1)
    avg_w_iou = np.mean([x.cpu().numpy() for x in total_weighted_iou_score if x is not None]) #/ max(num_val_batches, 1)
    
    ob_avg_iou = np.mean([x for x in total_ob_iou if x is not None]) if epoch > 5 else 0 # / max(num_val_batches, 1)
    ob_avg_w_iou_50 = np.mean([x for x in total_ob_w_iou_50 if x is not None]) if epoch > 5 else 0#/ max(num_val_batches, 1)
    ob_avg_w_iou_25 = np.mean([x for x in total_ob_w_iou_25 if x is not None]) if epoch > 5 else 0# / max(num_val_batches, 1)
    
    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'w_dice': avg_w_dice,
        'w_iou': avg_w_iou,
        'obj_iou_50': ob_avg_iou if epoch > 5 else 0,
        'obj_w_iou_50': ob_avg_w_iou_50 if epoch > 5 else 0,
        'obj_w_iou_25': ob_avg_w_iou_25 if epoch > 5 else 0,
        #'px_accuracy': px_avg_accuracy,
        #'px_precision': px_avg_precision,
        #'px_recall': px_avg_recall,
        #'px_f1': px_avg_f1,
        'ob_accuracy_50': accuracy_score(total_ob_50_ground_truth, total_ob_50_preds['label'])if epoch > 5 else 0,#
        'ob_precision_50': precision_score(total_ob_50_ground_truth, total_ob_50_preds['label'])if epoch > 5 else 0,#
        'ob_recall_50': recall_score(total_ob_50_ground_truth, total_ob_50_preds['label'])if epoch > 5 else 0,#
        'ob_f1_50': f1_score(total_ob_50_ground_truth, total_ob_50_preds['label'])if epoch > 5 else 0,#
        'ob_w_accuracy_50': accuracy_score(total_ob_w_50_ground_truth, total_ob_50_preds['label'])if epoch > 5 else 0,#
        'ob_w_precision_50': precision_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label'])if epoch > 5 else 0,#
        'ob_w_recall_50': recall_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label'])if epoch > 5 else 0,#
        'ob_w_f1_50': f1_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label'])if epoch > 5 else 0,#
        'ob_w_accuracy_25':  accuracy_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label'])if epoch > 5 else 0,#,
        'ob_w_precision_25':  precision_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label'])if epoch > 5 else 0,#,
        'ob_w_recall_25':  recall_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label'])if epoch > 5 else 0,#,
        'ob_w_f1_25':  f1_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label'])if epoch > 5 else 0,#,
        'predictions':total_ob_w_50_preds,
        'ground_truth':total_ob_w_50_ground_truth,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }