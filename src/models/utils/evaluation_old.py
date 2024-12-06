import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .scores import dice, weighted_dice, iou, weighted_iou, objectwise_classification_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@torch.inference_mode()
def evaluate(net, dataloader, device, epoch, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    
    total_prec_lst = []
    
    
    
    dice_score = []
    total_iou = []
    #total_px_accuracy = []
    #total_px_precision = []
    #total_px_recall = []
    #total_px_f1 = []
    total_ob_accuracy = []
    total_ob_precision = []
    total_ob_recall = []
    total_ob_f1 = []
    out_mask_pred = None
    out_mask_true = None
    out_image = None
    total_weighted_dice_score = []
    total_weighted_iou_score = []
    total_ob_iou = []
    
    total_ob_w_accuracy_50 = []
    total_ob_w_precision_50 = []
    total_ob_w_recall_50 = []
    total_ob_w_f1_50 = []
    total_ob_w_iou_50 = []
    
    total_ob_w_accuracy_25 = []
    total_ob_w_precision_25 = []
    total_ob_w_recall_25 = []
    total_ob_w_f1_25 = []
    total_ob_w_iou_25 = []
    
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
                
                total_weighted_dice_score.append(weighted_dice(mask_pred.squeeze(), mask_true.squeeze()))
                total_weighted_iou_score.append(weighted_iou(mask_pred.squeeze(), mask_true.squeeze()))

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

                    total_ob_accuracy.append(obj_cl_results_50['accuracy'])
                    total_ob_precision.append(obj_cl_results_50['precision'])
                    total_ob_recall.append(obj_cl_results_50['recall'])
                    total_ob_f1.append(obj_cl_results_50['f1'])
                    total_ob_iou.append(obj_cl_results_50['mean_iou'])

                    # DEBUG
                    total_prec_lst.append(obj_cl_results_50['precision'])
                    #


                    total_ob_w_accuracy_50.append(obj_w_cl_results_50['accuracy'])
                    total_ob_w_precision_50.append(obj_w_cl_results_50['precision'])
                    total_ob_w_recall_50.append(obj_w_cl_results_50['recall'])
                    total_ob_w_f1_50.append(obj_w_cl_results_50['f1'])
                    total_ob_w_iou_50.append(obj_w_cl_results_50['mean_iou'])

                    total_tp += obj_w_cl_results_50['true_positive_count']
                    total_fp += obj_w_cl_results_50['false_positive_count']
                    total_fn += obj_w_cl_results_50['false_negative_count']

                    total_ob_w_accuracy_25.append(obj_w_cl_results_25['accuracy'])
                    total_ob_w_precision_25.append(obj_w_cl_results_25['precision'])
                    total_ob_w_recall_25.append(obj_w_cl_results_25['recall'])
                    total_ob_w_f1_25.append(obj_w_cl_results_25['f1'])
                    total_ob_w_iou_25.append(obj_w_cl_results_25['mean_iou'])
                    total_preds['label'].extend(obj_w_cl_results_50['predictions']['label'])
                    total_preds['prob'].extend(obj_w_cl_results_50['predictions']['prob'])
                    total_ground_truth += obj_w_cl_results_50['ground_truth']

                #print(mask_pred.squeeze().size(), mask_true.squeeze().size())
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
    
    #px_avg_accuracy = total_px_accuracy / max(num_val_batches, 1)
    #px_avg_precision = total_px_precision / max(num_val_batches, 1)
    #px_avg_recall = total_px_recall / max(num_val_batches, 1)
    #px_avg_f1 = total_px_f1 / max(num_val_batches, 1)
    
    ob_avg_accuracy = np.mean([x for x in total_ob_accuracy if x is not None]) # / max(num_val_batches, 1)
    ob_avg_precision = np.mean([x for x in total_ob_precision if x is not None]) # / max(num_val_batches, 1)
    ob_avg_recall = np.mean([x for x in total_ob_recall if x is not None]) # / max(num_val_batches, 1)
    ob_avg_f1 = np.mean([x for x in total_ob_f1 if x is not None]) # / max(num_val_batches, 1)
    ob_avg_iou = np.mean([x for x in total_ob_iou if x is not None]) # / max(num_val_batches, 1)
    
    ob_avg_w_accuracy_50 = np.mean([x for x in total_ob_w_accuracy_50 if x is not None]) #/ max(num_val_batches, 1)
    ob_avg_w_precision_50 = np.mean([x for x in total_ob_w_precision_50 if x is not None]) #/ max(num_val_batches, 1)
    ob_avg_w_recall_50 = np.mean([x for x in total_ob_w_recall_50 if x is not None]) #/ max(num_val_batches, 1)
    ob_avg_w_f1_50 = np.mean([x for x in total_ob_w_f1_50 if x is not None]) #/ max(num_val_batches, 1)
    ob_avg_w_iou_50 = np.mean([x for x in total_ob_w_iou_50 if x is not None]) #/ max(num_val_batches, 1)
    
    ob_avg_w_accuracy_25 = np.mean([x for x in total_ob_w_accuracy_25 if x is not None]) # / max(num_val_batches, 1)
    ob_avg_w_precision_25 = np.mean([x for x in total_ob_w_precision_25 if x is not None]) # / max(num_val_batches, 1)
    ob_avg_w_recall_25 = np.mean([x for x in total_ob_w_recall_25 if x is not None]) # / max(num_val_batches, 1)
    ob_avg_w_f1_25 = np.mean([x for x in total_ob_w_f1_25 if x is not None]) # / max(num_val_batches, 1)
    ob_avg_w_iou_25 = np.mean([x for x in total_ob_w_iou_25 if x is not None]) # / max(num_val_batches, 1)
    
    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'w_dice': avg_w_dice,
        'w_iou': avg_w_iou,
        'obj_iou_50': ob_avg_iou,
        'obj_w_iou_50': ob_avg_w_iou_50,
        'obj_w_iou_25': ob_avg_w_iou_25,
        #'px_accuracy': px_avg_accuracy,
        #'px_precision': px_avg_precision,
        #'px_recall': px_avg_recall,
        #'px_f1': px_avg_f1,
        'ob_accuracy_50': ob_avg_accuracy,
        'ob_precision_50': ob_avg_precision,
        'ob_recall_50': ob_avg_recall,
        'ob_f1_50': ob_avg_f1,
        'ob_w_accuracy_50': ob_avg_w_accuracy_50,#accuracy_score(total_ground_truth, total_preds['label']),#
        'ob_w_precision_50': ob_avg_w_precision_50,#precision_score(total_ground_truth, total_preds['label']),#
        'ob_w_recall_50': ob_avg_w_recall_50,#recall_score(total_ground_truth, total_preds['label']),#
        'ob_w_f1_50': ob_avg_w_f1_50,#f1_score(total_ground_truth, total_preds['label']),#
        'ob_w_accuracy_25': ob_avg_w_accuracy_25,
        'ob_w_precision_25': ob_avg_w_precision_25,
        'ob_w_recall_25': ob_avg_w_recall_25,
        'ob_w_f1_25': ob_avg_w_f1_25,
        'tp_count': total_tp,
        'fp_count': total_fp,
        'fn_count': total_fn,
        'predictions':total_preds,
        'ground_truth':total_ground_truth,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred,
        'DEBUG': total_prec_lst
    }