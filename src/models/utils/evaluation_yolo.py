import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from .scores import dice, weighted_dice, iou, weighted_iou, objectwise_classification_metrics, weighted_dice_opt, weighted_iou_opt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import cv2

@torch.inference_mode()
def evaluate_yolo(net, dataloader, device, epoch, amp):
    #net.eval()
    num_val_batches = len(dataloader)
    
    dice_score = []
    total_iou = []

    total_ob_50_preds = {'label': [], 'prob': []}
    total_ob_50_ground_truth = []

    out_mask_pred = None
    out_mask_true = None
    out_image = None
    total_weighted_dice_score = []
    total_weighted_iou_score = []
    
    total_ob_iou = []
    total_ob_w_iou_50 = []
    total_ob_w_iou_25 = []
    
    total_ob_w_50_preds = {'label': [], 'prob': []}
    total_ob_w_50_ground_truth = []
    
    total_ob_w_25_preds = {'label': [], 'prob': []}
    total_ob_w_25_ground_truth = []
    
    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move data to the correct device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask using YOLO-based segmentation
            results = net(image, save=True, imgsz=512, conf=0.2)  # YOLO outputs could include multiple elements; extract segmentation mask
            
            try:
                for result in results:
                    masks = result.masks.data
                    boxes = result.boxes.data
                    clss = boxes[:, 5]
                    people_indices = torch.where(clss == 0)
                    people_masks = masks[people_indices]
                    mask_pred = torch.any(people_masks, dim=0).int()# * 255
                    cv2.imwrite('test_yolo_loc.jpg', (mask_pred * 255).cpu().numpy())
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            except:
                mask_pred = torch.zeros((512, 512), device=device)
            mask_pred = mask_pred.unsqueeze(0)
            #print(mask_pred.size(), mask_true.size())
            # Metrics
            dice_score.append(dice(mask_pred, mask_true, reduce_batch_first=False))
            total_iou.append(iou(mask_pred, mask_true, reduce_batch_first=False))

            # Weighted metrics
            total_weighted_dice_score.append(weighted_dice_opt(mask_pred, mask_true.float()))
            total_weighted_iou_score.append(weighted_iou_opt(mask_pred, mask_true.float()))

            # Object-wise metrics after a certain epoch
            if epoch > 5:
                obj_cl_results_50 = objectwise_classification_metrics(
                    mask_pred.long(),
                    mask_true,
                    threshold=0.5,
                    weighted=False
                )

                obj_w_cl_results_50 = objectwise_classification_metrics(
                    mask_pred.long(),
                    mask_true,
                    threshold=0.5,
                    weighted=True,
                    sigma=10
                )

                obj_w_cl_results_25 = objectwise_classification_metrics(
                    mask_pred.long(),
                    mask_true,
                    threshold=0.25,
                    weighted=True,
                    sigma=10
                )

                # Save object-wise metrics for analysis
                total_ob_50_preds['label'].extend(obj_cl_results_50['predictions']['label'])
                total_ob_50_preds['prob'].extend(obj_cl_results_50['predictions']['prob'])
                total_ob_50_ground_truth += obj_cl_results_50['ground_truth']
                total_ob_iou.append(obj_cl_results_50['mean_iou'])

                total_ob_w_50_preds['label'].extend(obj_w_cl_results_50['predictions']['label'])
                total_ob_w_50_preds['prob'].extend(obj_w_cl_results_50['predictions']['prob'])
                total_ob_w_50_ground_truth += obj_w_cl_results_50['ground_truth']
                total_ob_w_iou_50.append(obj_w_cl_results_50['mean_iou'])

                total_ob_w_25_preds['label'].extend(obj_w_cl_results_25['predictions']['label'])
                total_ob_w_25_preds['prob'].extend(obj_w_cl_results_25['predictions']['prob'])
                total_ob_w_25_ground_truth += obj_w_cl_results_25['ground_truth']
                total_ob_w_iou_25.append(obj_w_cl_results_25['mean_iou'])

            # Save a sample for visualization
            out_mask_pred = mask_pred.squeeze(0)
            out_mask_true = mask_true.squeeze(0)
            out_image = image.squeeze(0)

    # Calculate average metrics
    avg_dice = np.mean([x.cpu().numpy() for x in dice_score if x is not None])
    avg_iou = np.mean([x.cpu().numpy() for x in total_iou if x is not None])
    avg_w_dice = np.mean([x.cpu().numpy() for x in total_weighted_dice_score if x is not None])
    avg_w_iou = np.mean([x.cpu().numpy() for x in total_weighted_iou_score if x is not None])
    
    ob_avg_iou = np.mean([x for x in total_ob_iou if x is not None]) if epoch > 5 else 0
    ob_avg_w_iou_50 = np.mean([x for x in total_ob_w_iou_50 if x is not None]) if epoch > 5 else 0
    ob_avg_w_iou_25 = np.mean([x for x in total_ob_w_iou_25 if x is not None]) if epoch > 5 else 0
    
    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'w_dice': avg_w_dice,
        'w_iou': avg_w_iou,
        'obj_iou_50': ob_avg_iou,
        'obj_w_iou_50': ob_avg_w_iou_50,
        'obj_w_iou_25': ob_avg_w_iou_25,
        'ob_accuracy_50': accuracy_score(total_ob_50_ground_truth, total_ob_50_preds['label']) if epoch > 5 else 0,
        'ob_precision_50': precision_score(total_ob_50_ground_truth, total_ob_50_preds['label']) if epoch > 5 else 0,
        'ob_recall_50': recall_score(total_ob_50_ground_truth, total_ob_50_preds['label']) if epoch > 5 else 0,
        'ob_f1_50': f1_score(total_ob_50_ground_truth, total_ob_50_preds['label']) if epoch > 5 else 0,
        'ob_w_accuracy_50': accuracy_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label']) if epoch > 5 else 0,
        'ob_w_precision_50': precision_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label']) if epoch > 5 else 0,
        'ob_w_recall_50': recall_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label']) if epoch > 5 else 0,
        'ob_w_f1_50': f1_score(total_ob_w_50_ground_truth, total_ob_w_50_preds['label']) if epoch > 5 else 0,
        'ob_w_accuracy_25': accuracy_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label']) if epoch > 5 else 0,
        'ob_w_precision_25': precision_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label']) if epoch > 5 else 0,
        'ob_w_recall_25': recall_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label']) if epoch > 5 else 0,
        'ob_w_f1_25': f1_score(total_ob_w_25_ground_truth, total_ob_w_25_preds['label']) if epoch > 5 else 0,
        'predictions': total_ob_w_50_preds,
        'ground_truth': total_ob_w_50_ground_truth,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }
