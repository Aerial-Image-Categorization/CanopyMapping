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

from ...utils.evaluation import evaluate

import wandb

from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def test_net(
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
    results = evaluate(net, test_loader, device, 6, False)
    
    #wandb.log({
    #    'dice_score': results['dice_score'],
    #    'iou': results['iou'],
    #    #'px_accuracy': results['px_accuracy'],
    #    #'px_precision': results['px_precision'],
    #    #'px_recall': results['px_recall'],
    #    #'px_f1': results['px_f1'],
    #    'f1': results['ob_f1'],
    #    'accuracy': results['ob_accuracy'],
    #    'precision': results['ob_precision'],
    #    'recall': results['ob_recall'],
    #    'image': [wandb.Image(results['image'].cpu(), caption="Input Image")],
    #    'mask_true': [wandb.Image(results['mask_true'].float().cpu(), caption="True Mask")],
    #    'mask_pred': [wandb.Image(results['mask_pred'].float().cpu(), caption="Predicted Mask")]
    #})
    val_predictions = np.zeros((len(np.array(results['predictions']['prob'])), 2))
    val_predictions[:, 1] = np.array(results['predictions']['prob'])
    val_predictions[:, 0] = 1 - val_predictions[:, 1]
    
    y_true = results['ground_truth'].cpu().numpy().tolist() if isinstance(results['ground_truth'], torch.Tensor) else results['ground_truth']
    preds = results['predictions']['label'].cpu().numpy().tolist() if isinstance(results['predictions']['label'], torch.Tensor) else results['predictions']['label']

    wandb.log({
        'Segmentation metrics': {
            'Dice': results['dice_score'],
            'IoU':results['iou'],
            'Weighted IoU': results['w_iou'],
            'Weighted Dice': results['w_dice']
        },
        #'validation Dice': val_score['dice_score'],
        #'validation IoU': val_score['iou'],
        'Classification metrics': {
            'obj. IoU 50': results['obj_iou_50'],
            'Accuracy 50': results['ob_accuracy_50'],
            'Precision 50': results['ob_precision_50'],
            'Recall 50': results['ob_recall_50'],
            'F1-score 50': results['ob_f1_50'],
            'Weighted obj. IoU 50': results['obj_w_iou_50'],
            'Weighted Accuracy 50': results['ob_w_accuracy_50'],
            'Weighted Precision 50': results['ob_w_precision_50'],
            'Weighted Recall 50': results['ob_w_recall_50'],
            'Weighted F1-score 50': results['ob_w_f1_50'],
            'Weighted obj. IoU 25': results['obj_w_iou_25'],
            'Weighted Accuracy 25': results['ob_w_accuracy_25'],
            'Weighted Precision 25': results['ob_w_precision_25'],
            'Weighted Recall 25': results['ob_w_recall_25'],
            'Weighted F1-score 25': results['ob_w_f1_25'],
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
        'test': {
            'images': wandb.Image(results['image'].cpu()),
            'masks': {
                'true': wandb.Image(results['mask_true'].float().cpu()),
                'pred': wandb.Image(results['mask_pred'].float().cpu()),
            }
        },
        "Confusion Matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=preds,
            class_names=['background', 'tree']
        ),
        "Precision-Recall Curve": wandb.plot.pr_curve(
            np.array(results['ground_truth']),
            val_predictions,
            labels=["background", "tree"]
        ),
        "ROC curve": wandb.plot.roc_curve(
            np.array(results['ground_truth']),
            val_predictions,
            labels=["background", "tree"]
        ),
    })
    #print(
    #    results['tp_count'],
    #    results['fp_count'],
    #    results['fn_count']
    #)
    #print(results['DEBUG'])
