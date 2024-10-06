import torch
from torch import Tensor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of IoU loss for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    iou = (inter + epsilon) / (sets_sum + epsilon)
    iou_loss = iou
    return iou_loss.mean()
    
def iou(input, target, epsilon=1e-6):
    assert input.size() == target.size(), "Input and target must be the same size."
    assert input.dim() == 3

    intersection = (input * target).sum(dim=(-1, -2))  # Shape: [batch_size]
    union = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersection  # Shape: [batch_size]

    union = torch.where(union == 0, torch.tensor(1.0, device=union.device), union)

    iou = intersection / union

    return iou.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def jaccard_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_jaccard_coeff if multiclass else jaccard_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def multiclass_jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return jaccard_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)



def calculate_pixelwise_classification_metrics(mask_pred, mask_true, multiclass = False):
    # Apply threshold to logits
    #preds = torch.sigmoid(preds)  # convert logits to probabilities
    #preds = (preds > threshold).float()  # binarize predictions
    if multiclass:
        preds_flat = mask_pred.argmax(dim=1).cpu().numpy().flatten()
        true_flat = mask_true.argmax(dim=1).cpu().numpy().flatten()

        accuracy = accuracy_score(true_flat, preds_flat)
        precision = precision_score(true_flat, preds_flat, average='macro')
        recall = recall_score(true_flat, preds_flat, average='macro')
        f1 = f1_score(true_flat, preds_flat, average='macro')
    else:
        preds_flat = mask_pred.cpu().numpy().flatten()
        true_flat = mask_true.cpu().numpy().flatten()
    
        accuracy = accuracy_score(true_flat, preds_flat)
        precision = precision_score(true_flat, preds_flat)
        recall = recall_score(true_flat, preds_flat)
        f1 = f1_score(true_flat, preds_flat)

    return accuracy, precision, recall, f1

import torch
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_objectwise_classification_metrics(mask_pred, mask_true, threshold=0.5):
    """
    Calculate objectwise precision, recall, and F1 score by detecting connected components
    and matching objects between the predicted and true masks.
    """
    # Convert tensors to numpy arrays
    mask_pred_np = mask_pred.cpu().numpy()
    mask_true_np = mask_true.cpu().numpy()

    # Detect connected components (objects) in the predicted and true masks
    pred_labeled, num_pred = label(mask_pred_np)
    true_labeled, num_true = label(mask_true_np)

    # Initialize counts for true positives, false positives, and false negatives
    TP = 0  # True positives: correctly predicted objects
    FP = 0  # False positives: predicted objects that don't match any true object
    FN = 0  # False negatives: true objects that don't have any matching prediction

    # Match predicted objects to true objects using intersection-over-union (IoU)
    pred_objects = np.unique(pred_labeled)
    true_objects = np.unique(true_labeled)

    matched_true_objects = set()  # Keep track of matched true objects

    for pred_obj in pred_objects:
        #if pred_obj == 0:  # Skip background
        #    continue
        
        # Extract predicted object as a binary mask
        pred_mask = (pred_labeled == pred_obj)

        # Find the true object with the maximum overlap
        best_iou = 0
        best_true_obj = None
        
        for true_obj in true_objects:
        #    if true_obj == 0:  # Skip background
        #        continue
            
            # Extract true object as a binary mask
            true_mask = (true_labeled == true_obj)

            # Calculate intersection and union
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()

            # Calculate IoU (Intersection over Union)
            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_true_obj = true_obj

        # If IoU is above a threshold (e.g., 0.5), consider it a match
        if best_iou > 0.5:
            TP += 1  # It's a true positive
            matched_true_objects.add(best_true_obj)
        else:
            FP += 1  # It's a false positive

    # Count false negatives (true objects with no matching prediction)
    FN = len(true_objects) - 1 - len(matched_true_objects)  # Subtract 1 for background

    # Calculate precision, recall, and F1 score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return accuracy, precision, recall, f1

def calculate_objectwise_classification_metrics2(mask_pred, mask_true, threshold=0.5):
    """
    Calculate objectwise precision, recall, F1 score, and accuracy for multi-class segmentation
    ignoring the background (class 0).
    """
    mask_pred_np = mask_pred.cpu().numpy()
    mask_true_np = mask_true.cpu().numpy()

    pred_labeled, num_pred = label(mask_pred_np > threshold)
    true_labeled, num_true = label(mask_true_np > threshold)

    TP = 0  # True positives: correctly predicted objects
    FP = 0  # False positives: predicted objects that don't match any true object
    FN = 0  # False negatives: true objects that don't have any matching prediction

    matched_true_objects = set()  # Keep track of matched true objects

    pred_objects = np.unique(pred_labeled)
    true_objects = np.unique(true_labeled)

    for pred_obj in pred_objects:
        #if pred_obj == 0:  # Skip background
        #    continue

        pred_mask = (pred_labeled == pred_obj)
        best_iou = 0
        best_true_obj = None
        
        for true_obj in true_objects:
            #if true_obj == 0:
            #    continue

            true_mask = (true_labeled == true_obj)
            intersection = np.logical_and(pred_mask, true_mask).sum()
            union = np.logical_or(pred_mask, true_mask).sum()

            iou = intersection / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_true_obj = true_obj

        if best_iou > 0.5:
            TP += 1
            matched_true_objects.add(best_true_obj)
        else:
            FP += 1

    FN = len(true_objects) - 1 - len(matched_true_objects)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    return accuracy, precision, recall, f1


import torch

