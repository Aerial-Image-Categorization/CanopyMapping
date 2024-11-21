import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.ndimage import center_of_mass, label
import torch
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


from scipy.ndimage import label, distance_transform_edt

def weighted_iou_opt(pred, target, reduce_batch_first: bool = False, epsilon=1e-6):
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    #from border to center weights
    weit = 1 + F.avg_pool2d(target, kernel_size=13, stride=1, padding=6)
    weit = torch.where(weit < 1.2, torch.tensor(1.0, dtype=weit.dtype, device=weit.device), weit/1.2)

    #pred = torch.sigmoid(pred)
    inter = (pred * target * weit).sum(dim=sum_dim)
    union = (pred*weit).sum(dim=sum_dim) + (target*weit).sum(dim=sum_dim) - inter
    wiou = (inter + epsilon) / (union  + epsilon)
    return wiou.mean()


def weighted_dice_opt(pred, target, reduce_batch_first: bool = False, epsilon=1e-6):
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    #from border to center weights
    weit = 1 + F.avg_pool2d(target, kernel_size=13, stride=1, padding=6)
    weit = torch.where(weit < 1.2, torch.tensor(1.0, dtype=weit.dtype, device=weit.device), weit/1.2)

    #pred = torch.sigmoid(pred)
    inter = (pred * target * weit).sum(dim=sum_dim)
    union = (pred*weit).sum(dim=sum_dim) + (target*weit).sum(dim=sum_dim)
    dice = (2 * inter + epsilon) / (union + epsilon)
    return dice.mean()

def get_center_weight_map(mask, kernel_size=31, sigma=10):
    """
    Generates a Gaussian-weighted center map for each connected component in the binary mask.
    
    Parameters:
    - mask (torch.Tensor): Batch of binary masks with shape [batch, 1, height, width].
    - kernel_size (int): Size of the region around each center to apply the weight.
    - sigma (float): Standard deviation for the Gaussian weighting.

    Returns:
    - center_map (torch.Tensor): Weighted map with the same shape as `mask`.
    """
    center_map = torch.zeros_like(mask, dtype=torch.float32)
    
    for batch_idx in range(mask.size(0)):
        mask_np = mask[batch_idx, 0].cpu().numpy()
        
        if np.all(mask_np == 0): #no foreground
            continue

        
        labeled_mask, num_features = label(mask_np) #label connected components
        if len(np.unique(labeled_mask))==1:
            print(np.unique(labeled_mask))
        #print(f'{labeled_mask}, -> {np.unique(labeled_mask)}, -> {np.unique(mask_np)}')
        if len(np.unique(labeled_mask)) > 1:
            try:
                #calc. center
                centers = center_of_mass(mask_np, labels=labeled_mask, index=range(1, num_features + 1))

                for cy, cx in centers:
                    if np.isnan(cy) or np.isnan(cx):
                        continue  #skip if the center is invalid

                    cy, cx = int(cy), int(cx)
                    
                    #define bounds for the Gaussian-weighted region around the center
                    y_min, y_max = max(0, cy - kernel_size // 2), min(mask.size(2), cy + kernel_size // 2)
                    x_min, x_max = max(0, cx - kernel_size // 2), min(mask.size(3), cx + kernel_size // 2)

                    #apply Gaussian weight
                    for y in range(y_min, y_max):
                        for x in range(x_min, x_max):
                            dist_sq = ((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2)
                            center_map[batch_idx, 0, y, x] += torch.exp(-torch.tensor(dist_sq, device=mask.device))
            
            except Exception as e:
                print(f"Error calculating center of mass for batch index {batch_idx}: {e}, {num_features}, {labeled_mask}, -> {np.unique(labeled_mask)}, -> {np.unique(mask_np)}")
                continue

    return center_map

def weighted_dice(pred, target, reduce_batch_first: bool = False,  epsilon=1e-6):
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    #print(target.size(),target.dim(), np.unique(target.cpu().numpy()))
    center_weight_map = get_center_weight_map(target.unsqueeze(0).unsqueeze(0), 50, 10) #torch.ones_like(target.unsqueeze(0).unsqueeze(0), dtype=torch.float32) 
    center_weight_map+=epsilon

    #center_weight_map /= center_weight_map.sum(dim=sum_dim, keepdim=True) + epsilon
    if torch.isnan(center_weight_map).any():
        raise ValueError("dice: center_weight_map returned None!")
    #activation
    #pred = torch.sigmoid(pred)
    #calc. weighted intersection and union
    inter = (pred * target * center_weight_map).sum(dim=sum_dim)
    union = (pred * center_weight_map).sum(dim=sum_dim) + (target * center_weight_map).sum(dim=sum_dim)
    if torch.isnan(inter).any():
        raise ValueError("dice: NaN values detected in intersection!")
    if torch.isnan(union).any():
        raise ValueError("dice: NaN values detected in union!")
    #calc. weighted Dice coef.
    dice = (2 * inter + epsilon) / (union + epsilon)
    if torch.isnan(dice).any():
        raise ValueError(f"dice contains NaN values: {dice}")
    #loss = 1 - dice.mean()
    #print(f'dice_mean: {dice.mean()}')
    
    return dice.mean() #loss

def weighted_iou(pred, target, reduce_batch_first: bool = False,  epsilon=1e-6):
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first
    

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    
    center_weight_map = get_center_weight_map(target.unsqueeze(0).unsqueeze(0), 50, 10)  #torch.ones_like(target.unsqueeze(0).unsqueeze(0), dtype=torch.float32) 
    center_weight_map+=epsilon

    #center_weight_map /= center_weight_map.sum(dim=sum_dim, keepdim=True) + epsilon
    
    if torch.isnan(center_weight_map).any():
        raise ValueError("iou: center_weight_map returned None!")
    #activation
    #pred = torch.sigmoid(pred)
    #calc. weighted intersection and union
    inter = (pred * target * center_weight_map).sum(dim=sum_dim)
    union = (pred * center_weight_map).sum(dim=sum_dim) + (target * center_weight_map).sum(dim=sum_dim) - inter
    
    if torch.isnan(inter).any() or torch.isnan(union).any():
        raise ValueError("iou: NaN values detected in intersection or union!")
    #calc. weighted Dice coef.
    iou = (inter + epsilon) / (union  + epsilon)
    if torch.isnan(iou).any():
        raise ValueError(f"iou contains NaN values: {iou}")
    #loss = 1 - dice.mean()
    #print(f'iou_mean: {iou.mean()}')
    return iou.mean() #loss

def dice(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    inter = 2 * (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice_coeff = (inter + epsilon) / (union + epsilon)
    return dice_coeff.mean()

def iou(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    assert input.size() == target.size(), "Input and target must be the same size."
    assert input.dim() == 3 or not reduce_batch_first
    
    
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    
    intersection = (input * target).sum(dim=sum_dim)  # Shape: [batch_size]
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - intersection  # Shape: [batch_size]

    #union = torch.where(union == 0, torch.tensor(1.0, device=union.device), union)
    iou_coeff = (intersection + epsilon) / (union + epsilon)

    return iou_coeff.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice
    return 1 - fn(input, target, reduce_batch_first=True)

def weighted_dice_loss(pred, target, reduce_batch_first: bool = False, epsilon=1e-6):
    if target.sum() == 0:
        return (pred ** 2).mean()  # Penalize the prediction if it's not close to zero
    return 1 - weighted_dice_opt(pred, target, reduce_batch_first, epsilon=1e-6)

def iou_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_jaccard_coeff if multiclass else iou
    return 1 - fn(input, target, reduce_batch_first=True)

def multiclass_jaccard_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return iou_loss(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


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

def gaussian_center_map(component_mask, sigma):
    distance_map = distance_transform_edt(component_mask) #distance map from the center of the object
    center_map = 1-np.exp(-0.5 * (distance_map / sigma)**2) #apply Gaussian weighting based on distance from center
    return center_map * component_mask

def objectwise_classification_metrics(mask_pred, mask_true, threshold=0.5, weighted = False, sigma=10, kernel_size=50):
    assert mask_pred.shape == mask_true.shape, f"Shape mismatch: {mask_pred.shape} and {mask_true.shape}"
    batch_size = mask_pred.shape[0]
    
    total_true_positive_count = 0
    total_true_negative_count = 0 #empty image-mask pairs
    total_false_positive_count = 0
    total_false_negative_count = 0
    iou_scores = []
    
    tp_mask_batch = np.zeros_like(mask_true.cpu().numpy())
    fp_mask_batch = np.zeros_like(mask_true.cpu().numpy())
    fn_mask_batch = np.zeros_like(mask_true.cpu().numpy())
    
    preds = {
        'label': [],
        'prob': []
    }
    ground_truth = []
    
    total_true_objects = []
    total_pred_objects = []
    
    for i in range(batch_size):
        true_mask = mask_true[i].cpu().numpy()
        pred_mask = mask_pred[i].cpu().numpy()
        
        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0
        
        if np.all(true_mask == 0) and np.all(pred_mask == 0): #not returns 0 if both empty
            true_negative_count += 1
            preds['label'].append(0)
            preds['prob'].append(1.0)
            ground_truth.append(0)
            continue
        
        labeled_true = label(true_mask)
        labeled_pred = label(pred_mask)

        if isinstance(labeled_true, tuple):
            labeled_true = labeled_true[0]
        if isinstance(labeled_pred, tuple):
            labeled_pred = labeled_pred[0]
        if len(labeled_true.shape) != 2 or len(labeled_pred.shape) != 2:
            raise ValueError(f"Unexpected shape: labeled_true has shape {labeled_true.shape}, labeled_pred has shape {labeled_pred.shape}")
            
        labeled_true = np.asarray(labeled_true, dtype=np.int32)
        labeled_pred = np.asarray(labeled_pred, dtype=np.int32)

        tp_mask = np.zeros_like(true_mask)
        fp_mask = np.zeros_like(true_mask)
        fn_mask = np.zeros_like(true_mask)

        true_objects = np.unique(labeled_true)[1:]  # Exclude background (label 0)
        pred_objects = np.unique(labeled_pred)[1:]

        total_true_objects.append(true_objects)
        total_pred_objects.append(pred_objects)

        for pred_obj in pred_objects:
            pred_obj_mask = labeled_pred == pred_obj
            
            obj_iou_scores = []
            for true_obj in true_objects:
                true_obj_mask = labeled_true == true_obj
                if weighted:
                    weighted_true_obj = gaussian_center_map(true_obj_mask, sigma)
                    intersection = np.sum(true_obj_mask * pred_obj_mask * weighted_true_obj)
                    union = np.sum(pred_obj_mask*weighted_true_obj) + np.sum(true_obj_mask*weighted_true_obj) - intersection
                else:
                    intersection = np.logical_and(pred_obj_mask, true_obj_mask).sum()
                    union = np.logical_or(pred_obj_mask, true_obj_mask).sum()
                iou = intersection / union if union > 0 else 0
                obj_iou_scores.append(iou)

            max_iou = max(obj_iou_scores) if obj_iou_scores else 0
            iou_scores.append(max_iou)
            if max_iou >= threshold:
                true_positive_count += 1
                tp_mask[pred_obj_mask] = 1
                preds['label'].append(1)
                preds['prob'].append(max_iou)
                ground_truth.append(1)
            else:
                false_positive_count += 1
                fp_mask[pred_obj_mask] = 1
                preds['label'].append(1)
                preds['prob'].append(max_iou)
                ground_truth.append(0)

        for true_obj in true_objects:
            true_obj_mask = labeled_true == true_obj
            if not np.logical_and(true_obj_mask, pred_mask).any():
                false_negative_count += 1
                fn_mask[true_obj_mask] = 1
                preds['label'].append(0)
                preds['prob'].append(1.0)
                ground_truth.append(1)

        total_true_positive_count += true_positive_count
        total_true_negative_count += true_negative_count
        total_false_positive_count += false_positive_count
        total_false_negative_count += false_negative_count

        tp_mask_batch[i] = tp_mask
        fp_mask_batch[i] = fp_mask
        fn_mask_batch[i] = fn_mask

    total = total_true_positive_count + total_false_positive_count + total_false_negative_count
    # DEBUG
    #print("Length of preds['label']: ", len(preds['label']))
    #print("Length of ground_truth: ", len(ground_truth))
    #print("Total count: ", total)

    if total == 0:
        accuracy = None # 0.0 # 1.0
        precision = None # 0.0 # 1.0
        recall = None # 0.0 # 1.0
        f1 = None # 0.0 # 1.0
        mean_iou = None # 0.0 # 1.0
    else:
        accuracy = total_true_positive_count / total if total > 0 else 1.0
        precision = total_true_positive_count / (total_true_positive_count + total_false_positive_count) if (total_true_positive_count + total_false_positive_count) > 0 else 0.0
        recall = total_true_positive_count / (total_true_positive_count + total_false_negative_count) if (total_true_positive_count + total_false_negative_count) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        
    return {
        #'accuracy': accuracy_score(ground_truth, preds['label']), #accuracy,
        #'precision': precision_score(ground_truth, preds['label'], zero_division=0), #precision,
        #'recall': recall_score(ground_truth, preds['label'], zero_division=0), #recall,
        #'f1': f1_score(ground_truth, preds['label'], zero_division=0), #f1,
        'mean_iou': mean_iou,
        #'true_positive_count': total_true_positive_count,
        #'true_negative_count': total_true_negative_count, #empty image-mask pairs
        #'false_positive_count': total_false_positive_count,
        #'false_negative_count': total_false_negative_count,
        #'tp_mask_batch': tp_mask_batch,
        #'fp_mask_batch': fp_mask_batch,
        #'fn_mask_batch': fn_mask_batch,
        'predictions': preds,
        'ground_truth': ground_truth
    }


