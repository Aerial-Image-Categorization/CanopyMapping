import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label
import numpy as np

class TreeSpotLocalizationLoss(nn.Module):
    def __init__(
        self,
        w_dice_weight=0.6,
        tversky_weight=0.2,
        soft_l2_weight=0.2,
        tversky_alpha=0.5,
        focal_weight=0.2,
        focal_alpha=1,
        focal_gamma=2,
        focal_tversky_weight=0.2,
        focal_tversky_alpha=0.7,
        focal_tversky_gamma=4/3
        ):
        """
        Custom loss for tree spot localization in aerial images.

        Parameters:
            w_dice_weight (float): Weight for the Dice loss.
            tversky_weight (float): Weight for the Tversky loss.
            soft_l2_weight (float): Weight for the soft L2 distance loss.
            focal_weight (float): Weight for the Focal loss.
            focal_tversky_weight (float): Weight for the Focal Tversky loss.
        """
        super(TreeSpotLocalizationLoss, self).__init__()
        self.w_dice_weight = w_dice_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.soft_l2_weight = soft_l2_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_tversky_weight = focal_tversky_weight
        self.focal_tversky_alpha = focal_tversky_alpha
        self.focal_tversky_gamma = focal_tversky_gamma

    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice Loss for binary masks."""
        if target.sum() == 0:
            loss = (pred ** 2).mean()
            return loss
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice  # Returning Dice loss, not coefficient

    def soft_l2_loss(self, pred, target):
        """Compute a soft L2 distance loss on the masked areas."""
        pred_points = (pred > 0.5).float()  # Binarize predictions at threshold 0.5
        distances = F.mse_loss(pred_points, target, reduction='none')
        return (distances * target).mean()  # Only penalize areas with actual spots in target
    
    def get_center_weight_map(self, mask, kernel_size=31, sigma=10):
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

    def weighted_dice_loss(self, pred, target, epsilon=1e-6):
        assert pred.size() == target.size()
        sum_dim = (-1, -2)
        # DEBUG                                          
        #print(target.unsqueeze(1).size())
        center_weight_map = self.get_center_weight_map(target.unsqueeze(1), 50, 10).squeeze(1)
        center_weight_map+=epsilon
        
        # DEBUG                                          
        print(pred.size())
        print(target.size())
        print(center_weight_map.size())
        #calc. weighted intersection and union
        #inter = (pred * target * center_weight_map).sum(dim=(2, 3))
        #union = (pred * center_weight_map).sum(dim=(2, 3)) + (target * center_weight_map).sum(dim=(2, 3))
        inter = (pred * target * center_weight_map).sum(dim=sum_dim)
        union = (pred * center_weight_map).sum(dim=sum_dim) + (target * center_weight_map).sum(dim=sum_dim)
  
        #calc. weighted Dice coef.
        dice = (2 * inter + epsilon) / (union + epsilon)
        loss = 1 - dice.mean()

        return loss

    def weighted_dice_loss_opt(self, pred, target, epsilon=1e-6):
        assert pred.size() == target.size()
        sum_dim = (-1, -2)
        if target.sum() == 0:
            loss = (pred ** 2).mean()
            return loss
        #from border to center weights
        weit = 1 + F.avg_pool2d(target, kernel_size=13, stride=1, padding=6)
        weit = torch.where(weit < 1.2, torch.tensor(1.0, dtype=weit.dtype, device=weit.device), weit/1.2)

        #pred = torch.sigmoid(pred)
        inter = (pred * target * weit).sum(dim=sum_dim)
        union = (pred*weit).sum(dim=sum_dim) + (target*weit).sum(dim=sum_dim)
        dice = (2 * inter + epsilon) / (union + epsilon)
        return 1-dice.mean()
    
    def tversky_loss(self, pred, target, alpha=0.7, smooth=1e-6):
        """Calculate Tversky Loss for binary masks."""
        pred = torch.sigmoid(pred)

        #calc. tp, fp, and fn
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        #calc. Tversky coef.
        tversky_coeff = (TP + smooth) / (TP + alpha * FP + (1 - alpha) * FN + smooth)
        return 1 - tversky_coeff
    
    def focal_loss(self, pred, target):
        """Calculate Focal Loss for binary masks."""
        pred = torch.sigmoid(pred)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()
    
    def focal_tversky_loss(self, pred, target, alpha=0.7, gamma=4/3, smooth=1e-6):
        """Calculate Focal Tversky Loss for binary masks."""
        pred = torch.sigmoid(pred)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        #tversky
        tversky_index = (TP + smooth) / (TP + alpha * FP + (1-alpha) * FN + smooth)

        #focal
        focal_tversky_loss = torch.pow((1 - tversky_index.mean()), 1 / gamma)
        return focal_tversky_loss
    
    def forward(self, predictions, target):
        """
        Forward pass of the TreeSpotLocalizationLoss.

        Parameters:
            predictions (Tensor): Predicted mask from the model, logits (B, 1, H, W).
            target (Tensor): Ground truth mask with white spots for trees (B, 1, H, W).
        """
        
        #w_dice_loss = self.w_dice_weight * self.weighted_dice_loss_opt(predictions,target) if self.w_dice_weight !=0 else 0
        w_dice_loss = self.w_dice_weight * self.dice_loss(predictions,target) if self.w_dice_weight !=0 else 0
        tversky_loss = self.tversky_weight * self.tversky_loss(predictions, target, alpha=self.tversky_alpha) if self.tversky_weight != 0 else 0
        soft_l2_loss = soft_l2_loss * self.soft_l2_loss(predictions, target) if self.soft_l2_weight != 0 else 0
        focal_loss = self.focal_weight * self.focal_loss(predictions, target) if self.focal_weight != 0 else 0
        focal_tversky_loss = self.focal_tversky_weight * self.focal_tversky_loss(predictions, target, alpha=self.focal_tversky_alpha, gamma=self.focal_tversky_gamma) if self.focal_tversky_weight != 0 else 0

        #weight
        total_loss = w_dice_loss + tversky_loss + soft_l2_loss + focal_loss + focal_tversky_loss
        
        return total_loss

from pytorch_msssim import ssim, SSIM
#class PolygonCanopyLoss(nn.Module):
#    def __init__(self, dice_weight=0.0, ssim_weight=0.0):
#        """
#        Custom loss for polygon canopy prediction.
#        
#        Parameters:
#            dice_weight (float): Weight for the Dice loss.
#        """
#        super(PolygonCanopyLoss, self).__init__()
#        self.dice_weight = dice_weight
#        self.ssim_weight = ssim_weight
#
#    def dice_loss(self, pred, target, smooth=1e-6):
#        """Calculate Dice Loss for binary masks."""
#        #pred = torch.sigmoid(pred)
#        intersection = (pred * target).sum()
#        union = pred.sum() + target.sum()
#        dice = (2. * intersection + smooth) / (union + smooth)
#        return 1 - dice  # Returning Dice loss, not coefficient
#    
#    def ssim_loss(self, pred, target):
#        """
#        Calculate SSIM loss for structural similarity between predicted and ground truth masks.
#        SSIM is implemented using pytorch_msssim.
#        """
#        pred = torch.sigmoid(pred)
#        ssim_value = ssim(pred, target, data_range=1.0)  # SSIM score
#        return 1 - ssim_value
#
#    def forward(self, predictions, target):
#        """
#        Forward pass of the PolygonCanopyLoss.
#        
#        Parameters:
#            predictions (Tensor): Predicted canopy polygon, logits (B, 1, H, W).
#            target (Tensor): Ground truth polygon masks (B, 1, H, W).
#        """
#        dice_loss = self.dice_weight * self.dice_loss(predictions,target) if self.dice_weight !=0 else 0
#        #dice_loss = self.dice_loss(predictions, target)
#        #ssim_loss = self.ssim_loss(predictions, target)
#
#
#        total_loss = dice_loss# + self.ssim_weight * ssim_loss
#        return total_loss
#

import torch
import torch.nn as nn
from pytorch_msssim import ssim

class PolygonCanopyLoss(nn.Module):
    def __init__(
        self,
        dice_weight=0.0, 
        ssim_weight=0.0, 
        bce_weight=0.0, 
        diversity_weight=0.0,
        focal_tversky_weight = 0.0,
        focal_tversky_alpha = 0.0,
        focal_tversky_gamma = 0.0
    ):
        """
        Custom loss for polygon canopy prediction.

        Parameters:
            dice_weight (float): Weight for the Dice loss.
            ssim_weight (float): Weight for the SSIM loss.
            bce_weight (float): Weight for Binary Cross Entropy loss.
            diversity_weight (float): Weight for the diversity regularization loss.
        """
        super(PolygonCanopyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ssim_weight = ssim_weight
        self.bce_weight = bce_weight
        self.diversity_weight = diversity_weight
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.focal_tversky_weight = focal_tversky_weight
        self.focal_tversky_alpha = focal_tversky_alpha
        self.focal_tversky_gamma = focal_tversky_gamma

    #def dice_loss(self, pred, target, smooth=1e-6):
    #    """Calculate Dice Loss for binary masks."""
    #    #pred = torch.sigmoid(pred)
    #    pred = (F.sigmoid(pred) > 0.5).float()
    #    intersection = (pred * target).sum()
    #    union = pred.sum() + target.sum()
    #    dice = (2. * intersection + smooth) / (union + smooth)
    #    return 1 - dice  # Returning Dice loss, not coefficient
    def focal_tversky_loss(self, pred, target, reduce_batch_first: bool = False, alpha=0.7, gamma=4/3, smooth=1e-6):
        """Calculate Focal Tversky Loss for binary masks."""
        pred = F.sigmoid(pred).float()
        sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        TP = (pred * target).sum(dim=sum_dim)
        FP = ((1 - target) * pred).sum(dim=sum_dim)
        FN = (target * (1 - pred)).sum(dim=sum_dim)

        #tversky
        tversky_index = (TP + smooth) / (TP + alpha * FP + (1-alpha) * FN + smooth)

        #focal
        focal_tversky_loss = torch.pow((1 - tversky_index.mean()), 1 / gamma)
        return focal_tversky_loss
    
    def dice_loss(self, input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first
        #input = (F.sigmoid(input) > 0.5).float()
        input = F.sigmoid(input).float()
        
        
        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        #sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice_coeff = (inter + epsilon) / (union + epsilon)
        return 1-dice_coeff.mean()
    
    #def dice_loss_w(self, input, target, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    #    """
    #    Dice loss with equal weighting for black (background) and white (foreground) pixels.
#
    #    Parameters:
    #        input (Tensor): Predicted logits (before sigmoid) of shape (B, H, W) or (B, 1, H, W).
    #        target (Tensor): Ground truth binary mask of shape (B, H, W) or (B, 1, H, W).
    #        reduce_batch_first (bool): Whether to reduce the batch dimension first.
    #        epsilon (float): Smoothing term to avoid division by zero.
    #    Returns:
    #        Tensor: Weighted Dice loss.
    #    """
    #    assert input.size() == target.size()
    #    assert input.dim() == 3 or not reduce_batch_first
#
    #    input = F.sigmoid(input)  # Apply sigmoid to get probabilities
    #    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
#
    #    # Calculate foreground and background weights
    #    foreground_weight = 1.0 / (target.sum(dim=sum_dim) + epsilon)  # Weight for white pixels
    #    background_weight = 1.0 / ((1 - target).sum(dim=sum_dim) + epsilon)  # Weight for black pixels
#
    #    # Calculate the intersection and union
    #    inter_foreground = 2 * (input * target).sum(dim=sum_dim)
    #    union_foreground = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    #    dice_foreground = (inter_foreground + epsilon) / (union_foreground + epsilon)
#
    #    inter_background = 2 * ((1 - input) * (1 - target)).sum(dim=sum_dim)
    #    union_background = (1 - input).sum(dim=sum_dim) + (1 - target).sum(dim=sum_dim)
    #    dice_background = (inter_background + epsilon) / (union_background + epsilon)
#
    #    # Weighted Dice coefficient
    #    dice_coeff = (foreground_weight * dice_foreground + background_weight * dice_background) / (
    #        foreground_weight + background_weight
    #    )
#
    #    return 1 - dice_coeff.mean()


    def ssim_loss(self, pred, target):
        """
        Calculate SSIM loss for structural similarity between predicted and ground truth masks.
        SSIM is implemented using pytorch_msssim.
        """
        #pred = (F.sigmoid(pred) > 0.5).float()
        pred = F.sigmoid(pred).float()
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        ssim_value = ssim(pred, target, data_range=1.0)  # SSIM score
        return 1 - ssim_value

    def diversity_loss(self, pred, target):
        """
        Add diversity penalty to avoid all-white or all-black predictions.
        Penalize predictions that are too confident or saturated.
        """
        pred_prob = torch.sigmoid(pred)
        mean_pred = pred_prob.mean()
        return torch.abs(mean_pred - 0.5)  # Penalize mean predictions far from 0.5

    def forward(self, predictions, target):
        """
        Forward pass of the PolygonCanopyLoss.

        Parameters:
            predictions (Tensor): Predicted canopy polygon, logits (B, 1, H, W).
            target (Tensor): Ground truth polygon masks (B, 1, H, W).
        """
        # Compute losses
        dice_loss = self.dice_weight * self.dice_loss(predictions, target) if self.dice_weight != 0 else 0
        bce_loss = self.bce_weight * self.bce_loss_fn(torch.sigmoid(predictions), target) if self.bce_weight != 0 else 0
        ssim_loss = self.ssim_weight * self.ssim_loss(predictions, target) if self.ssim_weight != 0 else 0
        diversity_loss = self.diversity_weight * self.diversity_loss(predictions, target) if self.diversity_weight != 0 else 0
        focal_tversky_loss = self.focal_tversky_weight * self.focal_tversky_loss(predictions, target, alpha=self.focal_tversky_alpha, gamma=self.focal_tversky_gamma) if self.focal_tversky_weight != 0 else 0
        print(f'''
        - dice_loss: {dice_loss}
        - bce_loss: {bce_loss}
        - ssim_loss: {ssim_loss}
        - diversity_loss: {diversity_loss}
        - focal_tversky_loss: {focal_tversky_loss}
        ''')
        # Combine losses
        total_loss = dice_loss + bce_loss + ssim_loss + diversity_loss + focal_tversky_loss
        return total_loss
