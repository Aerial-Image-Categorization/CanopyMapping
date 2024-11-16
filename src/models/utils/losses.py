import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label
import numpy as np

class TreeSpotLocalizationLoss(nn.Module):
    def __init__(self, w_dice_weight=0.6, tversky_weight=0.2, soft_l2_weight=0.2, tversky_alpha=0.5):
        """
        Custom loss for tree spot localization in aerial images.

        Parameters:
            dice_weight (float): Weight for the Dice loss.
            distance_weight (float): Weight for the soft L2 distance loss.
        """
        super(TreeSpotLocalizationLoss, self).__init__()
        self.w_dice_weight = w_dice_weight
        self.tversky_weight = tversky_weight
        self.tversky_alpha = tversky_alpha
        self.soft_l2_weight = soft_l2_weight

    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice Loss for binary masks."""
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

    def tversky_loss(self, pred, target, alpha=0.7, smooth=1e-6):
        """Calculate Tversky Loss for binary masks."""
        pred = torch.sigmoid(pred)

        #calc. tp, fp, and fn
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        #calc. Tversky coef.
        tversky_coeff = (TP + smooth) / (TP + alpha * FP + (1 - alpha) * FN + smooth)
        return 1 - tversky_coeff  # Returning Tversky loss
    
    def forward(self, predictions, target):
        """
        Forward pass of the TreeSpotLocalizationLoss.

        Parameters:
            predictions (Tensor): Predicted mask from the model, logits (B, 1, H, W).
            target (Tensor): Ground truth mask with white spots for trees (B, 1, H, W).
        """
        
        #w_dice_loss = self.w_dice_weight * self.weighted_dice_loss(predictions,target) if self.w_dice_weight !=0 else 0
        w_dice_loss = self.w_dice_weight * self.dice_loss(predictions,target) if self.w_dice_weight !=0 else 0
        tversky_loss = self.tversky_weight * self.tversky_loss(predictions, target, alpha=self.tversky_alpha) if self.tversky_weight != 0 else 0
        soft_l2_loss = soft_l2_loss * self.soft_l2_loss(predictions, target) if self.soft_l2_weight != 0 else 0

        #weight
        total_loss = w_dice_loss + tversky_loss + soft_l2_loss
        
        return total_loss

from pytorch_msssim import ssim, SSIM
class PolygonCanopyLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ssim_weight=0.5):
        """
        Custom loss for polygon canopy prediction.
        
        Parameters:
            dice_weight (float): Weight for the Dice loss.
        """
        super(PolygonCanopyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ssim_weight = ssim_weight

    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice Loss for binary masks."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def ssim_loss(self, pred, target):
        """
        Calculate SSIM loss for structural similarity between predicted and ground truth masks.
        SSIM is implemented using pytorch_msssim.
        """
        pred = torch.sigmoid(pred)
        ssim_value = ssim(pred, target, data_range=1.0)  # SSIM score
        return 1 - ssim_value

    def forward(self, predictions, target):
        """
        Forward pass of the PolygonCanopyLoss.
        
        Parameters:
            predictions (Tensor): Predicted canopy polygon, logits (B, 1, H, W).
            target (Tensor): Ground truth polygon masks (B, 1, H, W).
        """
        dice_loss = self.dice_loss(predictions, target)
        ssim_loss = self.ssim_loss(predictions, target)


        total_loss = self.dice_weight * dice_loss + self.ssim_weight * ssim_loss
        return total_loss
