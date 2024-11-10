import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass, label

class TreeSpotLocalizationLoss(nn.Module):
    def __init__(self, w_dice_weight=0.7, tversky_weight=0.3):
        """
        Custom loss for tree spot localization in aerial images.

        Parameters:
            dice_weight (float): Weight for the Dice loss.
            distance_weight (float): Weight for the soft L2 distance loss.
        """
        super(TreeSpotLocalizationLoss, self).__init__()
        self.w_dice_weight = w_dice_weight
        self.tversky_weight = tversky_weight

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
        center_map = torch.zeros_like(mask, dtype=torch.float32)
        for batch_idx in range(mask.size(0)):
            mask_np = mask[batch_idx, 0].cpu().numpy()
            
            #label connected components
            labeled_mask, num_features = label(mask_np)
            
            #find the center of each labeled spot
            centers = center_of_mass(mask_np, labels=labeled_mask, index=range(1, num_features + 1))
            
            for cy, cx in centers:
                if not (torch.isnan(torch.tensor(cy)) or torch.isnan(torch.tensor(cx))):
                    cy, cx = int(cy), int(cx)
                    for y in range(max(0, cy - kernel_size // 2), min(mask.size(2), cy + kernel_size // 2)):
                        for x in range(max(0, cx - kernel_size // 2), min(mask.size(3), cx + kernel_size // 2)):
                            dist = torch.tensor(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2), device=mask.device)
                            center_map[batch_idx, 0, y, x] += torch.exp(-dist)
        return center_map

    def weighted_dice_loss(self, pred, target, epsilon=1e-6):
        center_weight_map = self.get_center_weight_map(target, 56, 12)
        #normalize
        center_weight_map /= center_weight_map.sum(dim=(2, 3), keepdim=True) + epsilon

        #activation
        pred = torch.sigmoid(pred)

        #calc. weighted intersection and union
        inter = (pred * target * center_weight_map).sum(dim=(2, 3))
        union = (pred * center_weight_map).sum(dim=(2, 3)) + (target * center_weight_map).sum(dim=(2, 3))

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
        ## Calculate Dice loss for overlap
        #dice_loss = self.dice_loss(predictions, target)

        ## Calculate Soft L2 distance loss for alignment accuracy
        #distance_loss = self.soft_l2_loss(predictions, target)
        
        w_dice_loss = self.weighted_dice_loss(predictions,target)
        tversky_loss = self.tversky_loss(predictions, target, alpha=0.7)

        #weight
        total_loss = self.w_dice_weight * w_dice_loss + self.tversky_weight * tversky_loss
        
        return total_loss


from chamferdist import ChamferDistance
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolygonCanopyLoss(nn.Module):
    def __init__(self, dice_weight=0.5, chamfer_weight=0.5):
        """
        Custom loss for polygon canopy prediction.
        
        Parameters:
            dice_weight (float): Weight for the Dice loss.
            chamfer_weight (float): Weight for the Chamfer Distance loss.
        """
        super(PolygonCanopyLoss, self).__init__()
        self.dice_weight = dice_weight
        self.chamfer_weight = chamfer_weight
        self.chamfer_dist = ChamferDistance()  # Chamfer distance for boundary alignment

    def dice_loss(self, pred, target, smooth=1e-6):
        """Calculate Dice Loss for binary masks."""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def chamfer_loss(self, pred, target):
        """Calculate Chamfer Distance Loss for polygon boundaries."""
        pred_points = self.extract_boundary_points(pred)
        target_points = self.extract_boundary_points(target)
        return self.chamfer_dist(pred_points, target_points)

    def extract_boundary_points(self, mask):
        """Extract boundary points from the binary mask."""
        contours = mask.contour().float()
        return contours.view(1, -1, 2)

    def forward(self, predictions, target):
        """
        Forward pass of the PolygonCanopyLoss.
        
        Parameters:
            predictions (Tensor): Predicted canopy polygon, logits (B, 1, H, W).
            target (Tensor): Ground truth polygon masks (B, 1, H, W).
        """
        dice_loss = self.dice_loss(predictions, target)

        #chamfer Loss for accurate polygon boundary alignment
        chamfer_loss = self.chamfer_loss(predictions, target)

        total_loss = self.dice_weight * dice_loss + self.chamfer_weight * chamfer_loss
        return total_loss
