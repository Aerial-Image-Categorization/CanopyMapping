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
