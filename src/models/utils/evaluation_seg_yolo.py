import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from .scores import dice, iou
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@torch.inference_mode()
def evaluate_seg_yolo(net, dataloader, device, epoch, amp):
    #net.eval()
    num_val_batches = len(dataloader)
    
    dice_score = []
    total_iou = []

    out_mask_pred = None
    out_mask_true = None
    out_image = None

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            # Unpack the batch
            image, mask_true = batch['image'], batch['mask']

            # Move data to device
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask using YOLO model
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
                # Process predictions
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            except:
                mask_pred = torch.zeros((512, 512), device=device)
            
            dice_score.append(dice(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))
            total_iou.append(iou(mask_pred.squeeze(), mask_true.squeeze(), reduce_batch_first=False))

            # Save a sample for visualization
            out_mask_pred = mask_pred.squeeze(0)
            out_mask_true = mask_true.squeeze(0)
            out_image = image.squeeze(0)

    # Compute averages
    avg_dice = np.mean([x.cpu().numpy() for x in dice_score if x is not None])
    avg_iou = np.mean([x.cpu().numpy() for x in total_iou if x is not None])

    return {
        'dice_score': avg_dice,
        'iou': avg_iou,
        'image': out_image,
        'mask_true': out_mask_true,
        'mask_pred': out_mask_pred
    }
