# %%
import os
import yaml
from ultralytics import YOLO
from models.utils import evaluate_yolo
import torch
from torch.utils.data import DataLoader
from models.biomed_UNet.datasets import ImageDataset
import wandb
import numpy as np

def create_yolo_yaml_file(base_dir, output_file, num_classes=1, class_names=None):
    if class_names is None:
        class_names = ['class_name']
    elif not isinstance(class_names, list):
        raise ValueError("class_names must be a list.")

    yaml_content = {
        'path': base_dir,
        'train': os.path.join('images', 'train'),
        'val': os.path.join('images', 'val'),
        'test': os.path.join('images', 'test'),
        'nc': num_classes,
        'names': class_names
    }
    print(yaml_content)

    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

if __name__ == '__main__':
    yolo_model_name = './models/YOLOv11n-seg/yolo11n_loc/yolo11n_512/weights/best.pt'
    dataset_folder = '../data/2024-10-30-loc-dataset-1024'
    num_classes = 1
    images_size = 512
    class_names = ['tree']
    
    # Login to Weights & Biases
    wandb.login(key="eed270f2f8c27af665f378c7ae0e25af584aa5dd")
    wandb.init(project="CanopyMapping", resume='allow', anonymous='must',name=f'test_yolo_{images_size}', magic=True)

    model = YOLO(yolo_model_name)
    print(f"Loaded model: {yolo_model_name}")

    # Create validation dataloader
    test_set_path = os.path.join(dataset_folder, 'u_test')
    test_set = ImageDataset(test_set_path)
    
    loader_args = dict(batch_size=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=False, **loader_args)
    # Perform evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run evaluation using evaluate_seg
    results = evaluate_yolo(model, test_loader, device, epoch=10, amp=True)

    # Print validation metrics
    print("Validation Metrics:")
    print(results)
    
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

