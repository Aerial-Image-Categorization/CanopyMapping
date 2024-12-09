# %%
import os
import yaml
from ultralytics import YOLO
from models.utils import evaluate_seg_yolo
import torch
from torch.utils.data import DataLoader
from models.biomed_UNet.datasets import SegImageDataset
import wandb

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
    yolo_model_name = './models/YOLOv11n-seg/yolo11n_seg/yolo11n_512/weights/best.pt'
    dataset_folder = '../data/2024-12-08-seg2-dataset-2048'
    num_classes = 1
    images_size = 512
    class_names = ['canopy']
    
    # Login to Weights & Biases
    wandb.login(key="eed270f2f8c27af665f378c7ae0e25af584aa5dd")
    wandb.init(project="CanopyMapping", resume='allow', anonymous='must',name=f'test_yolo_seg_{img_size}', magic=True)
    ## Create YOLO dataset YAML
    #create_yolo_yaml_file(
    #    base_dir=dataset_folder,
    #    output_file='yolo_seg_dataset.yaml',
    #    num_classes=num_classes,
    #    class_names=class_names
    #)
#
    ## Load YOLO model
    model = YOLO(yolo_model_name)
    print(f"Loaded model: {yolo_model_name}")

    # Create validation dataloader
    test_set_path = os.path.join(dataset_folder, 'u_test')
    test_set = SegImageDataset(test_set_path)
    
    loader_args = dict(batch_size=1, pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=False, **loader_args)
    # Perform evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Run evaluation using evaluate_seg
    val_metrics = evaluate_seg_yolo(model, test_loader, device, epoch=0, amp=True)

    # Print validation metrics
    print("Validation Metrics:")
    print(val_metrics)
    wandb.log({
        'Segmentation metrics': {
            'Dice': results['dice_score'],
            'IoU':results['iou'],
        },
        'test': {
            'images': wandb.Image(results['image'].cpu()),
            'masks': {
                'true': wandb.Image(results['mask_true'].float().cpu()),
                'pred': wandb.Image(results['mask_pred'].float().cpu()),
            }
        },
    })
