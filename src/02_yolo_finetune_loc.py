# %%
import os
import yaml
from ultralytics import YOLO
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
    yolo_model_name = 'yolo11n-seg.pt'
    dataset_folder = '../../data/2024-10-30-loc-dataset-192_yolo2'
    num_classes = 1
    epochs = 1
    images_size = 192
    class_names = ['tree']
    
    wandb.login(key="eed270f2f8c27af665f378c7ae0e25af584aa5dd")

    create_yolo_yaml_file(
        base_dir = dataset_folder,
        output_file = 'yolo_dataset.yaml',
        num_classes = num_classes,
        class_names = class_names
    )

    model = YOLO(yolo_model_name)

    train_metrics = model.train(
        data='yolo_dataset.yaml',
        epochs = epochs,
        imgsz = images_size,
        project="CanopyMapping",
        name=f"yolo11n_seg_{images_size}"
    )
    print(train_metrics)

    val_metrics = model.val()
    
    print(val_metrics)