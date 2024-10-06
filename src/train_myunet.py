#import sys
#sys.path.append('../')
import os
import torch
from models import my_UNet as UNet
from models.my_UNet.datasets import ImageDataset

if __name__ == '__main__':
    train_set_path = '../data/2024-09-29-seg-dataset-200/aug_train'
    valid_set_path = '../data/2024-09-29-seg-dataset-200/val'
    epochs = 50
    batch_size = 6
    lr = 1e-8
    scale = 1
    val = 0.1
    amp = False
    bilinear = False

    model = UNet.model(UNet.config(n_channels=3, n_classes=1, bilinear=bilinear))
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    #dataset = UNet.ImageDataset(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_set = ImageDataset(train_set_path)
    valid_set = ImageDataset(valid_set_path)
    
    try:
        UNet.train_model_Jaccard(
            train_set = train_set,
            valid_set = valid_set,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            img_scale=scale,
            val_percent=val / 100,
            amp=amp
        )
    except Exception:
    #except torch.cuda.OutOfMemoryError:
        #logging.error('Detected OutOfMemoryError! Enabling checkpointing to reduce memory usage, but this slows down training. Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        UNet.train_model(
            dataset=dataset,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            img_scale=scale,
            val_percent=val / 100,
            amp=amp
        )