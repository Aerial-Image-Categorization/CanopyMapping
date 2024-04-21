import sys
sys.path.append('../')
import torch
from models import UNet

if __name__ == '__main__':
    dataset_path = '../data/2024-04-13-dataset/augmentated/train'
    epochs = 5
    batch_size = 1
    lr = 1e-5
    scale = 1
    val = 0.1
    amp = False
    
    model = UNet.model(n_channels=3, n_classes=2, bilinear=True)
    dataset = UNet.ImageDataset(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    try:
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
    except torch.cuda.OutOfMemoryError:
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