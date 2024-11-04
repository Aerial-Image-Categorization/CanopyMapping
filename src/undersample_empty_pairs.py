import os
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

def undersample_image_mask_pairs(src_image_folder, src_mask_folder, dest_image_folder, dest_mask_folder, sample_ratio=0.5):
    """
    Copies image-mask pairs to destination folders with undersampling on fully black masks.
    Only copies 50% of pairs where the mask is fully black, and all pairs where the mask has non-black pixels.

    Parameters:
    - src_image_folder (str): Source folder for original images.
    - src_mask_folder (str): Source folder for mask images.
    - dest_image_folder (str): Destination folder for undersampled images.
    - dest_mask_folder (str): Destination folder for undersampled masks.
    - sample_ratio (float): Ratio of full-black mask pairs to keep (default is 0.5).
    """
    # Ensure destination folders exist
    os.makedirs(dest_image_folder, exist_ok=True)
    os.makedirs(dest_mask_folder, exist_ok=True)

    # Get all mask files in the source folder
    mask_files = [f for f in os.listdir(src_mask_folder) if f.endswith(".png")]

    # Loop through each mask file with a progress bar
    for mask_filename in tqdm(mask_files, desc="Processing mask files", leave=False):
        mask_path = os.path.join(src_mask_folder, mask_filename)
        image_path = os.path.join(src_image_folder, mask_filename.replace('_shp_','_tif_'))  # Assuming image and mask have the same filename

        # Check if corresponding image exists
        if not os.path.exists(image_path):
            print(f"Warning: Corresponding image for mask {mask_filename} not found. Skipping.")
            continue

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")  # Convert to grayscale for analysis
            pixels = np.array(mask)

            # Determine if mask is fully black
            is_black_only = np.all(pixels == 0)

            # Copy based on sampling criteria
            if not is_black_only or (is_black_only and random.random() < sample_ratio):
                # Copy both image and mask to destination folders
                shutil.copy(image_path, os.path.join(dest_image_folder, mask_filename.replace('_shp_','_tif_')))
                shutil.copy(mask_path, os.path.join(dest_mask_folder, mask_filename))

    print("Undersampling complete. Images and masks copied to destination folders.")


if __name__ == '__main__':
    sizes = [192,256,384,512]

    for size in tqdm(sizes, desc="Processing datasets", leave=True):
        dataset_folder = f'../data/2024-10-30-loc-dataset-{size}'
        sample_name = 'aug_train'

        src_image_folder = os.path.join(dataset_folder,sample_name,'images')  # Folder containing source images
        src_mask_folder = os.path.join(dataset_folder,sample_name,'masks')  # Folder containing source masks
        dest_image_folder = os.path.join(dataset_folder,'aug_train_u','images')   # Destination folder for undersampled images
        dest_mask_folder = os.path.join(dataset_folder,'aug_train_u','masks')     # Destination folder for undersampled masks

        undersample_image_mask_pairs(src_image_folder, src_mask_folder, dest_image_folder, dest_mask_folder)
