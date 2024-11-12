import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2

def create_yolo_segmentation_label(mask_path, img_width, img_height, class_id=0):
    """
    Convert a mask to Ultralytics YOLO format for segmentation.
    
    Parameters:
    - mask_path (str): Path to the mask image.
    - img_width (int): Width of the original image.
    - img_height (int): Height of the original image.
    - class_id (int): Class ID for the object (default is 0).
    
    Returns:
    - yolo_labels (list): List of YOLO-formatted label strings for segmentation.
    """
    # Read the mask in grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found at path: {mask_path}")
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_labels = []
    
    # Process each contour
    for contour in contours:
        # Simplify the contour to reduce the number of points if needed
        epsilon = 0.01 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Skip contours with fewer than 3 points (not valid polygons)
        if len(contour) < 3:
            continue
        
        # Normalize each point in the contour and format as <class-index> <x1> <y1> <x2> <y2> ...
        segmentation_points = [class_id]  # Start the row with the class ID
        for point in contour:
            px, py = point[0]
            normalized_x = px / img_width
            normalized_y = py / img_height
            segmentation_points.extend([normalized_x, normalized_y])
        
        # Convert list to string format required by YOLO
        label_str = " ".join(map(str, segmentation_points))
        yolo_labels.append(label_str)
    
    return yolo_labels


def create_yolo_label(mask_path, img_width, img_height, class_id=0):
    """
    Convert a mask to YOLO format for segmentation.
    
    Parameters:
    - mask_path (str): Path to the mask image.
    - img_width (int): Width of the original image.
    - img_height (int): Height of the original image.
    - class_id (int): Class ID for the object (default is 0).
    
    Returns:
    - yolo_labels (list): List of YOLO-formatted label strings.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_labels = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        
        segmentation_points = []
        for point in contour:
            px, py = point[0]
            segmentation_points.append(px / img_width)
            segmentation_points.append(py / img_height)

        label_str = f"{class_id} {x_center} {y_center} {width} {height} " + " ".join(map(str, segmentation_points))
        yolo_labels.append(label_str)
    
    return yolo_labels

def convert_image_mask_pairs_to_yolo(image_dir, mask_dir, output_label_dir, class_id=0):
    """
    Converts a set of image-mask pairs to YOLO segmentation format.
    
    Parameters:
    - image_dir (str): Directory with input images.
    - mask_dir (str): Directory with corresponding masks.
    - output_label_dir (str): Directory to save YOLO labels.
    - class_id (int): Class ID for the objects (default is 0).
    """
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)
    
    #for img_filename in os.listdir(image_dir):
    for img_filename in tqdm(os.listdir(image_dir), desc="Creating YOLO annotations", leave=False):
        img_path = os.path.join(image_dir, img_filename)
        mask_path = os.path.join(mask_dir, img_filename.replace('tif', 'shp'))
        
        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]
        
        #yolo_labels = create_yolo_label(mask_path, img_width, img_height, class_id)
        yolo_labels = create_yolo_segmentation_label(mask_path, img_width, img_height, class_id)
        
        
        label_path = os.path.join(output_label_dir, img_filename.replace('tif', 'shp').replace('.png','.txt'))
        
        with open(label_path, 'w') as label_file:
            label_file.write("\n".join(yolo_labels))
            
def create_new_dataset_structure(
    base_dir,
    new_base_dir,
    source_folders = ['train', 'val', 'test'],
    source_subfolders = ['images', 'labels'],
    dest_folders = ['images', 'labels'],
    dest_subfolders = ['train', 'val', 'test']
):

    os.makedirs(os.path.join(new_base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(new_base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(new_base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(new_base_dir, 'labels', 'val'), exist_ok=True)

    for i, src_folder in enumerate(source_folders):
        for i, src_subfolder in enumerate(source_subfolders):
            source_dir = os.path.join(base_dir, src_folder, src_subfolder)
            destination_dir = os.path.join(new_base_dir, dest_folders[i], dest_subfolders[i])

            for filename in os.listdir(source_dir):
                file_path = os.path.join(source_dir, filename)
                shutil.copy(file_path, os.path.join(destination_dir, filename))

def create_new_dataset_structure(
    base_dir,
    new_base_dir,
    source_folders=['aug_train', 'val', 'test'],
    source_subfolders=['images', 'mask_txts'],
    dest_folders=['images', 'labels'],
    dest_subfolders=['train', 'val', 'test']
):
    '''
    old_paths = {
        'aug_train': [
            'images',
            'masks'
        ],
        'val': [
            'images',
            'masks'
        ],
        'test': [
            'images',
            'masks'
        ]
        
    }
    
    new_paths = {
        'images': [
            'train',
            'val',
            'test'
        ],
        'labels': [
            'train',
            'val',
            'test'
        ]
    }
    '''
    for dest_folder in dest_folders:
        for dest_subfolder in dest_subfolders:
            os.makedirs(os.path.join(new_base_dir, dest_folder, dest_subfolder), exist_ok=True)
            
    for src_folder in tqdm(source_folders, desc="Copy folders", leave=False):
        for src_subfolder in tqdm(source_subfolders, desc="Copy subsets", leave=False):
            source_dir = os.path.join(base_dir, src_folder, src_subfolder)

            if os.path.exists(source_dir):
                for filename in os.listdir(source_dir):
                    file_path = os.path.join(source_dir, filename)
                    
                    if os.path.isfile(file_path):
                        dest_index = source_folders.index(src_folder)
                        destination_dir = os.path.join(new_base_dir, dest_folders[source_subfolders.index(src_subfolder)], dest_subfolders[dest_index])
                        
                        if 'shp' in filename:
                            shutil.copy(file_path, os.path.join(destination_dir, filename.replace('_shp_', '_')))
                        if 'tif' in filename:
                            shutil.copy(file_path, os.path.join(destination_dir, filename.replace('_tif_', '_')))
            else:
                print(f"Warning: {source_dir} does not exist and will be skipped.")


if __name__ == '__main__':
    datasets = [
        #'../data/2024-10-30-loc-dataset-192',
        #'../data/2024-10-30-loc-dataset-128',
        #'../data/2024-10-30-loc-dataset-256',
        #'../data/2024-10-30-loc-dataset-384',
        '../data/2024-10-30-loc-dataset-512',
        #'../data/2024-10-30-loc-dataset-1024'
    ]
    
    subfolders = [
        'aug_train',
        'train',
        'val',
        'test'
    ]
    
    #for subfolder in tqdm(subfolders, desc="Sets", leave=False):
    #    convert_image_mask_pairs_to_yolo(
    #        image_dir=os.path.join(datasets[-1], subfolder, 'images'),
    #        mask_dir=os.path.join(datasets[-1], subfolder, 'masks'),
    #        output_label_dir=os.path.join(datasets[-1], subfolder, 'mask_txts'),
    #        class_id=0
    #    )
    #    
    #create_new_dataset_structure(
    #    base_dir = datasets[-1],
    #    new_base_dir = datasets[-1]+'_yolo'
    #)
    
    
    for dataset in tqdm(datasets, desc="Datasets"):
        for subfolder in tqdm(subfolders, desc="Subsets", leave=False):
            convert_image_mask_pairs_to_yolo(
                image_dir=os.path.join(dataset, subfolder, 'images'),
                mask_dir=os.path.join(dataset, subfolder, 'masks'),
                output_label_dir=os.path.join(dataset, subfolder, 'mask_txts'),
                class_id=0
            )
            
        create_new_dataset_structure(
           base_dir = dataset,
           new_base_dir = dataset+'_yolo_seg'
        )
