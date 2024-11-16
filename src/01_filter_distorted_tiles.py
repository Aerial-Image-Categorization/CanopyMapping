import os
import shutil
from PIL import Image
import numpy as np
from tqdm import tqdm

def filter_distorted_images(src_folder, dest_folder, radius):
    """
    Copies mask images from src_folder to dest_folder if they contain white pixels
    within a specified radius around the center of the image.

    Parameters:
    - src_folder (str): Path to the folder containing mask images.
    - dest_folder (str): Path to the destination folder for qualifying masks.
    - radius (int): Radius around the center to check for white pixels.
    """
    os.makedirs(dest_folder, exist_ok=True)
    os.makedirs(dest_folder.replace('masks','images'), exist_ok=True)
    overall = 0
    found=0

    for filename in tqdm(os.listdir(src_folder), desc="Processing masks"):
        if filename.endswith(".png"):
            img_path = os.path.join(src_folder, filename)
            with Image.open(img_path) as img:
                img = img.convert("L")
                width, height = img.size
                center_x, center_y = width // 2, height // 2

                pixels = np.array(img)
                white_pixel_found = False

                #check each pixel within the radius
                for y in range(center_y - radius, center_y + radius + 1):
                    for x in range(center_x - radius, center_x + radius + 1):
                        #within image bounds and within radius
                        if (0 <= x < width) and (0 <= y < height):
                            if ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2:
                                if pixels[y, x] == 255:  #white
                                    white_pixel_found = True
                                    break
                    if white_pixel_found:
                        break

                #in region
                if white_pixel_found:
                    dest_path = os.path.join(dest_folder, filename)
                    found+=1
                    shutil.copy(img_path, dest_path)
                    shutil.copy(img_path.replace('masks','images'), dest_path.replace('masks','images'))
                    #print(f"Copied {filename} to destination folder.")
                else:
                    pass
                    #print(f"Skipped {filename}: No white pixel in the specified radius.")
                overall+=1
    print(f'result: {found}/{overall}')

if __name__ == '__main__':
    img_size = 1024
    radius = 10
    dirs = [
        'u_aug_train',
        'u_test',
        'u_val'
    ]
    for dir in dirs:
        src_folder = f'../data/2024-11-12-seg-dataset-{img_size}/{dir}/masks'
        dest_folder = f'../data/2024-11-12-seg-dataset-{img_size}/{dir}_f10/masks'
        filter_distorted_images(src_folder, dest_folder, radius)
