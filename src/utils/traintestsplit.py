import os
import math

def middle_split(images, masks, train_size):
    test_images_count = int(len(images)*round(1-train_size,5))
    X = Y = math.sqrt(test_images_count)
    max_x, max_y = os.path.splitext(os.path.basename(images[-1]))[0].split('_')[2:4]
    if type(Y)==float:
        if int(max_x)>int(max_y):
            X = int(X)
            Y = test_images_count // int(X)
        elif int(max_x)<int(max_y):
            Y = int(X)
            X = test_images_count // int(X)
        else:
            X = int(X)
            Y = test_images_count // int(X)
    mid_x = int(max_x)//2
    mid_y = int(max_y)//2
    x_start = mid_x - X//2
    y_start = mid_y - Y//2

    images_test=[]
    masks_test=[]

    x_end = x_start+X
    y_end = y_start+Y
    for x in range(x_start,x_end):
        for y in range(y_start,y_end):
            image_name = os.path.join(images[0][:-17],f'tile_tif_{x}_{y}.png')
            mask_name = os.path.join(images[0][:-24],f'masks/tile_shp_{x}_{y}.png')
            if image_name in images:
                images.remove(image_name)
                images_test.append(image_name)
                masks.remove(mask_name)
                masks_test.append(mask_name)
                
    return images, images_test, masks, masks_test