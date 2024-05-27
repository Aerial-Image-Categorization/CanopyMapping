import os

def middle_split(images, masks, train_size):
    test_images_count = int(len(images)*(1-train_size))
    X = Y = math.sqrt(test_images_count)
    if type(Y)==float:
        X = int(X)
        Y = test_images_count // int(Y)

    max_x, max_y = os.path.splitext(os.path.basename(images[-1]))[0].split('_')[2:4]
    mid_x = int(int(max_x)/2)
    mid_y = int(int(max_y)/2)
    x_start = mid_x//2 - X//2
    y_start = mid_y//2 - Y//2

    images_test=[]
    masks_test=[]

    x_end = x_start+X
    y_end = y_start+Y
    for x in range(x_start,x_end+1):
        for y in range(y_start,y_end+1):
            image_name = images[0][:-17]+f'tile_tif_{x}_{y}.png'
            mask_name = images[0][:-24]+f'masks/tile_shp_{x}_{y}.png'
            if image_name in images:
                images.remove(image_name)
                images_test.append(image_name)
                masks.remove(mask_name)
                masks_test.append(mask_name)
                
    return images, images_test, masks, masks_test