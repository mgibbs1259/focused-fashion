import os

import pandas as pd
from PIL import Image


def obtain_smallest_image_size(img_dir):
    """Returns the dimensions of the smallest image in terms of area."""
    image_sizes = {}
    for image in os.listdir(img_dir):
        print(image)
        try:
            with Image.open(os.path.join(img_dir, image)) as img:
                width, height = img.size
                image_sizes[image] = (width, height, width * height)
                print(image_sizes[image])
        except:
            with open('bad_images.txt', 'a') as file:
                file.write(image)
    smallest_image = min(image_sizes, key=lambda k: image_sizes[k][2])
    return image_sizes[smallest_image]


def check_label_balance(image_dir_path, image_info_path):
    """Prints False if there are missing images from the csv files."""
    images = []
    images += [each for each in os.listdir(image_dir_path)]
    images.sort()
    df = pd.read_csv(image_info_path)
    df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))
    mask = df['file name'].isin(images)
    return print(mask.value_counts())
