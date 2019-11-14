import os
import pandas as pd
#this script cross references the labels in the csv files with the images in the directories.
#if value counts returns false that means there are missing images from the csv files.
DATA_PATH_VAL = "/home/ubuntu/Final-Project-Group8/val_ann.csv"
IMG_DIR_VAL = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/output_validation"

images = []
images += [each for each in os.listdir(IMG_DIR_VAL)]
images.sort()

df = pd.read_csv(DATA_PATH_VAL)

df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))

mask = df['file name'].isin(images)

mask.value_counts()

DATA_PATH_TEST = "/home/ubuntu/Final-Project-Group8/test.csv"
IMG_DIR_TEST = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/output_test2"

images = []
images += [each for each in os.listdir(IMG_DIR_TEST)]
images.sort()

df = pd.read_csv(DATA_PATH_TEST)

df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))

mask = df['file name'].isin(images)

mask.value_counts()

DATA_PATH_TRAIN = "/home/ubuntu/Final-Project-Group8/train_ann.csv"
IMG_DIR_TRAIN = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/output_train"

images = []
images += [each for each in os.listdir(IMG_DIR_TRAIN)]
images.sort()

df = pd.read_csv(DATA_PATH_TRAIN)

df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))

mask = df['file name'].isin(images)

mask.value_counts()