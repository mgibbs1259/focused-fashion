import os
import pandas as pd


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"
TEST_INFO_PATH = "/home/ubuntu/Final-Project-Group8/test.csv"
TEST_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/test"


def check_label_balance(image_dir_path, image_info_path):
    """Prints False if there are missing images from the csv files."""
    images = []
    images += [each for each in os.listdir(image_dir_path)]
    images.sort()
    df = pd.read_csv(image_info_path)
    df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))
    mask = df['file name'].isin(images)
    return print(mask.value_counts())


if __name__ == '__main__':
    check_label_balance(TRAIN_INFO_PATH, TRAIN_IMG_DIR)
    check_label_balance(VAL_INFO_PATH, VAL_IMG_DIR)
    check_label_balance(TEST_INFO_PATH, TEST_IMG_DIR)
