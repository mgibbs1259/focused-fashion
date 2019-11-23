import os

import pandas as pd
import torch
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/test.csv"
IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"


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


class FashionDataset(Dataset):
    """A dataset for the fashion images and fashion image labels.

    Arguments:
        Data csv path
        Image directory path
        Image transformation
    """
    def __init__(self, data_csv_path, img_dir_path, img_transform):
        self.data_csv_path = data_csv_path
        self.img_dir_path = img_dir_path
        self.img_transform = img_transform
        self.df = pd.read_csv(self.data_csv_path, header=0, names=['label_id', 'image_id']).reset_index(drop=True)
        self.x_train = self.df['image_id']
        self.mlb = MultiLabelBinarizer()
        self.y_train = self.mlb.fit_transform(self.df['label_id'].apply(literal_eval))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir_path, str(self.x_train[index]) + '.jpg'))
        img = img.convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        img_label = torch.from_numpy(self.y_train[index])
        return img, img_label.float()

    def __len__(self):
        return self.x_train.shape[0]


def create_data_loader(data_path, img_dir, batch_size):
     """Returns an image loader for the model."""
     img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                         transforms.ToTensor()])
     dataset = FashionDataset(data_path, img_dir, img_transform)
     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
     return loader
