import os

import pandas as pd
import torch
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
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
            pass
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
        self.df = pd.read_csv(self.data_csv_path, header=1, names=['label_id', 'image_id'])
        self.x_train = self.df['image_id']
        self.mlb = MultiLabelBinarizer()
        self.y_train = self.mlb.fit_transform(self.df['label_id'].apply(literal_eval))

    def __getitem__(self, index):
        img = Image.open(self.img_dir_path + self.x_train[index] + '.jpg')
        img = img.convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        img_label = torch.from_numpy(self.y_train[index])
        return img, img_label

    def __len__(self):
        return self.x_train.shape[0]


def create_data_loader(data_path, img_dir):
     """Returns an image loader for the model."""
     img_transform = transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BILINEAR),
                                         transforms.ToTensor()])
     dataset = FashionDataset(data_path, img_dir, img_transform)
     loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=1, pin_memory=True)
     return loader