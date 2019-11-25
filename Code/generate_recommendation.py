import os

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from annoy import AnnoyIndex
from sklearn.neighbors import BallTree
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


EXAMPLE_DIR = "/home/ubuntu/Final-Project-Group8/Data/example_images/blazer"
EXAMPLE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Code/blazer_example_image.csv"
STORE_DIR = "/home/ubuntu/Final-Project-Group8/Code/banana_republic_images"
STORE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Code/banana_republic_images.csv"


def generate_image_mapping_csv(image_dir, csv_name):
    """Returns a csv file mapping image_id and image_label for a given image directory."""
    image_id = [i for i in range(len(os.listdir(image_dir)))]
    image_label = os.listdir(image_dir)
    image_dict = {"image_id": image_id, "image_label": image_label}
    image_df = pd.DataFrame(image_dict)
    image_df.to_csv("{}.csv".format(csv_name))


# Generate image mapping csv files
generate_image_mapping_csv(EXAMPLE_DIR, "blazer_example_image")
generate_image_mapping_csv(STORE_DIR, "banana_republic_images")


class FashionDataset(Dataset):
    """A dataset for the fashion images and fashion image labels.

    Arguments:
        Data csv path
        Image directory path
        Image transformation
    """
    def __init__(self, img_dir_path, img_transform, data_csv_path):
        self.data_csv_path = data_csv_path
        self.img_dir_path = img_dir_path
        self.img_transform = img_transform
        self.df = pd.read_csv(self.data_csv_path, header=0).reset_index(drop=True)
        self.img_id = self.df['image_id']
        self.img_label = self.df['image_label']

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir_path, str(self.img_label[index])))
        img = img.convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def __len__(self):
        return self.img_id.shape[0]


def create_data_loader(img_dir, info_csv_path, batch_size):
     """Returns an image loader for the model."""
     img_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
     img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
     data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
     return data_loader


# Create data loader
example_loader = create_data_loader(EXAMPLE_DIR, EXAMPLE_CSV_PATH, batch_size=1)
store_loader = create_data_loader(STORE_DIR, STORE_CSV_PATH, batch_size=64)
store_df = pd.read_csv(STORE_CSV_PATH)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.mobile_model = models.mobilenet_v2(pretrained=True)
        n = 0
        for child in self.mobile_model.children():
            n += 1
            if n < 2:
                for param in child.parameters():
                    param.requires_grad = False
        self.features = nn.Sequential(*list(self.mobile_model.children())[:-1])
        self.linear = nn.Linear(62720, 149)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x.view(len(x), -1))
        return x


def extract_feature_maps(data_loader, model):
    """Extract the feature maps from the given model."""
    with torch.no_grad():
        for train_idx, features in enumerate(data_loader):
            x = model.features(features)
            feature_maps = model.linear(x)
            return feature_maps


model = CNN()
model.load_state_dict(torch.load("mobilenet_model.pt"))
model.eval()


# Get feature maps
example_feature_maps = extract_feature_maps(example_loader, model)
store_feature_maps = example_feature_maps(store_loader, model)


# Approximate KNN
# Index store
store_item = AnnoyIndex(store_feature_maps.size()[1], 'dot')
for i in range(store_feature_maps.size()[0]):
    store_item.add_item(i, store_feature_maps[i])
store_item.build(150) # More trees gives higher precision when querying
store_item.save('store_items.ann')
# Index example
example_item = AnnoyIndex(example_feature_maps.size()[1], 'dot')
example_item.load('store_items.ann')
recommendations = example_item.get_nns_by_item(0, 5)
print(example_item.get_nns_by_item(0, 5, include_distances=True))
for recommendation in recommendations:
    print(store_df['image_label'][recommendation])


# Scikit-learn KNN
rng = np.random.RandomState(42)
tree = BallTree(store_feature_maps)
dist, ind = tree.query(example_feature_maps, k=5)
print(ind) # Indices of 5 closest neighbors
print(dist) # Distances to 5 closest neighbors
for i in ind:
    for idx in i:
        print(store_df['image_label'][idx])
