import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from annoy import AnnoyIndex
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class RecommendationDataset(Dataset):
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


def create_data_loader(data_path, img_dir, batch_size):
     """Returns an image loader for the model."""
     img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                         transforms.ToTensor()])
     dataset = RecommendationDataset(img_dir, img_transform, data_path)
     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
     return loader


# Define path to image of interest
EXAMPLE_PATH = "/home/ubuntu/Final-Project-Group8/Code/Example"

# Create a df mapping image of interest
ex_image_id = [0]
ex_image_label = ["example_img.jpg"]
ex_image_dict = {"image_id": ex_image_id, "image_label": ex_image_label}
ex_image_df = pd.DataFrame(ex_image_dict)
# ex_image_df.to_csv("example_image.csv")

# Define path to image of interest csv
EXAMPLE_CSV = "/home/ubuntu/Final-Project-Group8/Code/example_image.csv"


# Define path to store images
STORE_PATH = "/home/ubuntu/Final-Project-Group8/Code/Blouses"

# Create a df mapping Banana Republic Images
image_id = [i for i in range(len(os.listdir(STORE_PATH)))]
image_label = os.listdir(STORE_PATH)
image_dict = {"image_id": image_id, "image_label": image_label}
image_df = pd.DataFrame(image_dict)
# image_df.to_csv("banana_republic_images.csv")

# Define path to store csv
STORE_CSV = "/home/ubuntu/Final-Project-Group8/Code/banana_republic_images.csv"


# Create data loaders for both
example_loader = create_data_loader(EXAMPLE_CSV, EXAMPLE_PATH, batch_size=1)
store_loader = create_data_loader(STORE_CSV, STORE_PATH, batch_size=len(os.listdir(STORE_PATH)))


# Load model
BATCH_SIZE = 1024
DROPOUT = 0.5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (12, 12), stride=2, padding=1) # Output (n_examples, 32, 46, 46)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2) # Output (n_examples, 32, 23, 23)

        self.conv2 = nn.Conv2d(32, 64, (6, 6), stride=2, padding=1) # Output (n_examples, 64, 10, 10)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # Output (n_examples, 64, 5, 5)

        self.linear1 = nn.Linear(64*5*5, BATCH_SIZE) # Input will be flattened to (n_examples, 64, 5, 5)
        self.linear1_bn = nn.BatchNorm1d(BATCH_SIZE)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(BATCH_SIZE, 149)

        self.relu = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.relu(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.relu(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.relu(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return x


def extract_feature_maps(x, model):
    x = model.pool1(model.convnorm1(model.relu(model.conv1(x))))
    return model.pool2(model.convnorm2(model.relu(model.conv2(x))))


model = CNN()
model.load_state_dict(torch.load("model_number_1.pt"))


def get_feature_maps(loader, model):
    for train_idx, features in enumerate(loader):
        features = extract_feature_maps(features, model)
        return features

# Get features for example_loader
example_feature_maps = get_feature_maps(example_loader, model)
print(example_feature_maps.size())

# Get features for store_loader
store_feature_maps = get_feature_maps(store_loader, model)
print(store_feature_maps.size())

# Concatenate feature maps



#KNN
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree
# rng = np.random.RandomState(0)
# X = concatenated feature maps
# tree = BallTree(X, leaf_size=2)
# dist, ind = tree.query(X[:1], k=3)
# print(ind) # indices of 3 closest neighbors
# print(dist) # distances to 3 closest neighbors


# # K-means
# def kmeans_clustering_elbow_curve(X):
#     """Shows an elbow curve plot to determine the appropriate number of k-means clusters."""
#     distorsions = []
#     for k in range(1, 20):
#         kmeans_model = KMeans(n_clusters=k)
#         kmeans_model.fit(X)
#         distorsions.append(kmeans_model.inertia_)
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(range(1, 4), distorsions)
#     plt.title('Elbow Curve')
#     plt.show()
#
#
# def kmeans_clustering(X, clusters=10):
#     """Returns the kmeans model and predicted values."""
#     kmeans_model = KMeans(n_clusters=clusters).fit(X)
#     predict_values = kmeans_model.predict(X)
#     return kmeans_model, predict_values
