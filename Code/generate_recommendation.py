import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from PIL import Image
from annoy import AnnoyIndex
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
from torchvision import transforms, models
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
     img_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
     dataset = RecommendationDataset(img_dir, img_transform, data_path)
     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
     return loader


# Define path to image of interest
EXAMPLE_PATH = "/home/ubuntu/Final-Project-Group8/Code/Example"

# Create a df mapping image of interest
ex_image_id = [0]
ex_image_label = ["example_img.jpg"]
ex_image_dict = {"image_id": ex_image_id, "image_label": ex_image_label}
ex_image_df = pd.DataFrame(ex_image_dict)
ex_image_df.to_csv("example_image.csv")

# Define path to image of interest csv
EXAMPLE_CSV = "/home/ubuntu/Final-Project-Group8/Code/example_image.csv"


# Define path to store images
STORE_PATH = "/home/ubuntu/Final-Project-Group8/Code/Banana_Republic"

# Create a df mapping Banana Republic Images
image_id = [i for i in range(len(os.listdir(STORE_PATH)))]
image_label = os.listdir(STORE_PATH)
image_dict = {"image_id": image_id, "image_label": image_label}
image_df = pd.DataFrame(image_dict)
image_df.to_csv("banana_republic_images.csv")

# Define path to store csv
STORE_CSV = "/home/ubuntu/Final-Project-Group8/Code/banana_republic_images.csv"


# Create data loaders for both
example_loader = create_data_loader(EXAMPLE_CSV, EXAMPLE_PATH, batch_size=1)
store_loader = create_data_loader(STORE_CSV, STORE_PATH, batch_size=64)


# Load model
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


def extract_feature_maps(x, model):
    x = model.features(x)
    x = model.linear(x)
    return x


model = CNN()
model.load_state_dict(torch.load("mobilenet_model.pt"))
model.eval()


def get_feature_maps(loader, model):
    with torch.no_grad():
        for train_idx, features in enumerate(loader):
            features = extract_feature_maps(features, model)
            return features


# Get features for example_loader
example_feature_maps = get_feature_maps(example_loader, model)
print(example_feature_maps.size())


# Get features for store_loader
store_feature_maps = get_feature_maps(store_loader, model)
print(store_feature_maps.size())


# Annoy Approximate KNN
# Store
t = AnnoyIndex(store_feature_maps.size()[1], 'dot')  # Length of item vector that will be indexed
for i in range(store_feature_maps.size()[0]):
    t.add_item(i, store_feature_maps[i])
t.build(150) # 150 trees, more trees gives higher precision when querying
t.save('store.ann')

# Example
u = AnnoyIndex(example_feature_maps.size()[1], 'dot')
u.load('store.ann')
recommendations = u.get_nns_by_item(0, 5)
print(u.get_nns_by_item(0, 5, include_distances=True))
for recommendation in recommendations:
    print(image_df['image_label'][recommendation])


# Sklearn KNN
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree
rng = np.random.RandomState(42)
tree = BallTree(store_feature_maps)
dist, ind = tree.query(example_feature_maps, k=5)
print(ind) # indices of 5 closest neighbors
print(dist) # distances to 5 closest neighbors
for i in ind:
    for idx in i:
        print(image_df['image_label'][idx])


# # Sklearn KMeans
# kmeans_df = image_df
# kmeans_model = KMeans(n_clusters=15).fit(store_feature_maps)
# kmeans_df['cluster_labels'] = kmeans_model.labels_
# y = kmeans_model.predict(example_feature_maps)
# print("Predicted example label: {}".format(int(y)))
# print(kmeans_df[kmeans_df['cluster_labels'] == int(y)])
