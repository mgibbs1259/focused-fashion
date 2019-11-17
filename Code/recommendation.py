import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import KMeans


# Read in image of interest
img = Image.open("Blouses/3blouses.jpg")
img = img.resize((100, 100))
img_array = np.array(img)
print(img_array.shape)


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


# model = CNN()
# model.load_state_dict(torch.load("model_number_1.pt"))
# x = torch.zeros((BATCH_SIZE, 3, 100, 100))
# features = extract_feature_maps(x, model)


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
