import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def resize_feature_maps():
    """Returns resized feature maps."""



def kmeans_clustering_elbow_curve(X):
    """Shows an elbow curve plot to determine the appropriate number of k-means clusters."""
    distorsions = []
    for k in range(1, 20):
        kmeans_model = KMeans(n_clusters=k)
        kmeans_model.fit(X)
        distorsions.append(kmeans_model.inertia_)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(1, 4), distorsions)
    plt.title('Elbow Curve')
    plt.show()


def kmeans_clustering(X, clusters=10):
    """Returns the kmeans model and predicted values."""
    kmeans_model = KMeans(n_clusters=clusters).fit(X)
    predict_values = kmeans_model.predict(X)
    return kmeans_model, predict_values


