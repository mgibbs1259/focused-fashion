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


EXAMPLE_DIR = "/home/ubuntu/Final-Project-Group8/Data/example_images/jessica"
EXAMPLE_TYPE = "jessica"
STORE_DIR = "/home/ubuntu/Final-Project-Group8/Data/banana_republic_images"


def generate_image_mapping_csv(image_dir, csv_name):
    """Returns a csv file mapping image_id and image_label for a given image directory."""
    image_id = [i for i in range(len(os.listdir(image_dir)))]
    image_label = os.listdir(image_dir)
    image_dict = {"image_id": image_id, "image_label": image_label}
    image_df = pd.DataFrame(image_dict)
    image_df.to_csv("/home/ubuntu/Final-Project-Group8/Data/{}.csv".format(csv_name))


# Generate image mapping csv files
generate_image_mapping_csv(EXAMPLE_DIR, "{}_example_image".format(EXAMPLE_TYPE))
EXAMPLE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Data/{}_example_image.csv".format(EXAMPLE_TYPE)
generate_image_mapping_csv(STORE_DIR, "banana_republic_images")
STORE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Data/banana_republic_images.csv"


class FashionDataset(Dataset):
    """A dataset for the fashion images and fashion image labels.

    Arguments:
        Image directory path
        Image transformation
        Information csv path
    """
    def __init__(self, img_dir_path, img_transform, info_csv_path):
        self.img_dir_path = img_dir_path
        self.img_transform = img_transform
        self.info_csv_path = info_csv_path
        self.df = pd.read_csv(self.info_csv_path, header=0).reset_index(drop=True)
        self.img_id = self.df['image_id']
        self.img_label = self.df['image_label']

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir_path, str(self.img_label[index])))
        img = img.convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return index, img

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
store_loader = create_data_loader(STORE_DIR, STORE_CSV_PATH, batch_size=500)
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


MODEL_NAME = "mobilenet_model"


model = CNN().to(device)
model.load_state_dict(torch.load("/home/ubuntu/Final-Project-Group8/Models/{}.pt".format(MODEL_NAME)))
model.eval()


# Get example feature maps
with torch.no_grad():
    for batch_idx, (img_idx, img_features) in enumerate(example_loader):
        feat = img_features.to(device)
        x = model.features(feat)
        example_feature_maps = model.linear(x.view(len(x), -1))


# Get store feature maps
with torch.no_grad():
    for batch_idx, (img_idx, img_features) in enumerate(store_loader):
        feat = img_features.to(device)
        x = model.features(feat)
        store_feature_maps = model.linear(x.view(len(x), -1))


# Scikit-learn KNN
with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "w") as file:
    file.write("Model: {}, Example Type: {}, Scikit-learn KNN \n".format(MODEL_NAME, EXAMPLE_TYPE))
rng = np.random.RandomState(42)
tree = BallTree(store_feature_maps)
dist, ind = tree.query(example_feature_maps, k=5)
print(dist) # Distances to 5 closest neighbors
with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "a") as file:
    file.write("Distance to 5 closest neighbors: {} \n".format(dist))
for i in ind:
    for idx in i:
        print(store_df['image_label'][idx])
        with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "a") as file:
            file.write("Recommendation: {} \n".format(store_df['image_label'][idx]))


# Annoy KNN
with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "a") as file:
    file.write("Model: {}, Example Type: {}, Annoy KNN \n".format(MODEL_NAME, EXAMPLE_TYPE))
# Index store
store_item = AnnoyIndex(store_feature_maps.size()[1], 'angular')
for i in range(store_feature_maps.size()[0]):
    store_item.add_item(i, store_feature_maps[i])
store_item.build(500) # More trees gives higher precision when querying
store_item.save('store_items.ann')
# Index example
example_item = AnnoyIndex(example_feature_maps.size()[1], 'angular')
example_item.load('store_items.ann')
recommendations = example_item.get_nns_by_item(0, 5)
dist_recommendations = example_item.get_nns_by_item(0, 5, include_distances=True)
print(dist_recommendations)
with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "a") as file:
    file.write("Distance to 5 closest neighbors: {} \n".format(dist_recommendations))
for recommendation in recommendations:
    print(store_df['image_label'][recommendation])
    with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "a") as file:
        file.write("Recommendation: {} \n".format(store_df['image_label'][recommendation]))
