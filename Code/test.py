import os

import torch
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from ast import literal_eval
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


# Load data
class TestDataset(Dataset):
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
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    dataset = TestDataset(data_path, img_dir, img_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    return loader


TEST_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/test_ann.csv"
TEST_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_test"
test_data_loader = create_data_loader(TEST_DATA_PATH, TEST_IMG_DIR, batch_size=64)


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


MODEL_NAME = "mobilenet_model"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN()
model.load_state_dict(torch.load("{}.pt".format(MODEL_NAME)))
model.eval()

with torch.no_grad():
    for idx, (feat, tar) in enumerate(test_data_loader):
        test_input, test_target = feat.to(device), tar.to(device)
        logit_test_output = model(test_input)
        sigmoid_test_output = torch.sigmoid(logit_test_output)
        y_pred = (sigmoid_test_output > 0.5).float()

        bceloss = nn.BCELoss()
        bceloss_output = bceloss(sigmoid_test_output, tar)
        # Write to file
        with open("{}_test.txt".format(MODEL_NAME), "a") as file:
            file.write('Test BCELoss: {} \n'.format(bceloss_output))
        # Print status
        print('Test BCELoss: {}'.format(bceloss_output))

        cpu_tar = tar.cpu().numpy()
        cpu_val_output = y_pred.cpu().numpy()
        f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
        # Write to file
        with open("{}_test.txt".format(MODEL_NAME), "a") as file:
            file.write('Test F1 Score: {} \n'.format(f1))
        # Print status
        print('Test F1 Score: {}'.format(f1))
