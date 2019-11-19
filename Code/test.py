import os

import torch
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TestDataset(Dataset):
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
    img_transform = transforms.Compose([transforms.Resize((120, 120), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    dataset = TestDataset(img_dir, img_transform, data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    return loader


TEST_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/test_ann.csv"
TEST_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_test"
test_data_loader = create_data_loader(TEST_DATA_PATH, TEST_IMG_DIR, batch_size=len(os.listdir(TEST_IMG_DIR)))


# Load model
DROPOUT = 0.50

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (12, 12), stride=2, padding=1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(32, 64, (8, 8), stride=2, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv3 = nn.Conv2d(64, 128, (4, 4), stride=2, padding=1)
        self.convnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(128*1*1, 1024)
        self.linear1_bn = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(1024, 149)

        self.relu = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.relu(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.relu(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.relu(self.conv3(x))))
        x = self.drop(self.linear1_bn(self.relu(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return x


MODEL_NAME = "model_number_2"
model = CNN()
model.load_state_dict(torch.load("{}.pt".format(MODEL_NAME)))
model.eval()
with torch.no_grad():
    for idx, (feat, tar) in enumerate(test_data_loader):
        test_input, test_target = feat.to(device), tar.to(device)
        logit_val_output = model(test_input)
        sigmoid_val_output = torch.sigmoid(logit_val_output)
        cpu_tar = tar.cpu().numpy()
        cpu_val_output = np.where(sigmoid_val_output.cpu().numpy() > 0.5, 1, 0)
        f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
        # Write to file
        with open("test_{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Test F1 Score: {} \n'.format(f1))
        # Print status
        print('Test F1 Score: {}'.format(f1))
