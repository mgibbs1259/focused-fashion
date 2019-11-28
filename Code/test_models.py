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


TEST_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/test"
TEST_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/test.csv"


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
        self.df = pd.read_csv(self.info_csv_path, header=0, names=['label_id', 'image_id']).reset_index(drop=True)
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


def create_data_loader(img_dir, info_csv_path, batch_size):
    """Returns a data loader for the model."""
    img_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "mobilenet_model"
LR = 0.01
N_EPOCHS = 3
BATCH_SIZE = 64


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.mobilenet_model = models.mobilenet_v2(pretrained=True)
        n = 0
        for child in self.mobilenet_model.children():
            n += 1
            if n < 2:
                for param in child.parameters():
                    param.requires_grad = False
        self.features = nn.Sequential(*list(self.mobilenet_model.children())[:-1])
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


model = CNN()
model.load_state_dict(torch.load("/home/ubuntu/Final-Project-Group8/Models/{}.pt".format(MODEL_NAME)))
model.eval()


test_data_loader = create_data_loader(TEST_IMG_DIR, TEST_INFO_PATH, 500)


with open("{}_test.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}".format(MODEL_NAME))


with torch.no_grad():
    for idx, (feat, tar) in enumerate(test_data_loader):
        test_input, test_target = feat.to(device), tar.to(device)
        logit_test_output = model(test_input)
        sigmoid_test_output = torch.sigmoid(logit_test_output)
        y_pred = (sigmoid_test_output > 0.5).float()
        cpu_tar = tar.cpu().numpy()
        cpu_val_output = y_pred.cpu().numpy()
        f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
        # Write test F1 to file
        with open("{}_test.txt".format(MODEL_NAME), "a") as file:
            file.write('Test F1 Score: {} \n'.format(f1))
        # Print test F1
        print('Test F1 Score: {}'.format(f1))
