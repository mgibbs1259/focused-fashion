# I worked on these Python scripts

import os

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch import nn
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"


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
    img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "mary_model_1"
LR = 0.1
N_EPOCHS = 5
BATCH_SIZE = 1024
DROPOUT = 0.45


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (12, 12), stride=2, padding=1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(32, 64, (6, 6), stride=2, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(64*5*5, 1024)
        self.linear1_bn = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(1024, 149)

        self.relu = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.relu(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.relu(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.relu(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


train_data_loader = create_data_loader(TRAIN_IMG_DIR, TRAIN_INFO_PATH, BATCH_SIZE)
val_data_loader = create_data_loader(VAL_IMG_DIR, VAL_INFO_PATH, batch_size=len(os.listdir(VAL_IMG_DIR)))


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {}, DROPOUT: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                             BATCH_SIZE, DROPOUT))


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write model information to file
    with open("{}.txt".format(MODEL_NAME), "a") as file:
        file.write("EPOCH: {} \n".format(epoch))
    # Update epoch status
    epoch_status.set_description('Epoch: {}'.format(epoch))
    # Training
    model.train()
    loss_train = 0
    for train_idx, (features, target) in enumerate(train_data_loader):
        train_input, train_target = features.to(device), target.to(device)
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        # Write train loss to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print train loss
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            logit_val_output = model(val_input)
            sigmoid_val_output = torch.sigmoid(logit_val_output)
            y_pred = (sigmoid_val_output > 0.5).float()
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = y_pred.cpu().numpy()
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write validation F1 to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print validation F1
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))


import os

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch import nn
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"


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
    img_transform = transforms.Compose([transforms.Resize((120, 120), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "mary_model_2"
LR = 0.01
N_EPOCHS = 10
BATCH_SIZE = 1024
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


train_data_loader = create_data_loader(TRAIN_IMG_DIR, TRAIN_INFO_PATH, BATCH_SIZE)
val_data_loader = create_data_loader(VAL_IMG_DIR, VAL_INFO_PATH, batch_size=len(os.listdir(VAL_IMG_DIR)))


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {}, DROPOUT: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                             BATCH_SIZE, DROPOUT))


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write model information to file
    with open("{}.txt".format(MODEL_NAME), "a") as file:
        file.write("EPOCH: {} \n".format(epoch))
    # Update epoch status
    epoch_status.set_description('Epoch: {}'.format(epoch))
    # Training
    model.train()
    loss_train = 0
    for train_idx, (features, target) in enumerate(train_data_loader):
        train_input, train_target = features.to(device), target.to(device)
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        # Write train loss to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print train loss
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            logit_val_output = model(val_input)
            sigmoid_val_output = torch.sigmoid(logit_val_output)
            y_pred = (sigmoid_val_output > 0.5).float()
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = y_pred.cpu().numpy()
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write validation F1 to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print validation F1
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))


import os

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch import nn
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"


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
        self.x_train = self.df['image_id'].apply(literal_eval)
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
     img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                         transforms.ToTensor()])
     img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
     data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
     return data_loader


MODEL_NAME = "baseline_model"
LR = 0.01
N_EPOCHS = 5
BATCH_SIZE = 256
DROPOUT = 0.5


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(64*8*8, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 225)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return torch.sigmoid(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.BCELoss()


train_data_loader = create_data_loader(TRAIN_INFO_PATH, TRAIN_IMG_DIR, BATCH_SIZE)
val_data_loader = create_data_loader(VAL_INFO_PATH, VAL_IMG_DIR, batch_size=1000)


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {}, DROPOUT: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                             BATCH_SIZE, DROPOUT))


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write model information to file
    with open("{}.txt".format(MODEL_NAME), "a") as file:
        file.write("EPOCH: {} \n".format(epoch))
    # Update epoch status
    epoch_status.set_description('Epoch: {}'.format(epoch))
    # Training
    model.train()
    loss_train = 0
    for train_idx, (features, target) in enumerate(train_data_loader):
        train_input, train_target = features.to(device), target.to(device)
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        # Write train loss to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print train loss
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            logit_val_output = model(val_input)
            sigmoid_val_output = torch.sigmoid(logit_val_output)
            y_pred = (sigmoid_val_output > 0.5).float()
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = y_pred.cpu().numpy()
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write validation F1 to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print validation F1
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))


import os

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch import nn
from PIL import Image
from ast import literal_eval
from sklearn.metrics import f1_score
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"


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
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "mobilenet_model"
LR = 0.01
N_EPOCHS = 3
BATCH_SIZE = 64


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


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()


train_data_loader = create_data_loader(TRAIN_IMG_DIR, TRAIN_INFO_PATH, BATCH_SIZE)
val_data_loader = create_data_loader(VAL_IMG_DIR, VAL_INFO_PATH, batch_size=BATCH_SIZE)


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                BATCH_SIZE))


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write model information to file
    with open("{}.txt".format(MODEL_NAME), "a") as file:
        file.write("EPOCH: {} \n".format(epoch))
    # Update epoch status
    epoch_status.set_description('Epoch: {}'.format(epoch))
    # Training
    model.train()
    loss_train = 0
    for train_idx, (features, target) in enumerate(train_data_loader):
        train_input, train_target = features.to(device), target.to(device)
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        # Write train loss to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print train loss
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            logit_val_output = model(val_input)
            sigmoid_val_output = torch.sigmoid(logit_val_output)
            y_pred = (sigmoid_val_output > 0.5).float()
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = y_pred.cpu().numpy()
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write validation F1 to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print validation F1
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))


import os

import torch
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from ast import literal_eval
from sklearn.metrics import f1_score
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer



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


import os

import numpy as np
import pandas as pd
from PIL import Image
from annoy import AnnoyIndex
from torchvision import transforms
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset, DataLoader


EXAMPLE_DIR = "/home/ubuntu/Final-Project-Group8/Recommendations/example_images/jeans"
EXAMPLE_TYPE = "jeans"
STORE_DIR = "/home/ubuntu/Final-Project-Group8/Recommendations/banana_republic_images"


def generate_image_mapping_csv(image_dir, csv_name):
    """Returns a csv file mapping image_id and image_label for a given image directory."""
    image_id = [i for i in range(len(os.listdir(image_dir)))]
    image_label = os.listdir(image_dir)
    image_dict = {"image_id": image_id, "image_label": image_label}
    image_df = pd.DataFrame(image_dict)
    image_df.to_csv("/home/ubuntu/Final-Project-Group8/Recommendations/{}.csv".format(csv_name))


# Generate image mapping csv files
generate_image_mapping_csv(EXAMPLE_DIR, "{}_example_image".format(EXAMPLE_TYPE))
EXAMPLE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Recommendations/{}_example_image.csv".format(EXAMPLE_TYPE)
generate_image_mapping_csv(STORE_DIR, "banana_republic_images")
STORE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Recommendations/banana_republic_images.csv"


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
        img = np.asarray(img)
        return index, img

    def __len__(self):
        return self.img_id.shape[0]


def create_data_loader(img_dir, info_csv_path, batch_size):
     """Returns an image loader for the model."""
     img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC)])
     img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
     data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
     return data_loader


# Create data loader
example_loader = create_data_loader(EXAMPLE_DIR, EXAMPLE_CSV_PATH, batch_size=1)
store_loader = create_data_loader(STORE_DIR, STORE_CSV_PATH, batch_size=500)
store_df = pd.read_csv(STORE_CSV_PATH)


np.random.seed(42)
MODEL_NAME = "baseline"


# Get example flattened images
for batch_idx, (img_idx, img_features) in enumerate(example_loader):
    example_image = img_features.reshape((img_features.shape[0], -1))


# Get store flattened images
for batch_idx, (img_idx, img_features) in enumerate(store_loader):
    store_images = img_features.reshape((img_features.shape[0], -1))


# Scikit-learn KNN
with open("{}_{}_recommendations.txt".format(MODEL_NAME, EXAMPLE_TYPE), "w") as file:
    file.write("Model: {}, Example Type: {}, Scikit-learn KNN \n".format(MODEL_NAME, EXAMPLE_TYPE))
tree = BallTree(store_images)
dist, ind = tree.query(example_image, k=5)
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
store_item = AnnoyIndex(store_images.size()[1], 'angular')
for i in range(store_images.size()[0]):
    store_item.add_item(i, store_images[i])
store_item.build(500) # More trees gives higher precision when querying
store_item.save('store_items.ann')
# Index example
example_item = AnnoyIndex(example_image.size()[1], 'angular')
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


EXAMPLE_DIR = "/home/ubuntu/Final-Project-Group8/Recommendations/example_images/jeans"
EXAMPLE_TYPE = "jeans"
STORE_DIR = "/home/ubuntu/Final-Project-Group8/Recommendations/banana_republic_images"


def generate_image_mapping_csv(image_dir, csv_name):
    """Returns a csv file mapping image_id and image_label for a given image directory."""
    image_id = [i for i in range(len(os.listdir(image_dir)))]
    image_label = os.listdir(image_dir)
    image_dict = {"image_id": image_id, "image_label": image_label}
    image_df = pd.DataFrame(image_dict)
    image_df.to_csv("/home/ubuntu/Final-Project-Group8/Recommendations/{}.csv".format(csv_name))


# Generate image mapping csv files
generate_image_mapping_csv(EXAMPLE_DIR, "{}_example_image".format(EXAMPLE_TYPE))
EXAMPLE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Recommendations/{}_example_image.csv".format(EXAMPLE_TYPE)
generate_image_mapping_csv(STORE_DIR, "banana_republic_images")
STORE_CSV_PATH = "/home/ubuntu/Final-Project-Group8/Recommendations/banana_republic_images.csv"


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


# Jessica and I worked on these Python scripts together

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import json
import urllib3
import multiprocessing

import PIL
from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


TRAIN_JSON = "/home/ubuntu/Final-Project-Group8/Data/train.json"
TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
# Change to the validation.json for the validation set


def parse_dataset(_dataset, _outdir):
    """Parse the dataset to create a list of tuple containing absolute path and url of image."""
    _fnames_urls = []
    with open(_dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(_outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:500000]
# [:500000] is for the training set
# Change this to [700000:730000] for the test set


def download_image(fnames_and_urls):
    """Download image and save its with 90% quality as JPG format.
    Skip image downloading if image already exists at given path."""
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)


if __name__ == '__main__':
    # Parse train json
    fnames_urls = parse_dataset(TRAIN_JSON, TRAIN_IMG_DIR)
    # Download images
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)
    sys.exit(1)


import os

import tqdm
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch import nn
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer


TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"
TRAIN_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/train.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/validation"
VAL_INFO_PATH = "/home/ubuntu/Final-Project-Group8/Data/validation.csv"


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
        self.x_train = self.df['image_id'].apply(literal_eval)
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
     img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                         transforms.ToTensor()])
     img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
     data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
     return data_loader


MODEL_NAME = "baseline_model"
LR = 0.01
N_EPOCHS = 5
BATCH_SIZE = 256
DROPOUT = 0.5


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(64*8*8, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 225)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return torch.sigmoid(x)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = nn.BCELoss()


train_data_loader = create_data_loader(TRAIN_INFO_PATH, TRAIN_IMG_DIR, BATCH_SIZE)
val_data_loader = create_data_loader(VAL_INFO_PATH, VAL_IMG_DIR, batch_size=1000)


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {}, DROPOUT: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                             BATCH_SIZE, DROPOUT))


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write model information to file
    with open("{}.txt".format(MODEL_NAME), "a") as file:
        file.write("EPOCH: {} \n".format(epoch))
    # Update epoch status
    epoch_status.set_description('Epoch: {}'.format(epoch))
    # Training
    model.train()
    loss_train = 0
    for train_idx, (features, target) in enumerate(train_data_loader):
        train_input, train_target = features.to(device), target.to(device)
        optimizer.zero_grad()
        train_output = model(train_input)
        train_loss = criterion(train_output, train_target)
        train_loss.backward()
        optimizer.step()
        loss_train += train_loss.item()
        # Write train loss to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print train loss
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            logit_val_output = model(val_input)
            sigmoid_val_output = torch.sigmoid(logit_val_output)
            y_pred = (sigmoid_val_output > 0.5).float()
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = y_pred.cpu().numpy()
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write validation F1 to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print validation F1
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))


import time
import requests

from selenium import webdriver


def obtain_image_urls(driver):
    """Returns a list of image urls."""
    images = driver.find_elements_by_tag_name("img")
    image_links = []
    for image in images:
        image_url = (image.get_attribute("src"))
        if image_url.endswith(".jpg"):
            if image_url not in image_links:
                image_links.append(image_url)
                print(image_url)
    return image_links


def scrape_image_urls(driver, url_list):
    """Returns a set of all image urls scraped from a given list of urls."""
    try:
        image_urls = []
        for url in url_list:
            driver.get(url)
            time.sleep(3)
            driver.find_element_by_css_selector(".universal-modal__close-button").click()
            time.sleep(3)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*25/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*50/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight*75/100);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5)
            image_urls += obtain_image_urls(driver)
            driver.close()
    except:
        driver.close()
    image_urls = set(image_urls)
    return image_urls


def save_images(image_urls, image_urls_type):
    """Saves the images from a given set of image urls."""
    i = 1
    for url in image_urls:
        img_data = requests.get(url).content
        f = open("{}{}.jpg".format(i, image_urls_type), "wb")
        f.write(img_data)
        f.close()
        i += 1


if __name__ == '__main__':
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-notifications")
    driver = webdriver.Chrome(chrome_options=chrome_options)
    url_list = [
        "https://bananarepublic.gap.com/browse/category.do?cid=69883&mlink=5001,,flyout_women_apparel_Dresses&clink=15682852"]
    image_urls = scrape_image_urls(driver, url_list)
    save_images(image_urls, "sweater")
