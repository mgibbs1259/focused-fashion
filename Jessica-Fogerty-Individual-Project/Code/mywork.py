#I worked on these python scripts:
import os

import pandas as pd
from PIL import Image


def obtain_smallest_image_size(img_dir):
    """Returns the dimensions of the smallest image in terms of area."""
    image_sizes = {}
    for image in os.listdir(img_dir):
        print(image)
        try:
            with Image.open(os.path.join(img_dir, image)) as img:
                width, height = img.size
                image_sizes[image] = (width, height, width * height)
                print(image_sizes[image])
        except:
            with open('bad_images.txt', 'a') as file:
                file.write(image)
    smallest_image = min(image_sizes, key=lambda k: image_sizes[k][2])
    return image_sizes[smallest_image]


def check_label_balance(image_dir_path, image_info_path):
    """Prints False if there are missing images from the csv files."""
    images = []
    images += [each for each in os.listdir(image_dir_path)]
    images.sort()
    df = pd.read_csv(image_info_path)
    df['file name'] = df['imageId'].apply(lambda x: '{}.jpg'.format(x))
    mask = df['file name'].isin(images)
    return print(mask.value_counts())

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
    img_transform = transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_1"
LR = 5e-3
N_EPOCHS = 5
BATCH_SIZE = 256
DROPOUT = 0.10


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
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
    img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_3"
LR = 5e-5
N_EPOCHS = 10
BATCH_SIZE = 200
DROPOUT = 0.01


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(35)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(70)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(10080, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
    img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_4"
LR = 5e-7
N_EPOCHS = 10
BATCH_SIZE = 280
DROPOUT = 0.005


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(35)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(70)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(10080, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
    img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_5"
LR = 5e-5
N_EPOCHS = 25
BATCH_SIZE = 200
DROPOUT = 0.01


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(35)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(70)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(10080, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
    img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_5"
LR = 5e-5
N_EPOCHS = 25
BATCH_SIZE = 200
DROPOUT = 0.01


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(35)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(70)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(10080, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
    img_transform = transforms.Compose([transforms.Resize((50, 50), interpolation=Image.BICUBIC),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor()])
    img_dataset = FashionDataset(img_dir, img_transform, info_csv_path)
    data_loader = DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return data_loader


MODEL_NAME = "jessica_model_7"
LR = 5e-5
N_EPOCHS = 25
BATCH_SIZE = 200
DROPOUT = 0.01


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 35, (3, 3), stride=1, padding=1)
        self.convnorm1 = nn.BatchNorm2d(35)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2)

        self.conv2 = nn.Conv2d(35, 70, (3, 3), stride=1, padding=1)
        self.convnorm2 = nn.BatchNorm2d(70)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.linear1 = nn.Linear(10080, 256)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 149)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
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
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
from wordcloud import WordCloud
from sklearn.preprocessing import MultiLabelBinarizer


#Load Data
df_txt = pd.read_csv('new_model_number_6.txt', header=None)
df1 = pd.read_csv("train_ann_drop.csv")
label_map = pd.read_csv("label_map.csv")

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

data_path = "/home/ubuntu/Final-Project-Group8/Final-Project-Group8/Code/"

train={}
test={}
validation={}
with open('%s/train.json'%(data_path)) as json_data:
    train= json.load(json_data)
with open('%s/test.json'%(data_path)) as json_data:
    test= json.load(json_data)
with open('%s/validation.json'%(data_path)) as json_data:
    validation = json.load(json_data)

print('Train No. of images: %d'%(len(train['images'])))
print('Test No. of images: %d'%(len(test['images'])))
print('Validation No. of images: %d'%(len(validation['images'])))

# Train
train_img_url=train['images']
train_img_url=pd.DataFrame(train_img_url)
train_ann=train['annotations']
train_ann=pd.DataFrame(train_ann)
train=pd.merge(train_img_url, train_ann, on='imageId', how='inner')

# Test
test=pd.DataFrame(test['images'])

# Validation
val_img_url=validation['images']
val_img_url=pd.DataFrame(val_img_url)
val_ann=validation['annotations']
val_ann=pd.DataFrame(val_ann)
validation=pd.merge(val_img_url, val_ann, on='imageId', how='inner')

# Create test with labels
test2 = train_ann[700000:730000]


# Only keep first 500000 rows
train_ann = train_ann[:500000]

drop = pd.read_csv('/home/ubuntu/Final-Project-Group8/Final-Project-Group8/drop_2.csv')

drop = drop['imageId'].tolist()

train_ann['imageId'] = train_ann['imageId'].astype(int)

train_ann = train_ann[~train_ann['imageId'].isin(drop)]


datas = {'Train': train, 'Test': test, 'Validation': validation}
for data in datas.values():
    data['imageId'] = data['imageId'].astype(np.uint32)

# Write out val_ann and train_ann as csv
# val_ann.to_csv("validation.csv")
# test2.to_csv("test.csv")

# Convert labels using the multilabelbinarizer
mlb = MultiLabelBinarizer()
train_label = mlb.fit_transform(train['labelId'])
validation_label = mlb.transform(validation['labelId'])
test_label = mlb.transform(test2['labelId'])
dummy_label_col = list(mlb.classes_)
print(dummy_label_col)

for data in [validation_label, train_label, test_label]:
    print(data.shape)

# Dataframes of the labels
train_label = pd.DataFrame(data=train_label, columns=list(mlb.classes_))
train_label.head()
# train_label.to_csv("train_label")
validation_label = pd.DataFrame(data=validation_label, columns=list(mlb.classes_))
validation_label.head()
# train_label.to_csv("validation_label")
test_label = pd.DataFrame(data=validation_label, columns=list(mlb.classes_))
test_label.head()
# train_label.to_csv("test_label")
def clean_txt_file(txt_file, new_txt_file):
    """Use this to clean up the .txt files for use
    Removes lines containing strings we wish to remove and outputs to new txt file.
    Load the new txt file to plot the data"""
    words_to_remove = ['MODEL_NAME', 'EPOCH', 'Validation']
    with open(txt_file) as oldfile, open(new_txt_file, 'w') as newfile:
        for line in oldfile:
            if not any(words_to_remove in line for words_to_remove in words_to_remove):
                newfile.write(line)


def plot_loss(df):
    """Plots the loss for each Batch"""
    df = df.rename(columns={0: "Batch number", 1: "Loss"})
    loss = df[['x','Loss']] = df['Loss'].str.split(':',expand=True)
    df = df.drop(['x'], axis=1)
    Batch_number = df[['x','Batch number']] = df['Batch number'].str.split(':',expand=True)
    df = df.drop(['x'], axis=1)
    df["Loss"] = pd.to_numeric(df["Loss"])
    ax = df.plot(lw = 1, colormap = 'jet',x='Batch number', y=['Loss'], figsize=(20,10), grid=True, title='Train Loss')
    ax.set_xlabel("Batch number")
    ax.set_ylabel("Loss")
    fig = ax.get_figure()
    fig.savefig("loss_plot.png")


def explore_labels(df):
    """Cleans data, creates histogram, gets descriptive stats of the number of
    per image"""
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    df['Length'] = df.labelId.apply(lambda x: len(x))
    df['labelId'] = df['labelId'].apply(literal_eval)
    x = df['Length'].describe()
    ax = df['Length'].hist(bins=25)
    fig = ax.get_figure()
    fig.savefig("label_histogram.png")
    return x


def prep_data_for_cloud(df, label_map):
    """Prepared data for cloud visualization"""
    df = df.drop(['Unnamed: 0'], axis=1)
    df = df.drop(['Unnamed: 0.1'], axis=1)
    df['labelId'] = df['labelId'].apply(literal_eval)
    mlb = MultiLabelBinarizer()
    Labels = mlb.fit_transform(df['labelId'])
    Labels = pd.DataFrame(data=Labels, columns=list(mlb.classes_))
    # Obtain count of each label
    Label_Count = Labels.sum()
    df2 = Label_Count.to_frame().reset_index()
    df2 = df2.rename(columns={0: "count", 'index': "Label"})
    label_map = label_map.rename(columns={"labelId": "Label"})
    df2["Label"] = df2["Label"].astype(int)
    # Merge the dataframes together
    mergeddf = df2.merge(label_map, on='Label')
    mergeddf = mergeddf.drop(['taskId'], axis=1)
    mergeddf = mergeddf.drop(['taskName'], axis=1)
    mergeddf = mergeddf.drop(['Label'], axis=1)
    return mergeddf


def create_cloud(mergeddf):
    """Create word cloud visualization"""
    #Create frequency dictionary
    d = {}
    for word, count in mergeddf.values:
        d[count] = word
    #Create word cloud visualization
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies = d)
    plt.figure()
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis = 'off'
    plt.savefig("word_cloud")
    plt.show()


def F1_hist(df):
    df_txt = df.rename(columns={0: "F1"})
    F1 = df_txt[['x','F1']] = df_txt['F1'].str.split(':', expand=True)
    df_txt = df_txt.drop(['x'], axis=1)
    df_txt["F1"] = pd.to_numeric(df_txt["F1"])
    x = df_txt.describe()
    df_txt.plot.box()
    plt.title('F1 scores')
    plt.ylabel('Score')
    plt.title("F1 Box Plot")
    plt.show
    plt.savefig("box_plot_f1_model_MobileNet")
    return x


clean_txt_file("model_number_6.txt", "new_model_number_6.txt")
plot_loss(df_txt)
explore_labels(df1)
mergeddf = prep_data_for_cloud(df1, label_map)
create_cloud(mergeddf)
F1_hist(df_txt)

#Mary and I collaborated equally on these scripts:
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
