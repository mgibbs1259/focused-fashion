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
