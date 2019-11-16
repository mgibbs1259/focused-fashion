import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
from PIL import Image
from ast import literal_eval
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from sklearn.metrics import f1_score


class FashionDataset(Dataset):
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
     img_transform = transforms.Compose([transforms.Resize((100, 100), interpolation=Image.BICUBIC),
                                         transforms.ToTensor()])
     dataset = FashionDataset(data_path, img_dir, img_transform)
     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
     return loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Change this
MODEL_NAME = "model_number_3"


LR = 0.01
N_EPOCHS = 5
BATCH_SIZE = 1024


with open("{}.txt".format(MODEL_NAME), "w") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                BATCH_SIZE))


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5), stride=1, padding=1) # Output (98, 98)
        self.convnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, (5, 5), stride=1, padding=1) # Output (96, 96)
        self.convnorm2 = nn.BatchNorm2d(64)

        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.linear = nn.Linear(64*48*48, 149)

        self.relu = torch.relu

    def forward(self, x):
        x = self.convnorm1(self.relu(self.conv1(x)))
        x = self.convnorm2(self.relu(self.conv2(x)))
        x = self.convnorm3(self.relu(self.conv3(x)))
        x = self.convnorm4(self.relu(self.conv4(x)))
        x = self.pool(x)
        x = self.linear(x.view(len(x), -1))
        return x


TRAIN_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/train_ann.csv"
TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_train"
train_data_loader = create_data_loader(TRAIN_DATA_PATH, TRAIN_IMG_DIR, BATCH_SIZE)


VAL_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"
val_data_loader = create_data_loader(VAL_DATA_PATH, VAL_IMG_DIR, batch_size=len(os.listdir(VAL_IMG_DIR)))


model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


# Define epoch status
epoch_status = tqdm.tqdm(total=N_EPOCHS, desc='Epoch', position=0)
for epoch in range(N_EPOCHS):
    # Write to file
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
        # Write to file
        with open("{}.txt".format(MODEL_NAME), "a") as file:
            file.write('Batch Number: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print status
        print('Batch Number: {}, Train Loss: {}'.format(train_idx, train_loss))
    # Validation
    model.eval()
    with torch.no_grad():
        for val_idx, (feat, tar) in enumerate(val_data_loader):
            val_input, val_target = feat.to(device), tar.to(device)
            val_output = model(val_input)
            cpu_tar = tar.cpu().numpy()
            cpu_val_output = np.where(val_output.cpu().numpy() > 0.5, 1, 0)
            f1 = f1_score(cpu_tar, cpu_val_output, average='micro')
            # Write to file
            with open("{}.txt".format(MODEL_NAME), "a") as file:
                file.write('Validation F1 Score: {} \n'.format(f1))
            # Print status
            print('Validation F1 Score: {}'.format(f1))
    # Update epoch status
    epoch_status.update(1)


# Save model
torch.save(model.state_dict(), '{}.pt'.format(MODEL_NAME))
