import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as functional

from Data import load_images


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


LR = 5e-2
N_EPOCHS = 5
BATCH_SIZE = 256
DROPOUT = 0.5


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1) # Output (n_examples, 32, 32, 32)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2), stride=2) # Output (n_examples, 32, 16, 16)

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stide=1, padding=1) # Output (n_examples, 64, 16, 16)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # Output (n_examples, 64, 8, 8)

        self.linear1 = nn.Linear(64*8*8, ?)
        self.linear1_bn = nn.BatchNorm1d(?)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(?, ?)
        self.act = torch.relu


    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))

        return functional.sigmoid(x)


DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"
data_loader = load_images.create_data_loader(DATA_PATH, IMG_DIR, BATCH_SIZE)


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train_model(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.data[0]))


for epoch in range(N_EPOCHS):
    train_model(epoch)
