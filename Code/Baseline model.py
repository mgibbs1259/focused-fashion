import numpy as np
import torch
from torch import nn
import torch.optim as optim

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

        self.conv2 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1) # Output (n_examples, 64, 16, 16)
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2) # Output (n_examples, 64, 8, 8)

        self.linear1 = nn.Linear(64*8*8, 256) # Input will be flattened to (n_examples, 64, 8, 8)
        self.linear1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(256, 228)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        print(torch.sigmoid(x))
        return torch.sigmoid(x)


DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"
data_loader = load_images.create_data_loader(DATA_PATH, IMG_DIR, BATCH_SIZE)


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.BCEWithLogitsLoss()


def train_model(epoch):
    loss_train = 0.0

    for idx, (data, target) in enumerate(data_loader):
        model_input, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        model_output = model(model_input)
        loss = criterion(model_output, target)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        if idx % 2000 == 1999:  # Print every 2000
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, idx + 1, loss_train / 2000))
            loss_train = 0.0

        print('Finished training')


for epoch in range(N_EPOCHS):
    train_model(epoch)
