import os

import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch import nn
from sklearn.metrics import f1_score

from Data import load_images


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Change this
MODEL_NAME = "model_number_1"


LR = 5e-2
N_EPOCHS = 5
BATCH_SIZE = 256
DROPOUT = 0.5


with open("{}.txt".format(MODEL_NAME), "a") as file:
    file.write("MODEL_NAME: {}, LR: {}, N_EPOCHS: {}, BATCH_SIZE: {}, DROPOUT: {} \n".format(MODEL_NAME, LR, N_EPOCHS,
                                                                                             BATCH_SIZE, DROPOUT))


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
        self.linear2 = nn.Linear(256, 224)

        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        x = self.linear2(x)
        return torch.sigmoid(x)


TRAIN_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"
train_data_loader = load_images.create_data_loader(TRAIN_DATA_PATH, TRAIN_IMG_DIR, BATCH_SIZE)


VAL_DATA_PATH = "/home/ubuntu/Final-Project-Group8/Data/val_ann.csv"
VAL_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"
val_data_loader = load_images.create_data_loader(VAL_DATA_PATH, VAL_IMG_DIR, batch_size=len(os.listdir(VAL_IMG_DIR)))


model = CNN().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
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
            file.write('Train Index: {}, Train Loss: {} \n'.format(train_idx, train_loss))
        # Print status
        print('Train Index: {}, Train Loss: {}'.format(train_idx, train_loss))
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
torch.save(model.state_dict(), 'baseline_model.pt')
