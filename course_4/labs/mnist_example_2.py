import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import copy
import math
import os
import sys
import progressbar
import numpy as np


# Hyperparameters
num_epochs = 5
num_classes = 10
batch_size = 32
learning_rate = 0.001


# Data transformation
trans = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
        ])


# MNIST dataset
DATA_PATH = '/data/mnist'
MODEL_STORE_PATH = 'saved_models'
DOWNLOAD_DATASET = not os.path.exists(os.path.join(DATA_PATH, "MNIST"))

data_parts = ['train', 'valid']
mnist_datasets = dict()
mnist_datasets['train'] = torchvision.datasets.MNIST(
    root=DATA_PATH, train=True, transform=trans, download=DOWNLOAD_DATASET)
mnist_datasets['valid'] = torchvision.datasets.MNIST(
    root=DATA_PATH, train=False, transform=trans, download=DOWNLOAD_DATASET)

dataloaders = {p: DataLoader(mnist_datasets[p], batch_size=batch_size,
        shuffle=True, num_workers=4) for p in data_parts}

dataset_sizes = {p: len(mnist_datasets[p]) for p in data_parts}
num_batch = dict()
num_batch['train'] = math.ceil(dataset_sizes['train'] / batch_size)
num_batch['valid'] = math.ceil(dataset_sizes['valid'] / batch_size)
print(num_batch)


x, y = mnist_datasets['valid'][0] #sample0
print("The sample #0:", x.shape, y)
# MNIST image - 28×28 pixel

# Conv2d parameters
# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, 
# dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
# Example: nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2)

# Neural Network
class NeuralNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop2 = nn.Dropout2d()
        self.fc1 = nn.Linear(3  20, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop2(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        # Each epoch has a training and validation phase
        for phase in data_parts:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

            for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                bar.update(i_batch)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                #print('epoch {} [{}]: {}/{}'.format(epoch, phase, i_batch, num_batch[phase]))
                #print('preds: ', preds)
                #print('labels:', labels.data)
                #print('match: ', int(torch.sum(preds == labels.data)))

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            bar.finish()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('Epoch {} [{}]: loss={:.4f}, acc={:.4f}' .
                format(epoch, phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = NeuralNetworkModel()
    model = model.to(device)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Initialize optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    criterion = nn.CrossEntropyLoss()
    model = train_model(model, criterion, optimizer, exp_lr_scheduler,
    num_epochs=num_epochs)

    #PATH = 'saved/model'
    #torch.save(model, PATH)  