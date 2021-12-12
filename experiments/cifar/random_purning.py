## Code Cell 1.1

import time
import copy
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import math
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import captum
from captum.attr import IntegratedGradients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data

## Code Cell 1.2

# Load training data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# Load testing data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)
print('Finished loading datasets!')


# Using CIFAR-10 again as in Assignment 1
def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    '''
    A nn.Sequential layer executes its arguments in sequential order. In
    this case, it performs Conv2d -> BatchNorm2d -> ReLU. This is a typical
    block of layers used in Convolutional Neural Networks (CNNs). The
    ConvNet implementation below stacks multiple instances of this three layer
    pattern in order to achieve over 90% classification accuracy on CIFAR-10.
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )



class ConvNet(nn.Module):
    '''
    A 9 layer CNN using the conv_block function above. Again, we use a
    nn.Sequential layer to build the entire model. The Conv2d layers get
    progressively larger (more filters) as the model gets deeper. This
    corresponds to spatial resolution getting smaller (via the stride=2 blocks),
    going from 32x32 -> 16x16 -> 8x8. The nn.AdaptiveAvgPool2d layer at the end
    of the model reduces the spatial resolution from 8x8 to 1x1 using a simple
    average across all the pixels in each channel. This is then fed to the
    single fully connected (linear) layer called classifier, which is the output
    prediction of the model.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 32),
            conv_block(32, 64, stride=2),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 128, stride=2),
            conv_block(128, 128),
            conv_block(128, 256),
            conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        '''
        The forward function is called automatically by the model when it is
        given an input image. It first applies the 8 convolution layers, then
        finally the single classifier layer.
        '''
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)


net = ConvNet().to(device=device)


# tracks the highest accuracy observed so far
best_acc = 0


def train(net, epoch, train_loss_tracker, train_acc_tracker):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # update optimizer state
        optimizer.step()
        # compute average loss
        train_loss += loss.item()
        train_loss_tracker.append(loss.item())
        loss = train_loss / (batch_idx + 1)
        # compute accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        # Print status
        sys.stdout.write(f'\rEpoch {epoch}: Train Loss: {loss:.3f}' +
                         f'| Train Acc: {acc:.3f}')
        sys.stdout.flush()
    train_acc_tracker.append(acc)
    sys.stdout.flush()


def test(net, epoch, test_loss_tracker, test_acc_tracker):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            test_loss_tracker.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()

    # Save checkpoint.
    acc = 100. * correct / total
    test_acc_tracker.append(acc)
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, './checkpoint/model_vision.pth')
        best_acc = acc


torch.manual_seed(43)  # to give stable randomness
# PART 1.1: set the learning rate (lr) used in the optimizer.
lr = 0.001
epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# PART 1.2: try different learning rate scheduler
scheduler_name = 'cosine_annealing'  # set this to 'multistep' or 'cosine_annealing'
if scheduler_name == 'multistep':
    milestones = [25]
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=gamma)
elif scheduler_name == 'cosine_annealing':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

else:
    NotImplementedError

# Records the training loss and training accuracy during training
train_loss_tracker, train_acc_tracker = [], []

# Records the test loss and test accuracy during training
test_loss_tracker, test_acc_tracker = [], []


start_time = time.time()
for epoch in range(0, epochs):
    train(net, epoch, train_loss_tracker, train_acc_tracker)
    # print(module.weight)
    test(net, epoch, test_loss_tracker, test_acc_tracker)
    scheduler.step()

# torch.save(net.state_dict(), 'checkpoint.pth')
# download checkpoint file
# files.download('checkpoint.pth')

total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))

