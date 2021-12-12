## Code Cell 1.1

import time
import copy
import sys
from collections import OrderedDict
from captum.attr import LayerIntegratedGradients
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
from torchinfo import summary
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import captum
from captum.attr import IntegratedGradients
import pickle
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Train classification network')
parser.add_argument('--rate',
                    required=True,
                    type=float)
args, _ = parser.parse_known_args()
print(f"Running experiment for")
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

# Load testing data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
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


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3,  stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 8 * 8, 120)  # 5x5 image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#         self.classifier = nn.Linear(256, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = x.view(-1, int(x.nelement() / x.shape[0]))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
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
            conv_block(256, 256)
        )
        self.masks = [  # masks for pruning individual layers
            torch.ones([32, 32, 32]),
            torch.ones([32, 32, 32]),
            torch.ones([64, 16, 16]),
            torch.ones([64, 16, 16]),
            torch.ones([64, 16, 16]),
            torch.ones([128, 8, 8]),
            torch.ones([128, 8, 8]),
            torch.ones([256, 8, 8]),
            torch.ones([256, 8, 8]),
        ]
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 10)
        self.pruning = False

    def forward(self, x):
        '''
        The forward function is called automatically by the model when it is
        given an input image. It first applies the 8 convolution layers, then
        finally the single classifier layer.
        '''

        if self.pruning:  # forward pass when pruning is activated
            out = self.model[0](x) * self.masks[0].to(torch.float32)  # prune first layer
            for i in range(1, len(self.model)):
                out = self.model[i](out).to(torch.float32)
                out = out * self.masks[i]  # prune deeper layers
            h = self.adaptive_pool(out).to(torch.float32)
            B, C, _, _ = h.shape
            h = h.view(B, C)
            return self.classifier(h)
        else:  # forward pass without pruning
            out = self.model[0](x)
            for i in range(1, len(self.model)):
                out = self.model[i](out)
            h = self.adaptive_pool(out).to(torch.float32)
            B, C, _, _ = h.shape
            h = h.view(B, C)
            return self.classifier(h)

    # TODO
    #   * horizontal line - percentage of pruned nodes
    #   * vertical line - accuracy fine tuned model
    #   * we also need a metric on how much nodes we have pruned.
    #   (--> number of inputs pruned?)


if __name__ == '__main__':
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
            torch.save(state, f'./checkpoint/ckpt_pruned_{args.rate}.pth')
            best_acc = acc

    print("Device: ", device)
    net = ConvNet().to(device=device)

    summary(net, input_size=(batch_size, 3, 32, 32))  # print summary of network architecture

    module = net.model[-1]
    pre_trained_model = torch.load("checkpoint/ckpt.pth", map_location=device)
    net.load_state_dict(pre_trained_model['net'])


    def forward_func(image):
        return net(image)


    # get a smaller subset of the test set to compute attributions
    subset_size = 20
    indices = torch.randperm(len(testset))[:subset_size]
    test_subset = torch.utils.data.Subset(testset, indices)

    # --------------- integrated gradient pruning section ----------------
    baseline = 0
    threshold = args.rate
    best_accuracy_tracker = []
    convergence = []
    percentage_pruned = []
    torch.cuda.empty_cache()
    perc_pruned_per_layer = []
    for i in range(0, len(net.model)):  # compute attributions for each layer
        if 0:
            layer = net.model[i]
            ig = LayerIntegratedGradients(forward_func=forward_func, layer=layer)
            attribution_container = []
            for inputs, targets in tqdm(testloader):  # iterate over all batches of the testset
                inputs = inputs.to(device)
                batch_attributions, approximation_error = ig.attribute(inputs, baseline,
                                                                       method='gausslegendre',
                                                                       return_convergence_delta=True,
                                                                       target=3)
                attribution_container.append(batch_attributions)

            attributions = torch.cat(attribution_container, dim=0)
            attributions = attributions.mean(dim=0)
            attributions -= torch.min(attributions)  # attributions.min(1, keepdim=True)[0]  # normalize attributions
            attributions /= torch.max(attributions)  # attributions.max(1, keepdim=True)[0]
            print('Computed attributions for layer {}'.format(i))
            torch.save(attributions, f'./checkpoint/attribution_layer{i}.pth')
        else:
            attributions = torch.load(f'./checkpoint/attribution_layer{i}.pth', map_location=device)
        
        # bernoulli distribution masks
        if 0:
            mask = torch.where(attributions > threshold, 1.0, 0.0)
            threshold_ = threshold
            size = float(torch.numel(mask))
            ones = float(torch.count_nonzero(mask))
            perc_pruned = ((size - ones) / size)
            perc_pruned_per_layer.append(perc_pruned)
        else:
            perc_pruned = threshold  # now threshold means percentage
            casualty = int(attributions.numel() * perc_pruned)
            threshold_ = attributions.flatten().sort()[0][casualty]
            mask = torch.where(attributions > threshold_, 1.0, 0.0)    
            perc_pruned_per_layer.append(perc_pruned)
            
        print('threshold {} pruned {}% on layer {}'.format(threshold_, perc_pruned * 100.0, i))

        net.masks[i] = mask.to(device)
        # net.masks[i] = torch.bernoulli(attributions).squeeze(0).to(torch.float32)
    # --------------------------------------------------------------------
    percentage_pruned.append(perc_pruned_per_layer)
    with open(f'perc_pruned_{args.rate}.pkl', 'wb') as fp:
        pickle.dump(percentage_pruned, fp)

        ## Training pruned model
        # tracks the highest accuracy observed so far
        net.pruning = True
        best_acc = 0

        torch.manual_seed(43)  # to give stable randomness
        # PART 1.1: set the learning rate (lr) used in the optimizer.
        lr = 0.001
        epochs = 20

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                    weight_decay=5e-4)

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

        # print('Training for {} epochs, with learning rate {} and milestones {}'.format(
        #         epochs, lr, milestones))

        start_time = time.time()
        for epoch in range(0, epochs):
            train(net, epoch, train_loss_tracker, train_acc_tracker)
            test(net, epoch, test_loss_tracker, test_acc_tracker)
            scheduler.step()

        total_time = time.time() - start_time
        print('Total retraining time: {} seconds'.format(total_time))
        print('Threshold {} lead to {} % accuracy after fine tuning'.format(threshold, best_acc))
        best_accuracy_tracker.append(best_acc)
        convergence.append([train_loss_tracker, train_acc_tracker, test_loss_tracker, test_acc_tracker])
    
    with open(f'threshold_experiment_accuracies_rate{args.rate}.pkl', 'wb') as fp:
        pickle.dump([best_accuracy_tracker, convergence], fp)


#    with open('pruning_50_epochs_0_1_last_2nd_layer.pkl', 'wb') as fp:
#        pickle.dump(test_acc_tracker, fp)
# epochs = 10 # Part 1.1: Change to 1000 epochs
# # net = LeNet().to(device=device)
# criterion = nn.CrossEntropyLoss()
# milestones = [250, 500, 750]
#
# # Part 1.1: Train the device model for 100 epochs and plot the result
# # Standard Training Loop
# start_time = time.time()
# for epoch in range(epochs):
#     train(epoch, device)
#     # To speed up running time, only evaluate the test set every 10 epochs
#     if epoch > 0 and epoch % 10 == 0:
#         test(epoch, device)
#     device['scheduler'].step()
#
#
# total_time = time.time() - start_time
# print('Total training time: {} seconds'.format(total_time))
