## Code Cell 1.1

import time
import copy
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Using CIFAR-10 again as in Assignment 1
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

# Load testing data
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


# Using same ConvNet as in Assignment 1
def conv_block(in_channels, out_channels, kernel_size=3, stride=1,
               padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class ConvNet(nn.Module):
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
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)


## Code Cell 1.2
import copy


class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, torch.tensor(label)


def create_device(net, device_id, trainset, data_idxs, lr=0.1,
                  milestones=None, batch_size=128):
    if milestones == None:
        milestones = [25, 50, 75]

    device_net = copy.deepcopy(net)
    optimizer = torch.optim.SGD(device_net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.1)
    device_trainset = DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size=batch_size,
                                                     shuffle=True)
    return {
        'net': device_net,
        'id': device_id,
        'dataloader': device_trainloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_tracker': [],
        'train_acc_tracker': [],
        'test_loss_tracker': [],
        'test_acc_tracker': [],
    }


def train(epoch, device):
    net.train()
    device['net'].train()
    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(device['dataloader']):
        inputs, targets = inputs.cuda(), targets.cuda()
        device['optimizer'].zero_grad()
        outputs = device['net'](inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        device['optimizer'].step()
        train_loss += loss.item()
        device['train_loss_tracker'].append(loss.item())
        loss = train_loss / (batch_idx + 1)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
        dev_id = device['id']
        sys.stdout.write(f'\r(Device {dev_id}/Epoch {epoch}) ' +
                         f'Train Loss: {loss:.3f} | Train Acc: {acc:.3f}')
        sys.stdout.flush()
    device['train_acc_tracker'].append(acc)
    sys.stdout.flush()


def test(epoch, device):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = device['net'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()
    acc = 100. * correct / total
    device['test_acc_tracker'].append(acc)


## Code Cell 1.3
import random


def iid_sampler(dataset, num_devices, data_pct):
    '''
    dataset: PyTorch Dataset (e.g., CIFAR-10 training set)
    num_devices: integer number of devices to create subsets for
    data_pct: percentage of training samples to give each device
              e.g., 0.1 represents 10%

    return: a dictionary of the following format:
      {
        0: [3, 65, 2233, ..., 22] // device 0 sample indexes
        1: [0, 2, 4, ..., 583] // device 1 sample indexes
        ...
      }

    iid (independent and identically distributed) means that the indexes
    should be drawn independently in a uniformly random fashion.
    '''

    # total number of samples in the dataset
    total_samples = len(dataset)

    # Part 1.1: Implement!
    samples_per_device = int(total_samples * data_pct)  # Get the #samples per devices
    sampled = dict()  # create dictionary for samples
    for i in range(num_devices):
        sampled[i] = random.choices(range(total_samples), k=samples_per_device)  # sample with replacement
    return sampled

# sampled = iid_sampler(trainset, 1, 0.1)
# print(sampled)

## Code Cell 1.5
import copy
import random


def average_weights(devices):
    '''
    devices: a list of devices generated by create_devices
    Returns an the average of the weights.
    '''
    # Part 1.2: Implement!
    # Hint: device['net'].state_dict() will return an OrderedDict of all
    #       tensors in the model. Return the average of each tensor using
    #       and OrderedDict so that you can update the global model using
    #       device['net'].load_state_dict(w_avg), where w_avg is the
    #       averaged OrderedDict over all devices

    # Get parameters
    params = [device['net'].state_dict() for device in devices]

    w_avg = params[0]
    for device in params[1:]:
        for key in w_avg.keys():
            w_avg[key] += device[key]

    for key in w_avg.keys():
        w_avg[key] = torch.div(w_avg[key], len(devices))
    return w_avg


def get_devices_for_round(devices, device_pct):
    '''
    '''
    # Part 1.2: Implement!
    selected_devices = random.sample(range(len(devices)), k=int(len(devices) * device_pct))
    return [devices[i] for i in selected_devices]


# Test code for average_weights
# Hint: This test may be useful for Part 1.3!
class TestNetwork(nn.Module):
    '''
    A simple 2 layer MLP used for testing your average_weights implementation.
    '''

    def __init__(self):
        super(TestNetwork, self).__init__()
        self.layer1 = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 4)

    def forward(self, x):
        h = F.relu(self.layer1(x))
        return self.layer2(h)


data_pct = 0.05
num_devices = 2
net = TestNetwork()
data_idxs = iid_sampler(trainset, num_devices, data_pct)
devices = [create_device(net, i, trainset, data_idxs[i])
           for i in range(num_devices)]

# Fixed seeding to compare against precomputed correct_weight_averages below
torch.manual_seed(0)
devices[0]['net'].layer1.weight.data.normal_()
devices[0]['net'].layer1.bias.data.normal_()
devices[0]['net'].layer2.weight.data.normal_()
devices[0]['net'].layer2.bias.data.normal_()
devices[1]['net'].layer1.weight.data.normal_()
devices[1]['net'].layer1.bias.data.normal_()
devices[1]['net'].layer2.weight.data.normal_()
devices[1]['net'].layer2.bias.data.normal_()

# Precomputed correct averages
correct_weight_averages = OrderedDict(
    [('layer1.weight', torch.tensor([[0.3245, -0.9013], [-0.9042, 1.0125]])),
     ('layer1.bias', torch.tensor([-0.0724, -0.3119])),
     ('layer2.weight', torch.tensor([[0.2976, 1.0509], [-1.0048, -0.5972],
                                     [-0.3088, -0.2682], [-0.1690, -0.1060]])),
     ('layer2.bias', torch.tensor([-0.4396, 0.3327, -1.3925, 0.3160]))
     ])

# Computed weight averages
computed_weight_averages = average_weights(devices)

mismatch_found = False
for correct, computed in zip(correct_weight_averages.items(),
                             computed_weight_averages.items()):
    if not torch.allclose(correct[1], computed[1], atol=1e-2):
        mismatch_found = True
        print('Mismatch in tensor:', correct[0])

if not mismatch_found:
    print('Implementation output matches!')

## Code Cell 2.1
## Code Cell 2.1
import random


# creates noniid TRAINING datasets for each group
def noniid_group_sampler(dataset, num_items_per_device):
    '''
      dataset: PyTorch Dataset (e.g., CIFAR-10 training set)
      num_devices: integer number of devices to create subsets for
      num_items_per_device: how many samples to assign to each device

      return: a dictionary of the following format:
        {
          0: [3, 65, 2233, ..., 22] // device 0 sample indexes
          1: [0, 2, 4, ..., 583] // device 1 sample indexes
          ...
        }

    '''

    # how many devices per non-iid group
    devices_per_group = [20, 20, 20]

    # label assignment per group
    dict_group_classes = {}
    dict_group_classes[0] = [0, 1, 2, 3]
    dict_group_classes[1] = [4, 5, 6]
    dict_group_classes[2] = [7, 8, 9]

    # Part 2.1: Implement!

    # data filtered by label
    dict_data_by_label = {k: [] for k in range(0, 10)}
    for i in range(0, len(dataset)):
        img, label = dataset[i]
        dict_data_by_label[label].append(i)

    # data filtered by group
    dict_data_by_group = {k: [] for k in range(0, len(dict_group_classes))}
    for i in range(0, len(dict_group_classes)):
        for j in dict_group_classes[i]:
            dict_data_by_group[i].extend(dict_data_by_label[j])

    # DEBUG code
    # for i in range(0, len(dict_data_by_group)):
    #  print(i, "->", len(dict_data_by_group[i]))

    # assign samples to devices
    # whereas some devices in the same group share some samples (see Xin answer in canvas)
    num_devices = 60
    out = {k: [] for k in range(num_devices)}
    offset = 0
    for i in range(0, len(dict_data_by_group)):
        for j in range(0, devices_per_group[i]):
            samples = random.choices(dict_data_by_group[i],
                                     k=num_items_per_device)  # pick n random indices from the group array
            out[offset + j] = samples  # add samples to the output device list
        offset += devices_per_group[i]

    # DEBUG code
    # for i in range(0, len(out)):
    #  print(i, "->", len(out[i]))

    return out

    pass


## Code Cell 2.2

# get which devices in each group should participate in a current round
# by explicitly saying number of each devices desired for each group
def get_devices_for_round_GROUP(devices, device_nums, user_group_idxs):
  # PART 2.2: Implement!
  # Assume first 20 are group 0, second 20 are group 1, third 20 are group 2
  offset = 0
  my_range = 20
  selected_device_idx = []

  for i in range(len(device_nums)):
    selected_device_idx.extend(random.sample(list(range(i * my_range, (i + 1) * my_range)), device_nums[i]))
  return [devices[j] for j in selected_device_idx]


## Code Cell 2.3

# creates noniid TEST datasets for each group
def cifar_noniid_group_test(dataset):
    dict_group_classes = {}
    dict_group_classes[0] = [0, 1, 2, 3]
    dict_group_classes[1] = [4, 5, 6]
    dict_group_classes[2] = [7, 8, 9]

    # Part 2.3: Implement!

    # sort data by label
    dict_data_by_label = {k: [] for k in range(0, 10)}
    for i in range(0, len(dataset)):
        img, label = dataset[i]
        dict_data_by_label[label].append(i)

    # assign labels to group
    out_dict = {k: [] for k in range(0, len(dict_group_classes))}
    for i in range(0, len(out_dict)):
        for j in range(0, len(dict_group_classes[i])):
            label = dict_group_classes[i][j]
            out_dict[i] += dict_data_by_label[label]
        print(len(out_dict[i]))
        # for i in range(0, len(out_dict)):
    #  print("Data group ", i, ": ", len(out_dict[i]))
    return out_dict


def test(epoch, device):
    net.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # print(inputs)
            # print(outputs)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = device['net'](inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            device['test_loss_tracker'].append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total
    sys.stdout.write(f' | Test Loss: {loss:.3f} | Test Acc: {acc:.3f}\n')
    sys.stdout.flush()
    acc = 100. * correct / total
    device['test_acc_tracker'].append(acc)


# gets per-group accuracy of global model
def test_group(epoch, device, group_idxs_dict):
    # Part 2.3: Implement!
    # Hint: refer to test function in PART 1
    # Hint: check https://pytorch.org/docs/stable/data.html?highlight=subset#torch.utils.data.Subset
    for i in range(0, len(group_idxs_dict)):
        net.eval()
        test_loss, correct, total = 0, 0, 0
        indices = group_idxs_dict[i]
        subset = torch.utils.data.Subset(testset, indices)
        subsetLoader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(subsetLoader):
                inputs, targets = inputs.cuda(), targets.cuda()
                # for target_st in range(len(targets)) :
                #   set_labels.add(targets[target_st].item())
                #   #print(inputs[target_st].shape)
                outputs = device['net'](inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                device['test_loss_tracker'].append(loss.item())
                _, predicted = outputs.max(1)
                # print(predicted[:3], targets[:3])
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                loss = test_loss / (batch_idx + 1)
                # print("loss : ", loss)
                acc = 100. * correct / total
            acc = 100. * correct / total
            print("Accuracy : ", acc)
            device['test_acc_tracker_%d' % i].append(acc)


## Code Cell 2.4
def create_device(net, device_id, trainset, data_idxs, lr=0.1,
                  milestones=None, batch_size=128):
    if milestones == None:
        milestones = [25, 50, 75]

    device_net = copy.deepcopy(net)
    optimizer = torch.optim.SGD(device_net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.1)
    device_trainset = DatasetSplit(trainset, data_idxs)
    device_trainloader = torch.utils.data.DataLoader(device_trainset,
                                                     batch_size=batch_size,
                                                     shuffle=True)
    return {
        'net': device_net,
        'id': device_id,
        'dataloader': device_trainloader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'train_loss_tracker': [],
        'train_acc_tracker': [],
        'test_loss_tracker': [],
        'test_acc_tracker': [],
        'test_acc_tracker_0': [],
        'test_acc_tracker_1': [],
        'test_acc_tracker_2': []
    }


rounds = 100
local_epochs = 1
num_items_per_device = 5000
device_nums = [1, 1, 1]
net = ConvNet().cuda()
criterion = nn.CrossEntropyLoss()
milestones = [250, 500, 750]
num_devices = 60
# Part 2.4: Implement non-iid sampling
data_idxs = noniid_group_sampler(trainset, num_items_per_device)
img, label = trainset[data_idxs[24][0]]
print(label)

print(sum([len(data_idxs[i]) == num_items_per_device for i in data_idxs.keys()]))

# Part 2.4: Implement device creation here
devices = [create_device(net, i, trainset, data_idxs[i])
           for i in range(num_devices)]  # Implement this!

test_idxs = cifar_noniid_group_test(testset)
## Non-IID Federated Learning
start_time = time.time()
for round_num in range(rounds):

    # Get devices for each round
    round_devices = get_devices_for_round_GROUP(devices, device_nums, None)

    print('Round: ', round_num)
    for device in round_devices:
        for local_epoch in range(local_epochs):
            train(local_epoch, device)

    # Weight averaging
    w_avg = average_weights(round_devices)

    for device in devices:
        device['net'].load_state_dict(w_avg)
        device['optimizer'].zero_grad()
        device['optimizer'].step()
        device['scheduler'].step()

    # Test accuracy
    test_group(round_num, device, test_idxs)

total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))






import pickle

with open('devices_2_4_group_0.pkl', 'wb') as fp, open('devices_2_4_group_1.pkl', 'wb') as fp1, open('devices_2_4_group_2.pkl', 'wb') as fp2:
    pickle.dump(devices[59]['test_acc_tracker_%d' % 0], fp)
    pickle.dump(devices[59]['test_acc_tracker_%d' % 1], fp1)
    pickle.dump(devices[59]['test_acc_tracker_%d' % 2], fp2)

