'''Train with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import os.path as osp
import argparse

from models import *
from utils import progress_bar

from logger import CustomLogger
from apex import amp

import datetime
import yaml


parser = argparse.ArgumentParser(description='PyTorch Training Baseline')
# parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batch-train', type=int, default=128, help='batch size for train set')
parser.add_argument('--batch-test', type=int, default=100, help='batch size for test set')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--amp',  action='store_true', help='train with amp')
parser.add_argument('--scheduler',  action='store_true', help='train with scheduler')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Data loading code
# traindir = os.path.join(args.data, 'train')
# valdir = os.path.join(args.data, 'val')


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_train, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_test, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

if args.amp:
    print('==> Operate amp')
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

if args.scheduler:
    print('==> Operate scheduler')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=1, min_lr=1e-10, verbose=True)


# logger
here = os.getcwd()
now = datetime.datetime.now()
args.out = now.strftime('%Y%m%d_%H%M%S.%f')
log_dir = osp.join(here, 'logs', args.out)
os.makedirs(log_dir)
logger = CustomLogger(out=log_dir)

# make dirs for the checkpoint
check_dir = osp.join(here, 'checkpoint', args.out)
os.makedirs(check_dir)

# for .yaml
args.dataset = ['CIFAR10']
args.optimizer = 'SGD'
args.model = 'ResNet18'

with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
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
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint. (every case)
    print('Saving..')
    torch.save(net.state_dict(), osp.join(check_dir, 'bengali_{}.pth'.format(epoch)))

    logger.write(True, epoch, batch_idx, train_loss/(batch_idx+1), 100.*correct/total)


def test(epoch):
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
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint. (only for the best case)
    # acc = 100.*correct/total
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc

    logger.write(False, epoch, batch_idx, test_loss/(batch_idx+1), 100.*correct/total)


# manually adjust learning rate
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10"""
    #lr = args.lr * (0.1 ** (epoch // 30))
    if epoch == 0:
        print('LR is set to {}'.format(args.lr))

    if epoch == 30 or epoch == 60 or epoch == 80:
        for param_group in optimizer.param_groups:
            lr = param_group['lr'] * 0.1
            param_group['lr'] = lr 
        print('LR is set to {}'.format(lr))


for epoch in range(start_epoch, start_epoch+100):
    # adjust_learning_rate(optimizer, epoch, args)
    train(epoch)
    test(epoch)
    if args.scheduler:
        scheduler.step(float(test_loss))