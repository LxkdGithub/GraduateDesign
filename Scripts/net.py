from __future__ import print_function
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataloader

import torchvision
from torchvision import transforms

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

import dataset

# 链接 https://blog.csdn.net/liyaohhh/article/details/50614380
# SPPLayer其实就是一个灵活的多层次的 Pool
# 注意和 nn.AdaptivePool的区别
class SSPLayer(nn.Module):
    def __init__(self, num_layers, pool_type='max_pool'):
        super(SSPLayer, self).__init__()

        self.num_layers = num_layers
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_layers):
            h_kernel_size = h // (2 ** i)
            w_kernel_size = w // (2 ** i)
            if self.pool_type == "max_pool":
                tensor = F.max_pool2d(x, kernel_size=(h_kernel_size, w_kernel_size), stride=(h_kernel_size, w_kernel_size)).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=(h_kernel_size, w_kernel_size), stride=(h_kernel_size, w_kernel_size)).view(bs, -1)
            pooling_layers.append(tensor)
        print("SSP Layer: ",torch.cat(pooling_layers, 1).size())
        return torch.cat(pooling_layers, 1)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64,3)          # 1280 x 720 -> 1278 x 718
        self.conv2 = nn.Conv2d(64, 128, 3)      #  639 x 359 -> 637 x 357
        self.conv3 = nn.Conv2d(128, 256, 3)     # 318 x 178 -> 316 x 176
        self.conv4 = nn.Conv2d(256, 512, 3)     # 158 x 88 -> 156 x 86

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)


        # G-Net
        self.conv5 = nn.Conv2d(512, 512, 3)     # 78 x 43 -> 76 x 41
        self.pool5 = nn.MaxPool2d(2, 2)         # 76 x 41 -> 38 x 20
        # SSP Layer(大小不确定)
        self.SSP_layer = SSPLayer(2)            # 1 + 4 + 16 (1*1 + 2*2 + 4*4) = 21 pixels
        self.fc_conv1 = nn.Linear(512*5, 2)

        # L-Net - 1
        self.conv6 = nn.Conv2d(512, 512, 1)     # 78 x 43 -> 78 x 43
        self.pool6 = nn.AdaptiveMaxPool2d(1)    # 78 x 43 -> 1 x 1
        self.fc_conv2 = nn.Linear(512*1*1, 2)

        # L-Net - 2
        self.pool7 = nn.AdaptiveAvgPool2d(1)    # 78 x 43 -> 1 x 1
        self.fc_conv3 = nn.Linear(512*1*1, 2)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x), inplace=True))
        x = self.pool2(F.relu(self.conv2(x), inplace=True))
        x = self.pool3(F.relu(self.conv3(x), inplace=True))
        x = self.pool4(F.relu(self.conv4(x), inplace=True))

        # G-Net
        g = self.pool5(F.relu(self.conv5(x), inplace=True))
        g = self.SSP_layer(g)
        g = self.fc_conv1(g)

        # L-Net
        l = F.relu(self.conv6(x), inplace=True)
        l1 = self.pool6(l)
        l1 = l1.view(l1.shape[0], -1)
        l1 = self.fc_conv2(l1)
        l2 = self.pool7(l)
        l2 = l2.view(l2.shape[0], -1)
        l2 = self.fc_conv3(l2)

        # softmax
        g_r = nn.LogSoftmax(dim=1)(g)
        l1_r = nn.LogSoftmax(dim=1)(l1)
        l2_r = nn.LogSoftmax(dim=1)(l2)
        del x, g, l, l1, l2

        return g_r, l1_r, l2_r

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        output = model(data)
        # print(output)
        # TODO  怎么学习(只要不是直接setp()了，就无所谓)
        loss1 = F.nll_loss(output[0], target)
        loss2 = F.nll_loss(output[1], target)
        loss3 = F.nll_loss(output[2], target)
        loss = loss1 + loss2 + 0.1 * loss3
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        del loss1, loss2, loss3, loss


def valid(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            output = model(data)
            loss1 = F.nll_loss(output[0], target)
            loss2 = F.nll_loss(output[1], target)
            loss3 = F.nll_loss(output[2], target)
            loss = loss1 + loss2 + 0.1 * loss3
            test_loss += loss  # sum up batch loss
            output_merge = output[0] * 0.5 + output[1] * 0.4 + output[2] * 0.1
            print(output[0][0])
            print(output[1][0])
            print(output[2][0])
            print(output_merge[0])
            pred = output_merge.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            del loss1, loss2, loss3, loss

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)        # 设置种子数，参数初始化一致
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 3,
                       'pin_memory': True,
                       'shuffle': True}             # 这个shuffle是batch_size内部shuffle 所以Dataset还是要自己写Shuffle
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Dataset那里使用
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])


    # -------------------      Dataset      -----------------------
    # TODO 无法打乱
    train_dataset = dataset.TorchDataset("../images/train/Shuffle_Data.txt")
    test_dataset = dataset.TorchDataset("../images/test/Shuffle_Data.txt")
    # test_dataset = torchvision.datasets.ImageFolder(root="../images/test", transform=transform)
    # TODO
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    torch.cuda.empty_cache()

    # --------------------    Model + optimizer  --------------------
    torch.backends.cudnn.enabled = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    model = Net()
    model = nn.DataParallel(model)
    model = model.to(device)
    if os.path.exists("VideoFake_cnn.pt"):
        model.load_state_dict(torch.load("VideoFake_cnn.pt"))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5000, gamma=args.gamma)    # TODO 衰减速度

    # ----------------------  train and test  ------------------------
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        torch.cuda.empty_cache()
        valid(model, device, test_loader)
        scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "VideoFake_cnn.pt")


if __name__ == '__main__':
    main()









