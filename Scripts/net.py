from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

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
            kernel_size = h // (2 ** i)
            if self.pool_type == "max_pool":
                tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        torch.cat(pooling_layers, 1)




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
        self.SSP_layer = SSPLayer(3)            # 1 + 4 + 16 (1*1 + 2*2 + 4*4) = 21 pixels
        self.fc_conv1 = nn.Linear(512*21, 2)

        # L-Net - 1
        self.conv6 = nn.Conv2d(512, 512, 1)     # 78 x 43 -> 78 x 43
        self.pool6 = nn.AdaptiveMaxPool2d(1)    # 78 x 43 -> 1 x 1
        self.fc_conv2 = nn.Conv2d(512*1*1, 2, 1)

        # L-Net - 2
        self.pool7 = nn.AdaptiveAvgPool2d(1)    # 78 x 43 -> 1 x 1
        self.fc_conv3 = nn.Conv2d(512*1*1, 2, 1)


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # G-Net
        g = self.pool5(F.relu(self.conv5(x)))
        g = self.fc_conv1(g)

        # L-Net
        l = F.relu(self.conv6(x))
        l1 = self.fc_conv2(self.pool6(l))
        l2 = self.fc_conv3(self.pool7(l))

        # softmax
        g_r = F.softmax(g)
        l1_r = F.softmax(l1)
        l2_r = F.softmax(l2)

        return g, l1, l2

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
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
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # TODO
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    # TODO

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)    # TODO 衰减速度
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "VideoFake_cnn.pt")


if __name__ == '__main__':
    main()













