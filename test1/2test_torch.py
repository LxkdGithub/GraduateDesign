import torch
import torch.nn as nn
from torch.utils import data
import os

if __name__ == "__main__":
    # a = torch.rand([2,3,4])
    # print(a.size())
    # print(type(a))
    # nn.AdaptiveMaxPool2d((3,3))
    # print(a)
    # nn.AdaptiveMaxPool2d((4, 4))
    # print(a)
    # nn.AdaptiveMaxPool2d((5, 5))
    # print(a)

    # m = nn.Conv2d(1, 1, 3)
    # m = nn.MaxPool2d(2, 2)
    # a = torch.randn([1, 1, 5, 4])
    # print(a)
    # b = m(a)
    # print(b)

    # print(os.getcwd())
    # path = os.getcwd() + "\\GraduteDesign\\test1\\images"
    # op = os.listdir(path)
    # print(len(op))

    pool = []
    a = torch.rand(2, 2)
    b = torch.rand(2, 2)
    pool.append(a)
    pool.append(b)
    print(torch.cat(pool, 1))








