import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


def isTrue(a):
    target = a[0]
    res = a[1]
    if (target + res == 0) or (target > 0 and res > 0):  # target 只有0/3 res 有0/1/2/3
        return 1
    return 0

def t1():
    img = cv2.imread("../images/train/forged/00001-000136.png")
    cv2.imwrite("136.bmp", img)



def t2():
    result = {"001": [0, 0], "002": [0, 1], "003": [1, 3], "004": [1, 1], "005": [0, 0] ,"006": [1, 0]}
    result.update({"008": [0,0]})
    print(result.values())
    # new_result = map(isTrue, result.values())
    # print(sum(list(new_result)))

def dropout(X, drop_prob):
    X = X.float()


if __name__ == "__main__":
    t2()
    d = torch.nn.Dropout()
    x = torch.rand((2, 4))
    d()


