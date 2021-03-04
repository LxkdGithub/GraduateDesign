import os
import cv2
import time
from torchvision import transforms, datasets as ds
import torchvision as tv
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def t1():
    tim1 = time.time()
    img0 = cv2.imread("../images/train/prist/00001-000134.png")
    img1 = cv2.imread("../images/train/prist/00001-000135.png")
    img2 = cv2.imread("../images/train/forged/00001-000136.png")
    img3 = cv2.imread("../images/train/forged/00001-000137.png")
    print(time.time() - tim1)
    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    img4 = cv2.subtract(img0_gray, img1_gray)
    img5 = cv2.subtract(img1_gray, img2_gray)
    img6 = cv2.subtract(img2_gray, img3_gray)
    # cv2.imshow("img0", img0_gray)
    # cv2.imshow("img1", img1_gray)
    # cv2.imshow("img2", img2_gray)
    cv2.imshow("img4", img4)
    cv2.imshow("img5", img5)
    cv2.imshow("img6", img6)

    cv2.waitKey(0)


def t2():
    train_set = tv.datasets.ImageFolder(root="../images", transforms=None, target_transform=None)
    pass


def t3():
    print(np.array([1,2,3]))


if __name__ == "__main__":
    t3()