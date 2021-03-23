import os
import numpy as np
import cv2
import argparse


class CreateList:
    def __init__(self, root, isTrain):
        self.DataFile = os.path.abspath(root).replace("\\", "\\\\", 1)
        self.s = []
        self.label = {}
        self.label_num = {}
        self.dataNum = 0
        self.isTrain = isTrain

    def create(self):
        # files = os.listdir(self.DataFile)
        if self.isTrain:
            files = ["prist_crop", "forged_crop", "prist_crop_aug", "forged_crop_aug"]
        else:
            files = ["prist", "forged"]
        for labels in files:
            if os.path.isfile(self.DataFile + "/" + labels):
                continue
            tempData = os.listdir(self.DataFile + "/" + labels)
            self.label[labels] = len(self.label) % 2 # label start from 0
            self.label_num[labels] = len(tempData)
            for img in tempData:
                self.dataNum += 1
                self.s.append([self.DataFile + "/" + labels + "/" + img, self.label[labels]])

    def detail(self):
        print("-------------------------------------------")
        print("  [data_num]  : ", self.dataNum)
        for label in self.label:
            print("  [{}]  : {}".format(label, self.label_num[label]))
        print("-------------------------------------------")

    def shuffle(self):
        temp = self.s
        np.random.shuffle(temp)
        np.random.shuffle(temp)
        shuffle_file = open(self.DataFile + "/Shuffle.txt", "w")
        for i in temp:
            shuffle_file.write(i[0] + " " + str(i[1]) + "\n")

    def get_all(self):
        print(self.s)

    def get_root(self):
        return self.DataFile

    def label_id(self, label):
        # from class_name => label_id
        return self.label[label]

    def test(self):
        print(self.s[0][0])
        img = cv2.imread(self.s[0][0])
        cv2.imshow("img", img)
        cv2.waitKey(0)


def split_train_valid(idx):
    shuffle_train_file = open("../images/train/Shuffle_Train.txt", "w")
    shuffle_valid_file = open("../images/train/Shuffle_Valid.txt", "w")
    shuffle_file = open("../images/train/Shuffle.txt", "r")
    temp = shuffle_file.readlines()
    num = len(temp)
    valid_num = num // 5
    if idx == 4:
        temp_valid = temp[-valid_num:]
        temp_train = temp[:-valid_num]
    else:
        temp_valid = temp[valid_num*idx:valid_num*(idx+1)]
        temp_train = temp[:valid_num*idx] + temp[valid_num*(idx+1):]
        temp = temp[valid_num:]
    for i in temp_valid:
        shuffle_valid_file.write(i)
    for i in temp_train:
        shuffle_train_file.write(i)


def train_process():
    train_data = CreateList(os.path.abspath("../images/train"), True)
    train_data.create()
    train_data.detail()
    train_data.shuffle()


def test_process(train_mode):
    test_data = CreateList(os.path.abspath("../images/test"), train_mode)
    test_data.create()
    test_data.detail()
    test_data.shuffle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Shuffle")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train-mode", action="store_true")
    args = parser.parse_args()
    if args.train:
        print("train")
        train_process()
        split_train_valid(0)
    if args.test:
        print("test")
        print(args.train_mode)
        test_process(args.train_mode)
