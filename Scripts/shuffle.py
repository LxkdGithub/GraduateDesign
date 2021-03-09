import os
import numpy as np
import cv2


class CreateList:
    def __init__(self, root):
        self.DataFile = os.path.abspath(root).replace("\\", "\\\\", 1)
        self.s = []
        self.label = {}
        self.label_num = {}
        self.dataNum = 0

    def create(self):
        files = os.listdir(self.DataFile)
        for labels in files:
            if os.path.isfile(self.DataFile + "/" + labels):
                continue
            tempData = os.listdir(self.DataFile + "/" + labels)
            self.label[labels] = len(self.label)  # label start from 0
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
        shuffle_file = open(self.DataFile + "/Shuffle_Data.txt", "w")
        temp = self.s
        np.random.shuffle(temp)
        for i in temp:
            shuffle_file.write(i[0] + " " + str(i[1]) + "\n")
        return self.DataFile + "/Shuffle_Data.txt"

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


def all_process():
    train_data = CreateList(os.path.abspath("../images/train"))
    valid_data = CreateList(os.path.abspath("../images/valid"))
    test_data = CreateList(os.path.abspath("../images/test"))
    train_data.create()
    train_data.detail()
    train_data.shuffle()
    valid_data.create()
    valid_data.detail()
    valid_data.shuffle()
    test_data.create()
    test_data.detail()
    test_data.shuffle()


if __name__ == "__main__":
    all_process()
