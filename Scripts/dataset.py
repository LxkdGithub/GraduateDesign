import torch
from torch.utils import data
import numpy as np
from torchvision import datasets
# https://github.com/ondyari/FaceForensics/tree/master/dataset
# DeepFake python download-FaceForensics.py <output path> -d <DeepFakeDetection or DeepFakeDetection_original>-c raw -t videos

class MyDataSet(data.Dataset):
    def __init__(self):
        # 读取csv文件中的数据
        xy = np.loadtxt('data-diabetes.csv', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        # 除去最后一列为数据位，存在x_data中
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        # 最后一列为标签为，存在y_data中
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len

def Video2Img(path):
    pass

# 处理视频=>(data, label)，将其变为map结构或者对应起来
def GetDataPair(img_path, fake_index_arr):
    pass

# DALI 加速 https://github.com/tanglang96/DataLoaders_DALI/blob/master/imagenet.py
