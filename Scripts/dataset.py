from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import os


class TorchDataset(Dataset):
    def __init__(self, filename, resize_height=720, resize_width=720, repeat=1, mode=1):
        """
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        """
        self.image_label_list = []
        self.read_file(filename)
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.mode = mode

        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize = transforms.Normalize((0.1307,), (0.3081,))

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_path, label = self.image_label_list[index]
        img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        img = self.data_preproccess(img)
        if self.mode == 1:
            return img, np.array(label)
        elif self.mode == 2:
            dir_file = os.path.splitext(image_path)[0][-25:-2]  # 分类+VId+IId
            return img, np.array(label), dir_file
        else:
            dir_file = os.path.splitext(image_path)[0][-8:]     # IId+Crop_Id
            return img, np.array(label), dir_file


    def __len__(self):
        if self.repeat is None:
            self.repeat = 1
        data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = int(content[1])
                self.image_label_list.append((name, labels))

    def load_data(self, path, resize_height, resize_width, normalization):
        """
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        """
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (resize_width, resize_height))
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data

# DALI 加速 https://github.com/tanglang96/DataLoaders_DALI/blob/master/imagenet.py
