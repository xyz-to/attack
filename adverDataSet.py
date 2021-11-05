"""将数据转换为dataset"""

import os
from PIL import Image


import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda")


class adverDataSet(Dataset):
    def __init__(self, root, label, transforms):
        self.root = root  # 数据路径
        self.label = torch.from_numpy(label).long()  # 标签要转换成LONGTENSER
        self.transform = transforms  # 图片预处理
        self.fname = []
        for i in range(200):
            self.fname.append("{:03d}".format(i))

    def __getitem__(self, idex):
        img = Image.open(os.path.join(self.root, self.fname[idex] + '.png'))
        img = self.transform(img)
        label = self.label[idex]
        return img, label

    def __len__(self):
        return 200
