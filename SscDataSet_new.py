# Author: cuijia1247
# Date: 2024-7-19
# version: 2.0
# 相比 v1.0 改动：getFeature 只存原始图片（numpy），__getitem__ 中实时做随机 transform，
# 保证每次迭代 DataLoader 时 view1/view2 都是不同的随机裁剪，防止分类器过拟合固定增强结果。
import os.path

from torch.utils.data import Dataset
import numpy as np
from utils import image_processing as ip
from PIL import Image
import torch
import torchvision.transforms as transforms_o
import random
import tqdm
from scipy import spatial
from ssc.utils import criterion, get_byol_transforms, MultiViewDataInjector
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset for painting91（懒加载版：__getitem__ 中实时做随机增强）
class SscDataset(Dataset):
    def __init__(self, dataSource, dataTyep, transform, resize_height=128, resize_width=128):
        """
        :param dataSource: 数据集根目录
        :param dataTyep: 'train' 或 'test'
        :param transform: MultiViewDataInjector，用于生成两路随机增强视图
        :param resize_height: 预留参数，暂未使用
        :param resize_width:  预留参数，暂未使用
        """
        classNum = -999
        dataPath = dataSource + dataTyep + '/'
        for root, dirs, files in os.walk(dataPath):
            classNum = dirs.__len__()
            break
        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        self.transforms_original = transforms_o.Compose([
            transforms_o.ToTensor(),
            transforms_o.Resize((224, 224)),
            transforms_o.Normalize(norm_mean, norm_std),
        ])
        self.transforms = transform

        self.features, self.labels, self.names, self.originalF = self.getFeature(dataPath, classNum)
        self.len = self.__len__()
        self.resize_height = resize_height
        self.resize_width = resize_width

    def getFeature(self, dataPath, classNum):
        """加载图片时只存原始 numpy 数组，不做随机 transform，延迟到 __getitem__ 中执行。"""
        featureList = []
        originalfeautreLsit = []
        labelList = []
        nameList = []
        for subFolder in range(classNum):
            label = subFolder + 1
            newPath = dataPath + str(label) + '/'
            for filename in os.listdir(newPath):
                imgPath = newPath + filename
                img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (256, 256))
                    imgO = self.transforms_original(img)
                    originalfeautreLsit.append(imgO)
                    featureList.append(img)      # 存原始 numpy，不做 transform
                    labelList.append(label)
                    nameList.append(filename)
                else:
                    print('The {} is removed'.format(imgPath))
                    os.remove(imgPath)
        return featureList, labelList, nameList, originalfeautreLsit

    def __getitem__(self, i):
        index = i % self.len
        img = self.features[index]               # 原始 numpy (256, 256, 3)
        img1, img2 = self.transforms(img)        # 每次调用都重新做随机裁剪/旋转等增强
        label = self.labels[index]
        name = self.names[index]
        original = self.originalF[index]
        return img1, img2, label, name, original

    def __len__(self):
        data_len = len(self.features)
        if data_len == 0:
            raise ValueError('Dataset Length ERROR.')
        return data_len

    def load_data(self, path, resize_height, resize_width, normalization=False):
        img = ip.read_image(path, resize_height, resize_width, normalization=False)
        return img

    def data_preprocess(self, data):
        img = Image.open(data).convert('RGB')
        data = self.toTransform(img)
        return data

if __name__ == '__main__':
    dataSource = './data/Painting91/'
    dataTyep = 'train'
    transformT, transformT1, transformEvalT = get_byol_transforms(64, (0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))
    temp = SscDataset(dataSource, dataTyep, transform=MultiViewDataInjector([transformT, transformT1]))
    print('Dataset is done.')
