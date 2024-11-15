# Author: cuijia1247
# Date: 2024-7-19
# version: 1.0
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

# Dataset for painting91
class SscDataset(Dataset):
    def __init__(self, dataSource, dataTyep, transform, resize_height=128, resize_width=128):
        """
        :param featureName: feature file name , xxx.npy if the file exists
        :param imageFolder: image dir
        :param resize_height: 为None时，不缩放
        :param resize_width: 为None时，不缩放
        """
        dataPath = dataSource + dataTyep + '/'
        for root, dirs, files in os.walk(dataPath):
            # print('root={}'.format(root))
            # print('dirs={}'.format(dirs))
            # print('files={}'.format(files))
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
        # self.transforms = transforms
        # self.multiFeatures = self.transforms(self.features)

    def getFeature(self, dataPath, classNum):
        featureList = []
        originalfeautreLsit = []
        labelList = []
        nameList = []
        for subFolder in range(classNum):
            label = subFolder + 1
            newPath = dataPath + str(label) + '/'
            # for filename in tqdm.tqdm(os.listdir(newPath)):
            for filename in os.listdir(newPath):
                imgPath = newPath + filename
                # print('label is {}, filename is {}'.format(label, filename))
                # img = Image.open(imgPath).convert('RGB')
                img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
                # img = cv2.imread(imgPath)
                # if filename == 'impressionism_james-tissot-sarah-hears-and-laughs.jpg':
                #     print('ERROR')
                if img is not None:
                    # print(imgPath)
                    img = cv2.resize(img, (256, 256))
                    imgO = img
                    imgO = self.transforms_original(imgO)
                    originalfeautreLsit.append(imgO)
                    img1, img2 = self.transforms(img)
                    # img1 = img1.numpy().transpose(1,2,0)
                    # img2 = img2.numpy().transpose(1,2,0)
                    tempList = []
                    tempList.append(img1)
                    tempList.append(img2)
                    featureList.append(tempList)
                    labelList.append(label)
                    nameList.append(filename)
                else:
                    print('The {} is removed'.format(imgPath))
                    os.remove(imgPath)
        return featureList, labelList, nameList, originalfeautreLsit

    def __getitem__(self, i):
        index = i % self.len
        # image_name, label = self.image_label_list[index]
        img1, img2 = self.features[index]
        label = self.labels[index]
        name = self.names[index]
        original = self.originalF[index]
        return img1, img2, label, name, original

    def __len__(self):
        data_len = -999
        data_len = len(self.features)
        if data_len == -999:
            raise ValueError('Dataset Length ERROR.')
        return data_len

    def load_data(self, path, resize_height, resize_width, normalization=False):
        """
        :param path: data path
        :param resize_height:
        :param resize_width:
        :param normalization: whether the normalization is required.
        :return:
        """
        img = ip.read_image(path, resize_height, resize_width, normalization=False)
        return img

    def data_preprocess(self, data):
        """
        :param data: normally the data are images
        :return:
        """
        # img = cv2.imread(data)
        # if img is None:
        #     print(os.path.join(self.imageFolder, data))
        #     raise ValueError('Image Path ERROR.')
        img = Image.open(data).convert('RGB')
        data = self.toTransform(img)
        return data

if __name__ == '__main__':
    #__init__(self, featureName, resize_height=128, resize_width=128):
    # featureName = './pretrainFeatures/Painting91_Resnet50_train.pkl'
    dataSource = './data/Painting91/'
    dataTyep = 'train'
    transformT, transformT1, transformEvalT = get_byol_transforms(64, (0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))
    temp = SscDataset(dataSource, dataTyep, transform=MultiViewDataInjector([transformT, transformT1]))
    print('Dataset is done.')