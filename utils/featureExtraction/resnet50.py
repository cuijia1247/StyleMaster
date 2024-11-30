# Author: cuijia1247
# Date: 2024-7-25
# version: 1.0
# the features are extracted as a dict form (name: feature) by pretrained resnet50 model
import numpy as np
import os
os.environ['TORCH_HOME'] = '../../pretrainModels/resnet50'  #指定预训练模型下载地址
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import tqdm
import pickle
import os

# 检查是否有可用的 GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_models(target_layer=4):
    resnet = models.resnet50(pretrained=True).to(device)

    # 获取模型的特定层的输出
    layers = [
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3),
        nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
                      resnet.layer3, resnet.layer4)
    ]

    if target_layer < 1 or target_layer > 4:
        raise ValueError(f"Invalid target layer: {target_layer}. Target layer should be between 1 and 4.")

    # 截取模型到指定层
    feature_extractor = nn.Sequential(*layers[target_layer - 1]).eval()
    return feature_extractor

def featureE(model, imgPath, labelPath, featurePath, featureName):
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # features = []
    # names = []
    final_features = {}

    with open(os.path.join(getRootPath(),labelPath), 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            content = line.rstrip().split(',')
            name = content[0][1:-1]
            # print(name)
            imgP = os.path.join(imgPath, name)
            img = Image.open(os.path.join(getRootPath(), imgP)).convert('RGB')
            tensor = train_transform(img).to(device)
            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(tensor).to(device).data.cpu().numpy()
            feature = feature.flatten()
            final_features[name] = feature
            # features.append(feature)
            # names.append(name)
    outputP = os.path.join(featurePath, featureName)
    np.save(os.path.join(getRootPath(), outputP), final_features)
    print('The ResNet50 Feature Extraction DONE.')

def getRootPath():
    return os.path.abspath(os.path.join(os.getcwd(), "../../"))

def styleDataset(model, imgFolder, targetPath, featureName):
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    #get label numbers
    for root, dirs, files in os.walk(imgFolder):
        # print('root={}'.format(root))
        # print('dirs={}'.format(dirs))
        # print('files={}'.format(files))
        classNum = dirs.__len__()
        break

    styleList_ = []
    for subFolder in range(classNum):
        label = subFolder + 1
        newPath = imgFolder + '/' + str(label) + '/'
        for filename in tqdm.tqdm(os.listdir(newPath)):
            imgPath = newPath + filename
            img = Image.open(imgPath).convert('RGB')
            tensor = train_transform(img).to(device)
            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(tensor).to(device).data.cpu().numpy()
            feature = feature.flatten()
            feature_ = []
            feature_.append({"feature":feature})
            feature_.append({"label":label})
            styleList_.append(feature_)
        # styleList.append(styleList_)
    outputPath = targetPath + featureName
    with open(outputPath, 'wb') as file:
        pickle.dump(styleList_, file)
    print('The ResNet50 Style DataSet Feature Extraction DONE.')


if __name__ == '__main__':
    model = make_models()
    imgPath = '../../data/Painting91/train'
    targetPath = '../../pretrainFeatures/'
    featureName = 'Painting91_Resnet50_train.pkl'
    styleDataset(model, imgPath, targetPath, featureName)