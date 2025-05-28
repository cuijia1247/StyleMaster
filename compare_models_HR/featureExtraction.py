import os
os.environ['TORCH_HOME'] = '../PretrainedModels/densenet121'  #指定预训练模型下载地址
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def models_v3():
    model = models.inception_v3(pretrained=True).to(device)
    # model = nn.Sequential(model.Conv2d_1a_3x3,model.Conv2d_2a_3x3,model.Conv2d_2b_3x3,model.maxpool1,model.Conv2d_3b_1x1,model.Conv2d_4a_3x3,model.maxpool2,model.Mixed_5b,model.Mixed_5c,model.Mixed_5d,model.Mixed_6a,model.Mixed_6b,model.Mixed_6c,model.Mixed_6d,model.Mixed_6e,model.AuxLogits,model.Mixed_7a,model.Mixed_7b,model.Mixed_7c,model.avgpool,model.dropout)
    # model = model.features.eval()
    model.dropout = nn.Sequential()
    model.fc = nn.Sequential()
    # model.avgpool = nn.Sequential()
    print(model)
    return model
# def make_densenet121():
#     model = models.densenet121(pretrained=True).to(device)
#     # model = model.features.eval()
#     model.classifier = nn.Sequential()
#     print(model)
#     return model

def make_densenet121():
    model = models.densenet121(pretrained=True).to(device)
    # model = model.features.eval()
    # model.classifier = nn.Sequential()
    # model.features = nn.Sequential(*list(model.features.children())[:-1])
    model.features.eval()
    print(model)
    return model

def make_resnet50():
    resnet = models.resnet50(pretrained=True).to(device)
    print(resnet)
    # b = resnet.state_dict()
    # # resnet = models.__dict__['resnet50']
    # checkpoint = torch.load('../modelv2_800_best.pth.tar', map_location="cpu")
    # state_dict = checkpoint["state_dict"]
    # resnet.load_state_dict(state_dict, strict=False)

    resnet.fc = nn.Sequential()
    # resnet.avgpool = nn.Sequential()


    # # 获取模型的特定层的输出
    # layers = [
    #     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
    #     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2),
    #     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
    #                   resnet.layer3),
    #     nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2,
    #                   resnet.layer3, resnet.layer4)
    # ]
    
    # if target_layer < 1 or target_layer > 4:
    #     raise ValueError(f"Invalid target layer: {target_layer}. Target layer should be between 1 and 4.")
    
    # # 截取模型到指定层
    # feature_extractor = nn.Sequential(*layers[target_layer - 1]).eval()
    return resnet

def make_vgg16():
    # pretrainedPath = './PretrainedModels/vgg16/hub/checkpoints/vgg16-397923af.pth'
    # if os.path.exists(pretrainedPath)==False: #no pretrained model
    model = models.vgg16(pretrained=True).to(device)
    print(model)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-2])
    # model = models.vgg16(pretrained=True).features[:layers].to(device)
    print(device)
    print(model)
    # if torch.cuda.is_available():
    #     model.cuda()
    # else:
    #     model = models.vgg16(weight)
    #     model = torch.load('./PretrainedModels/vgg16/hub/checkpoints/vgg16-397923af.pth').features[:layers]
    model = model.eval()
    return model

def deep_feature_extraction(model, img_folder_path, label_path, feature_name):#get all the images under the folder
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    # img_to_tensor = transforms.ToTensor()
    features = []
    names = []
    pca_features = {}
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
                
        for line in lines:
            # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content = line.rstrip().split(',')
        
            name = content[0][1:-1]
            imgPath = img_folder_path + name
            imgPath = os.path.join(img_folder_path, name)  # 组合完整路径


            img = Image.open(imgPath).convert('RGB')
            tensor = train_transform(img).to(device)

            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(Variable(tensor.float(), requires_grad=False)).data.cpu().numpy()
            feature = feature.flatten()

            features.append(feature)
            names.append(name)
            print(name)

            
    pca = PCA(n_components=800)
    temp = pca.fit_transform(features)
    # temp = features
    joblib.dump(pca,'./Models/new-wikiart3-densenet121.pkl')

    for i in range(names.__len__()):
        pca_features[names[i]] = temp[i]
    np.save('/home/huangrui/Codes/SubStyleClassfication/compare_models/Features/' + feature_name, pca_features)
    print('deep_feature_extraction DONE.')

def deep_feature_extraction1(model, img_folder_path, label_path, feature_name):#get all the images under the folder
    model.eval()
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])
    # img_to_tensor = transforms.ToTensor()
    features = []
    names = []
    pca_features = {}
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
            content = line.rstrip().split(',')
            name = content[0][1:-1]
            imgPath = img_folder_path + name
            img = Image.open(imgPath).convert('RGB')
            tensor = train_transform(img).to(device)

            tensor = torch.unsqueeze(tensor, dim=0)
            feature = model(Variable(tensor.float(), requires_grad=False)).data.cpu().numpy()
            feature = feature.flatten()

            features.append(feature)
            names.append(name)
            print(name)

    pca = joblib.load('./Models/new-wikiart3-densenet121.pkl')
    temp = pca.transform(features)



    for i in range(names.__len__()):
        pca_features[names[i]] = temp[i]

    np.save('/home/huangrui/Codes/SubStyleClassfication/compare_models/Features/' + feature_name, pca_features)

    print('deep_feature_extraction DONE.')


if __name__ == '__main__':
    # model = make_vgg16() #5 conv1; 10 conv2; 17 conv3; 24 conv4;31 conv5 
    # model = make_resnet50()
    model = make_densenet121()
    # model = modelsv3()
    name1 = 'train'
    name2 = 'test'

    img_path = '/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Images/'
    train_path = '/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Labels/train.txt'
    test_path = '/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Labels/test.txt'

    # feature_name = 'painting91-vgg16-' + name_ + '.npy' #大家尽量按这个形式命名：数据库-模型-结构.npy
    train_name = 'new-wikiart3-densenet121-' + name1 #大家尽量按这个形式命名：数据库-模型-结构.npy
    test_name = 'new-wikiart3-densenet121-' + name2  # 大家尽量按这个形式命名：数据库-模型-结构.npy
  
    deep_feature_extraction(model, img_path, train_path, train_name)
    deep_feature_extraction1(model, img_path, test_path, test_name)
