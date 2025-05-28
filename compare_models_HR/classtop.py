from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import numpy as np
from utils import image_processing
import os

os.environ['TORCH_HOME'] = './PretrainedModels/densenet121'  # 指定预训练模型下载地址
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from statistics import mean
import heapq
import copy
from tqdm import tqdm
#from prettytable import PrettyTable


def listtop(list, K):
    max_val_lis = heapq.nlargest(K, list)
    max_idx_lis = []
    for item in max_val_lis:
        idx = list.index(item)
        max_idx_lis.append(idx)
        list[idx] = float('-inf')
    return max_idx_lis


class TorchDataset(Dataset):
    def __init__(self, filename, feature_name, image_dir, resize_height=256, resize_width=256, repeat=1):
        '''
        :param filename: 数据文件TXT：格式：imge_name.jpg label1_id labe2_id
        :param image_dir: 图片路径：image_dir+imge_name.jpg构成图片的完整路径
        :param resize_height 为None时，不进行缩放
        :param resize_width  为None时，不进行缩放，
                              PS：当参数resize_height或resize_width其中一个为None时，可实现等比例缩放
        :param repeat: 所有样本数据重复次数，默认循环一次，当repeat为None时，表示无限循环<sys.maxsize
        '''
        self.image_label_list = self.read_file(filename)
        self.features = np.load(feature_name, allow_pickle=True).item()
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat
        self.resize_height = resize_height
        self.resize_width = resize_width

        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。
        self.toTensor = transforms.ToTensor()

        '''class torchvision.transforms.Normalize(mean, std)
        此转换类作用于torch. * Tensor,给定均值(R, G, B) 和标准差(R, G, B)，
        用公式channel = (channel - mean) / std进行规范化。
        '''
        # self.normalize=transforms.Normalize()

    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        # img = self.load_data(image_path, self.resize_height, self.resize_width, normalization=False)
        # img = self.data_preproccess(img)
        img = image_name
        label = np.array(label)
        keys = self.features.keys()
        feature = []
        for key in keys:
            if key == image_name:
                feature = self.features[key]
                break;
        if feature.__len__() == 0:
            raise ValueError('特征值检索出错！')

        return img, label, feature

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len

    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(',')
                # pdb.set_trace()
                name = content[0][1:-1]
                labels = []
                count = 1
                for value in content[1:]:  # 将one-hot向量变成整型，例如0,1,0,0,0,0,0,0,0,0,0,0,0,0 --> 2
                    if int(value) == 1:
                        labels.append(count)
                        break
                    count = count + 1
                    # labels.append(int(value))
                image_label_list.append((name, labels))
        # pdb.set_trace()
        return image_label_list

    def load_data(self, path, resize_height, resize_width, normalization):
        '''
        加载数据
        :param path:
        :param resize_height:
        :param resize_width:
        :param normalization: 是否归一化
        :return:
        '''
        image = image_processing.read_image(path, resize_height, resize_width, normalization)
        return image

    def data_preproccess(self, data):
        '''
        数据预处理
        :param data:
        :return:
        '''
        data = self.toTensor(data)
        return data


def train_svc_model(train_filename, image_dir, layer_name, batch_size,data_name,feature_path,model_path):
    #feature_path='./Features/painting91-resnet50-' 
    #model_path='./Models/painting91-resnet50-'

    epoch_num = 1  # 总样本循环次数
    # batch_size = 10587 # 训练时的一组数据的大小
    # train_data_nums = 10587
    max_iterate = int((batch_size + batch_size - 1) / batch_size * epoch_num)  # 总迭代次数

    feature_name_ = feature_path + layer_name + '.npy'

    step_ = 1  # 重复训练次数
    train_data = TorchDataset(filename=train_filename, feature_name=feature_name_, image_dir=image_dir, repeat=step_)
    # test_data = TorchDataset(filename=test_filename, image_dir=image_dir,repeat=1)
    # pdb.set_trace()
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    accuracy_list = []

    model_name = model_path + layer_name + '.pkl'


    print("开始训练 SVM 模型...")
    
    for step, (batch_image, batch_label, batch_feature) in enumerate(train_loader):
        # classifier = SVC(kernel='poly', class_weight='balanced',probability=True)
        classifier = SVC(kernel='poly', class_weight='balanced', probability=True)
        classifier.fit(batch_feature, batch_label)
        y_train = classifier.predict(batch_feature)
        accuracy_train = accuracy_score(batch_label, y_train)

        #print(f"Step [{step+1}/{len(train_loader)}]: 训练准确率: {accuracy_train:.4f}")
        print("step:{},batch_image.top1accuracy_train:{}".format(step, accuracy_train))
        if accuracy_list.__len__() == 0:  # 循环迭代训练时，第一次存训练模型pkl，后面的只有在准确率高的时候才保存
            joblib.dump(classifier, model_name)
            print(f"新模型已保存: {model_name}")
        else:
            threshold = max(accuracy_list)
            if accuracy_train > threshold:
                joblib.dump(classifier, model_name)
        accuracy_list.append(accuracy_train)
        if step >= step_:
            break
    average_accuracy = mean(accuracy_list)
    print ('The average accuracy of top1' + layer_name + '-SVC in ' + data_name + ' is : {}'.format(average_accuracy))


def test1(train_filename, image_dir, layer_name, batch_size, model_name,data_name,feature_path):
    feature_name_ = feature_path + 'test' + '.npy'
    step_ = 1
    train_data_nums = 10587
    train_data = TorchDataset(filename=train_filename, feature_name=feature_name_, image_dir=image_dir, repeat=step_)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    error_list = []  # 用于存储错误分类的信息
    correct_list=[]

    print("开始测试 SVM 模型...")

    for step, (batch_image, batch_label, batch_feature) in enumerate(train_loader):
        classifier = joblib.load(model_name)
        y_train = classifier.predict(batch_feature)
    
        accuracy_train = accuracy_score(batch_label, y_train)
        print('The average accuracy of top1' + layer_name + '-SVC in ' + data_name + ' is : {}'.format(accuracy_train))

        # 记录错误分类的图片名称、真实类别和预测类别
        for i in range(len(batch_image)):
            predict_label=batch_label.squeeze().numpy()
            if y_train[i] != batch_label[i]:
                error_list.append((batch_image[i], predict_label[i], y_train[i]))
            else:
                correct_list.append((batch_image[i], y_train[i]))
                

    # 将错误分类的信息写入txt文件
    with open('new15-densenet121-wikiart3-classification.txt', 'w') as f:
        for error in error_list:
            f.write(f"Image: {error[0]}, True Label: {error[1]}, Predicted Label: {error[2]}\n")
        for correct in correct_list:
            f.write(f"Image: {correct[0]}, True Label: {correct[1]}\n")
        


def test(train_filename, image_dir, layer_name, num, model_name):
    feature_name_ = './Features/painting91-vgg16-' + layer_name + '.npy'
    step_ = 1  # 重复训练次数
    train_data = TorchDataset(filename=train_filename, feature_name=feature_name_, image_dir=image_dir, repeat=step_)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    classifier = joblib.load(model_name)
    correct = 0
    for batch_image, batch_label, batch_feature in train_loader:
        temp = classifier.predict(batch_feature)
        temp = int(temp)
        batch_label = int(batch_label)
        if temp == batch_label:
            correct = correct + 1
    pred1 = correct / num
    print('The average accuracy of top1' + layer_name + '-SVC in painting91 dataset is : {}'.format(pred1))
    correct = 0
    for batch_image, batch_label, batch_feature in train_loader:
        temp = classifier.predict_proba(batch_feature)
        temp = np.squeeze(temp)
        temp = list(temp)
        batch_label = int(batch_label) - 1
        temp2 = listtop(temp, 2)
        if batch_label in temp2:
            correct = correct + 1
    pred2 = correct / num
    print('The average accuracy of top2' + layer_name + '-SVC in painting91 dataset is : {}'.format(pred2))
    correct = 0
    for batch_image, batch_label, batch_feature in train_loader:
        temp = classifier.predict_proba(batch_feature)
        temp = np.squeeze(temp)
        temp = list(temp)
        batch_label = int(batch_label) - 1
        temp2 = listtop(temp, 3)
        if batch_label in temp2:
            correct = correct + 1
    pred3 = correct / num
    print('The average accuracy of top3' + layer_name + '-SVC in painting91 dataset is : {}'.format(pred3))

if __name__ == '__main__':
    train_filename = "/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Labels/train.txt"
    test_filename = "/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Labels/test.txt"
    image_dir = '/home/huangrui/Codes/SubStyleClassfication/data/WikiArt3/Images/'
    train_name = 'train'
    test_name = 'test'  # 可调参数

    data_name='wikiart3'
    feature_path='./Features/new-wikiart3-densenet121-' 
    model_path='./Models/new-wikiart3-densenet121-'
    train_svc_model(train_filename, image_dir, train_name, 10571,data_name,feature_path,model_path)
    model_name = model_path + train_name + '.pkl'
    # test1(test_filename, image_dir, test_name, 2000, model_name,data_name,feature_path)
    test1(test_filename, image_dir, test_name, 15000, model_name,data_name,feature_path)
    # test(test_filename, image_dir, test_name, 2663,model_name)
