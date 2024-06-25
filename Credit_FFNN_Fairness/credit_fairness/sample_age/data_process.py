#读取均匀采样后的数据，然后放到训练好的网络得到输出，计算公平性

# 可视化
import matplotlib.pyplot as plt  # 绘图
import matplotlib
import time
import datetime
import pandas as pd
import numpy as np
# from lib_models import *
# from utils import *

from pandas.plotting import scatter_matrix #绘制散布矩阵图
from sklearn.model_selection import StratifiedShuffleSplit #分层交叉验证
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import math
import ast

# 定义数据表头即参数名
headers = ['age', 'workclass', 'fnlwgt',
           'education', 'education.num',
           'marital.status', 'occupation',
           'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss',
           'hours.per.week', 'native.country',
           'income'] 

# age,workclass,fnlwgt,education,education.num,marital.status,occupation,
# relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income
# 列举变量取值
wc = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
      'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
edu = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
       'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th',
       '10th', 'Doctorate', '5th-6th', 'Preschool']
ms = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
      'Married-spouse-absent', 'Married-AF-spouse']
# 将neverworked对应的occupation从？变成none
occu = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'none']
rel = ['Wife', 'Own-child', 'Husband',
       'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
nc = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
      'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
      'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
sex = ['Male','Female']
ic = ['<=50K','>50K']

class CensusNet(nn.Module):
    def __init__(self, num_of_features):
        super().__init__()
        self.fc1 = nn.Linear(num_of_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = torch.flatten(x,1 )
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        output = x # cross entropy in pytorch already includes softmax
        return output

# 加载已有网络并计算二分类数量
def test(model,test_x,device):
    tensor_test_x = torch.FloatTensor(test_x.copy()) # transform to torch tensor 
    test_dataset = TensorDataset(tensor_test_x) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle = False) # create dataloader
    # print(type(test_dataloader))
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # print(size)
    # print(num_batches)
    test_loss, correct = 0, 0
    pos = 0
    neg = 0
    gt50 = torch.tensor([[1,0]])
    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            # x = x.to(device)
            # print(x)
            # print(type(x[0]))
            pred = model(x[0])
            pred = pred.type(torch.FloatTensor)
            # print(pred.shape)
            dim0,dim1 = pred.shape
            for i in range(dim0):
                element = pred[0,:]
                element = element.unsqueeze(0)
                # print(element.shape)
                # print(gt50.shape)
                if(element.argmax(1)==gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size
    negtive = neg /size
    print(postive)
    print(negtive)
    return postive, negtive

# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_r1_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r1.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r1_feature.txt", feature)

# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_r2_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r2.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r2_feature.txt", feature)

# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_r3_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r3.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r3_feature.txt", feature)

# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_r4_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r4.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r4_feature.txt", feature)

# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_r5_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r5.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r5_feature.txt", feature)

def cal_fairness(net):
    device = 'cpu'
    time_start = time.time()
     # 修改网络
    PATH = "gender_fairness/census_modify/"+ str(net) +".pt"
    # 修改输入文件
    test_x_r1 = np.loadtxt('gender_fairness/sample_race/r1_feature.txt')
    test_x_r2 = np.loadtxt('gender_fairness/sample_race/r2_feature.txt') 
    test_x_r3 = np.loadtxt('gender_fairness/sample_race/r3_feature.txt')
    test_x_r4 = np.loadtxt('gender_fairness/sample_race/r4_feature.txt') 
    test_x_r5 = np.loadtxt('gender_fairness/sample_race/r5_feature.txt') 
                   
    model.load_state_dict(torch.load(PATH))
 
    positive_r1 = 0
    positive_r2 = 0
    positive_r3 = 0
    positive_r4 = 0
    positive_r5 = 0
    negitive_r1 = 0
    negitive_r2 = 0
    negitive_r3 = 0
    negitive_r4 = 0
    negitive_r5 = 0

    
    #返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
    pos_r1, neg_r1 = test(model, test_x_r1, device)
    positive_r1 = positive_r1 + pos_r1
    negitive_r1 = negitive_r1 + neg_r1

    pos_r2, neg_r2 = test(model, test_x_r2, device)
    positive_r2 = positive_r2 + pos_r2
    negitive_r2 = negitive_r2 + neg_r2

    pos_r3, neg_r3 = test(model, test_x_r3, device)
    positive_r3 = positive_r3 + pos_r3
    negitive_r3 = negitive_r3 + neg_r3

    pos_r4, neg_r4 = test(model, test_x_r4, device)
    positive_r4 = positive_r4 + pos_r4
    negitive_r4 = negitive_r4 + neg_r4

    pos_r5, neg_r5 = test(model, test_x_r5, device)
    positive_r5 = positive_r5 + pos_r5
    negitive_r5 = negitive_r5 + neg_r5
 
    f1 = abs(positive_r1 - positive_r2)
    f2 = abs(positive_r1 - positive_r3)
    f3 = abs(positive_r1 - positive_r4)
    f4 = abs(positive_r1 - positive_r5)
    f5 = abs(positive_r2 - positive_r3)
    f6 = abs(positive_r2 - positive_r4)
    f7 = abs(positive_r2 - positive_r5)
    f8 = abs(positive_r3 - positive_r4)
    f9 = abs(positive_r3 - positive_r5)
    f10 = abs(positive_r4 - positive_r5)
    fair = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
    fairness = sum(fair) / len(fair)

    time_end = time.time()
    time_sum = time_end - time_start
    
    # n = 10
    # #测试n次得到二分类的平均数
    # for i in range(n):
    #   #返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
    #   pos_male,neg_male = test(model, test_x_male, device)
    #   positive_male = positive_male + pos_male
    #   negtive_male = negtive_male + neg_male

    #   pos_female,neg_female = test(model, test_x_female, device)
    #   positive_female = positive_female + pos_female
    #   negtive_female = negtive_female + neg_female
    # fairness = abs(positive_female/n - positive_male/n)
    # time_end = time.time()
    # # 由于取了平均，时间对应除以n
    # time_sum = (time_end - time_start) / n
    
    # print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male/n, net))#对应性别的>=50K$分类的平均比例
    # print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male/n, net)) #对应性别的<50K$分类的平均比例
    # print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female/n, net))#对应性别的>=50K$分类的平均比例
    # print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female/n, net)) #对应性别的<50K$分类的平均比例
    print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % (net,"race", fairness, time_sum)) 

def recal_acc(model,net):
    time_start = time.time()
    device = 'cpu'
     # 修改网络
    PATH = "gender_fairness/optimize/"+ str(net) +".pt"
    # 修改输入文件
    test_x = np.loadtxt('data4/testx.txt')
    test_y = np.loadtxt('data4/testy.txt') 

    tensor_test_x = torch.FloatTensor(test_x.copy())                                                            
    tensor_test_y = torch.FloatTensor(test_y.copy()) 

    test_dataset = TensorDataset(tensor_test_x, tensor_test_y) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100) # create dataloader

    model.load_state_dict(torch.load(PATH))
 
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            
            pred = model(x)
            pred = pred.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor)
            # print(pred)
            # print(y)
            test_loss += loss_fn(pred, y).item()
            # print(pred.argmax(1))
            # print(y.argmax(1))
            correct += (pred.argmax(1) == y.argmax(1)).sum().item()
    
    test_loss /= num_batches
    error = size - correct
    correct /= size

    print(f"Test: \n Test size: {(size):>0.1f}, Error size: {(error):>0.1f}, Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    # return correct

if __name__ == "__main__":
    # calculate_r1_acc()
    # calculate_r2_acc()
    # calculate_r3_acc()
    # calculate_r4_acc()
    # calculate_r5_acc()
    model = CensusNet(14)
    # recal_acc(model,'optimizednet1')
    cal_fairness('census')
    



