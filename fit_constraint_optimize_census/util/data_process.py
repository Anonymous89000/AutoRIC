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

device='cpu'

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
def rtest(model,test_x, device):
    tensor_test_x = torch.FloatTensor(test_x.copy()) # transform to torch tensor 
    test_dataset = TensorDataset(tensor_test_x) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle = True) # create dataloader
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
    return postive, negtive
    
# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_male_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_sex/gen_male.csv", "r") as ins:
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
    np.savetxt("gender_fairness/sample_sex/male_feature.txt", feature)
    
# 数据处理函数，生成可以用于输入网络的数据文件
def calculate_female_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_sex/gen_female.csv", "r") as ins:
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
    np.savetxt("gender_fairness/sample_sex/female_feature.txt", feature)

def cal_fairness(net):
    time_start = time.time()
     # 修改网络
    PATH = "result/"+ str(net) +".pt"
    # 修改输入文件
    test_x_female = np.loadtxt('gender_fairness/sample_sex/female_feature.txt')
    test_x_male = np.loadtxt('gender_fairness/sample_sex/male_feature.txt') 

    model.load_state_dict(torch.load(PATH))
 
    positive_male = 0
    positive_female = 0
    negtive_male = 0
    negtive_female = 0
    n = 10
    #测试n次得到二分类的平均数
    for i in range(n):
      #返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
      pos_male,neg_male = rtest(model, test_x_male, device)
      positive_male = positive_male + pos_male
      negtive_male = negtive_male + neg_male

      pos_female,neg_female = rtest(model, test_x_female, device)
      positive_female = positive_female + pos_female
      negtive_female = negtive_female + neg_female
    fairness = abs(positive_female/n - positive_male/n)
    time_end = time.time()
    # 由于取了平均，时间对应除以n
    time_sum = (time_end - time_start) / n
    
    print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male/n, net))#对应性别的>=50K$分类的平均比例
    print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male/n, net)) #对应性别的<50K$分类的平均比例
    print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female/n, net))#对应性别的>=50K$分类的平均比例
    print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female/n, net)) #对应性别的<50K$分类的平均比例
    print("网络参数扰动为%f下，公平性取值为：%f, 所用时间为：%fs\n" % (net, fairness, time_sum)) 

def cal_fairness1(model,device='cpu'):
    time_start = time.time()
    # 修改网络


    # 修改输入文件
    test_x_female = np.loadtxt('data/female_feature.txt')
    test_x_male = np.loadtxt('data/male_feature.txt')

    model1=CensusNet(14)
    model1.load_state_dict(model)
    positive_male = 0
    positive_female = 0
    negtive_male = 0
    negtive_female = 0
    n = 10
    # 测试n次得到二分类的平均数
    for i in range(n):
        # 返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
        pos_male, neg_male = rtest(model1, test_x_male, device)
        positive_male = positive_male + pos_male
        negtive_male = negtive_male + neg_male

        pos_female, neg_female = rtest(model1, test_x_female, device)
        positive_female = positive_female + pos_female
        negtive_female = negtive_female + neg_female
    fairness = abs(positive_female / n - positive_male / n)
    time_end = time.time()
    # 由于取了平均，时间对应除以n
    time_sum = (time_end - time_start) / n

    # print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male / n, 1))  # 对应性别的>=50K$分类的平均比例
    # print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male / n, 1))  # 对应性别的<50K$分类的平均比例
    # print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female / n, 1))  # 对应性别的>=50K$分类的平均比例
    # print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female / n, 1))  # 对应性别的<50K$分类的平均比例
    # print("网络参数扰动为%f下，公平性取值为：%f, 所用时间为：%fs\n" % (1, fairness, time_sum))
    print("一次公平性计算完成")
    return  fairness



def recal_acc(model,net):
    time_start = time.time()
    device = 'cpu'
     # 修改网络
    PATH = str(net)
    # 修改输入文件
    test_x = np.loadtxt('data/testx.txt')
    test_y = np.loadtxt('data/testy.txt')

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
            #print(x)
            #print(x.shape)
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

    return correct

if __name__ == "__main__":
    # calculate_female_acc()
    # calculate_male_acc()
    model = CensusNet(14)
    recal_acc(model,'result/optimizednetys_1.pt')
    model=torch.load('result/optimizednetys_1.pt')
    sum=0
    for i in range(10):
        tmp=cal_fairness1(model)
        print(tmp)
        sum+=tmp
    sum=sum/10
    print("公平性：",sum)

    #recal_acc(model,'data/census.pt')
    



