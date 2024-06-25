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
headers = ['age', 'job', 'marital',
               'education', 'default',
               'balance', 'housing',
               'loan', 'contact', 'day',
               'month', 'duration',
               'campaign', 'pdays',
               'previous', 'poutcome',
               'y'] 

# list all the values of enumerate features
job = ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
                                       "blue-collar","self-employed","retired","technician","services"]
marital = ["married","divorced","single"]
education = ["unknown","secondary","primary","tertiary"]
default = ["no","yes"]
housing = ["no","yes"]
loan = ["no","yes"]
contact = ["unknown","telephone","cellular"]
month = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
poutcome = ["unknown","other","failure","success"]
output = ["no","yes"]


class BankNet(nn.Module):
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


def recal_acc(model, net):
    time_start = time.time()
    device = 'cpu'
    # 修改网络
    PATH = str(net)
    # 修改输入文件
    test_x = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/data/testx.txt')
    test_y = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/data/testy.txt')

    tensor_test_x = torch.FloatTensor(test_x.copy())
    tensor_test_y = torch.FloatTensor(test_y.copy())

    test_dataset = TensorDataset(tensor_test_x,
                                 tensor_test_y)  # create dataset
    test_dataloader = DataLoader(test_dataset, batch_size=100)  # create dataloader

    model.load_state_dict(torch.load(PATH))

    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in test_dataloader:
            x, y = x.to(device), y.to(device)
            # print(x)
            # print(x.shape)
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

    print(
        f"Test: \n Test size: {(size):>0.1f}, Error size: {(error):>0.1f}, Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct
# 加载已有网络并计算二分类数量
def rtest(model,test_x,device):
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
                #如果element较大的值的index和[1,0]保持一致
                if(element.argmax(1)==gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size#此时postive表示"yes",即[1,0]
    negtive = neg /size
    return postive, negtive
    
# 将字符转换成数字，用于输入到网络
def split_a1():
    feature = []
    # income = []
    with open("bank_fairness/sample_age/gen_a1.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            # print(features)
            # features = features[0].split(',')
            features[0] = np.clip(int(features[1]) / 10, 1, 9)
            # print(features[0])
            features[1] = job.index(features[2])
            features[2] = marital.index(features[3])
            features[3] = education.index(features[4])
            features[4] = default.index(features[5])
            features[5] = np.clip(int(features[6]) / 50, -20, 180 - 1)
            features[6] = housing.index(features[7])
            features[7] = loan.index(features[8])
            features[8] = contact.index(features[9])
            features[9] = int(features[10])
            features[10] = month.index(features[11])
            features[11] = np.clip(int(features[12]) / 10, 0, 100 - 1)
            features[12] = int(features[13])
            if int(features[13])==-1:
                features[13] = 1
            else:
                features[13] = 0
            features[14] = np.clip(int(features[15]), 0, 1)
            # print(features[16])
            features[15] = poutcome.index(features[16])
            # print(features[:16])
            feature.append(features[:16])
            # print(features[16])
            # print(output)
            # features[16] = output.index(features[16].split('\"')[1])
            # s = features[16]
            # if(s==0):
            #     label.append([0,1])
            # else:
            #     label.append([1,0])  

    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("bank_fairness/sample_age/a1_feature.txt", feature)

def split_a2():
    feature = []
    # income = []
    with open("bank_fairness/sample_age/gen_a2.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            # print(features)
            # features = features[0].split(',')
            features[0] = np.clip(int(features[1]) / 10, 1, 9)
            # print(features[0])
            features[1] = job.index(features[2])
            features[2] = marital.index(features[3])
            features[3] = education.index(features[4])
            features[4] = default.index(features[5])
            features[5] = np.clip(int(features[6]) / 50, -20, 180 - 1)
            features[6] = housing.index(features[7])
            features[7] = loan.index(features[8])
            features[8] = contact.index(features[9])
            features[9] = int(features[10])
            features[10] = month.index(features[11])
            features[11] = np.clip(int(features[12]) / 10, 0, 100 - 1)
            features[12] = int(features[13])
            if int(features[13])==-1:
                features[13] = 1
            else:
                features[13] = 0
            features[14] = np.clip(int(features[15]), 0, 1)
            # print(features[16])
            features[15] = poutcome.index(features[16])
            # print(features[:16])
            feature.append(features[:16])
            # print(features[16])
            # print(output)
            # features[16] = output.index(features[16].split('\"')[1])
            # s = features[16]
            # if(s==0):
            #     label.append([0,1])
            # else:
            #     label.append([1,0])  

    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("bank_fairness/sample_age/a2_feature.txt", feature)

def split_a3():
    feature = []
    # income = []
    with open("bank_fairness/sample_age/gen_a3.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            # print(features)
            # features = features[0].split(',')
            features[0] = np.clip(int(features[1]) / 10, 1, 9)
            # print(features[0])
            features[1] = job.index(features[2])
            features[2] = marital.index(features[3])
            features[3] = education.index(features[4])
            features[4] = default.index(features[5])
            features[5] = np.clip(int(features[6]) / 50, -20, 180 - 1)
            features[6] = housing.index(features[7])
            features[7] = loan.index(features[8])
            features[8] = contact.index(features[9])
            features[9] = int(features[10])
            features[10] = month.index(features[11])
            features[11] = np.clip(int(features[12]) / 10, 0, 100 - 1)
            features[12] = int(features[13])
            if int(features[13])==-1:
                features[13] = 1
            else:
                features[13] = 0
            features[14] = np.clip(int(features[15]), 0, 1)
            # print(features[16])
            features[15] = poutcome.index(features[16])
            # print(features[:16])
            feature.append(features[:16])
            # print(features[16])
            # print(output)
            # features[16] = output.index(features[16].split('\"')[1])
            # s = features[16]
            # if(s==0):
            #     label.append([0,1])
            # else:
            #     label.append([1,0])  

    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("bank_fairness/sample_age/a3_feature.txt", feature)

def split_a4():
    feature = []
    # income = []
    with open("bank_fairness/sample_age/gen_a4.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            # print(features)
            # features = features[0].split(',')
            features[0] = np.clip(int(features[1]) / 10, 1, 9)
            # print(features[0])
            features[1] = job.index(features[2])
            features[2] = marital.index(features[3])
            features[3] = education.index(features[4])
            features[4] = default.index(features[5])
            features[5] = np.clip(int(features[6]) / 50, -20, 180 - 1)
            features[6] = housing.index(features[7])
            features[7] = loan.index(features[8])
            features[8] = contact.index(features[9])
            features[9] = int(features[10])
            features[10] = month.index(features[11])
            features[11] = np.clip(int(features[12]) / 10, 0, 100 - 1)
            features[12] = int(features[13])
            if int(features[13])==-1:
                features[13] = 1
            else:
                features[13] = 0
            features[14] = np.clip(int(features[15]), 0, 1)
            # print(features[16])
            features[15] = poutcome.index(features[16])
            # print(features[:16])
            feature.append(features[:16])
            # print(features[16])
            # print(output)
            # features[16] = output.index(features[16].split('\"')[1])
            # s = features[16]
            # if(s==0):
            #     label.append([0,1])
            # else:
            #     label.append([1,0])  

    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("bank_fairness/sample_age/a4_feature.txt", feature)


def cal_fairness1(model):
    time_start = time.time()
    # 修改网络
    # PATH = "census_fairness/census_modify/"+ str(net) +".pt"

    # 修改输入文件
    test_x_a1 = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/util/bank_fairness/sample_age/a1_feature.txt')
    test_x_a2 = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/util/bank_fairness/sample_age/a2_feature.txt')
    test_x_a3 = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/util/bank_fairness/sample_age/a3_feature.txt')
    test_x_a4 = np.loadtxt('C:/Users\dell/Desktop/project/coding/bank_optimize/util/bank_fairness/sample_age/a4_feature.txt')

    model1=BankNet(16)
    model1.load_state_dict(model)
    model=model1
    positive_a1 = 0
    positive_a2 = 0
    positive_a3 = 0
    positive_a4 = 0
    negtive_a1 = 0
    negtive_a2 = 0
    negtive_a3 = 0
    negtive_a4 = 0
    n = 10
    device = 'cpu'
    # 测试n次得到二分类的平均数
    for i in range(n):
        # 返回二分类的比例，pos表示"yes"分类的比例，neg表示"no"分类的比例，pos+neg=1
        pos_a1, neg_a1 = rtest(model, test_x_a1, device)
        positive_a1 = positive_a1 + pos_a1
        #   print(positive_a1)
        negtive_a1 = negtive_a1 + neg_a1

        pos_a2, neg_a2 = rtest(model, test_x_a2, device)
        positive_a2 = positive_a2 + pos_a2
        #   print(positive_a1)
        negtive_a2 = negtive_a2 + neg_a2

        pos_a3, neg_a3 = rtest(model, test_x_a3, device)
        positive_a3 = positive_a3 + pos_a3
        #   print(positive_a1)
        negtive_a3 = negtive_a3 + neg_a3

        pos_a4, neg_a4 = rtest(model, test_x_a4, device)
        positive_a4 = positive_a4 + pos_a4
        #   print(positive_a1)
        negtive_a4 = negtive_a4 + neg_a4

    f1 = abs(positive_a1 / n - positive_a2 / n)
    f2 = abs(positive_a1 / n - positive_a3 / n)
    f3 = abs(positive_a1 / n - positive_a4 / n)
    f4 = abs(positive_a2 / n - positive_a3 / n)
    f5 = abs(positive_a2 / n - positive_a4 / n)
    f6 = abs(positive_a3 / n - positive_a4 / n)
    f = (f1 + f2 + f3 + f4 + f5 + f6) / 6
    time_end = time.time()
    # 由于取了平均，时间对应除以n
    time_sum = (time_end - time_start) / n
   # print("sss")
    # print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male/n, net))#对应性别的>=50K$分类的平均比例
    # print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male/n, net)) #对应性别的<50K$分类的平均比例
    # print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female/n, net))#对应性别的>=50K$分类的平均比例
    # print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female/n, net)) #对应性别的<50K$分类的平均比例
    print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % ("test","age", f, time_sum))
    return  f

def cal_fairness(model,net):
    time_start = time.time()
     # 修改网络
    # PATH = "census_fairness/census_modify/"+ str(net) +".pt"
    PATH = "../data/bank.pt"
    # 修改输入文件
    test_x_a1 = np.loadtxt('bank_fairness/sample_age/a1_feature.txt')
    test_x_a2 = np.loadtxt('bank_fairness/sample_age/a2_feature.txt')
    test_x_a3 = np.loadtxt('bank_fairness/sample_age/a3_feature.txt')
    test_x_a4 = np.loadtxt('bank_fairness/sample_age/a4_feature.txt')

    model.load_state_dict(torch.load(PATH))
 
    positive_a1 = 0
    positive_a2 = 0
    positive_a3 = 0
    positive_a4 = 0
    negtive_a1 = 0
    negtive_a2 = 0
    negtive_a3 = 0
    negtive_a4 = 0
    n = 10
    device = 'cpu'
    #测试n次得到二分类的平均数
    for i in range(n):
      #返回二分类的比例，pos表示"yes"分类的比例，neg表示"no"分类的比例，pos+neg=1
      pos_a1,neg_a1 = rtest(model, test_x_a1, device)
      positive_a1 = positive_a1 + pos_a1
    #   print(positive_a1)
      negtive_a1 = negtive_a1 + neg_a1

      pos_a2,neg_a2 = rtest(model, test_x_a2, device)
      positive_a2 = positive_a2 + pos_a2
    #   print(positive_a1)
      negtive_a2 = negtive_a2 + neg_a2

      pos_a3,neg_a3 = rtest(model, test_x_a3, device)
      positive_a3 = positive_a3 + pos_a3
    #   print(positive_a1)
      negtive_a3 = negtive_a3 + neg_a3

      pos_a4,neg_a4 = rtest(model, test_x_a4, device)
      positive_a4 = positive_a4 + pos_a4
    #   print(positive_a1)
      negtive_a4 = negtive_a4 + neg_a4

    f1 = abs(positive_a1/n - positive_a2/n)
    f2 = abs(positive_a1/n - positive_a3/n)
    f3 = abs(positive_a1/n - positive_a4/n)
    f4 = abs(positive_a2/n - positive_a3/n)
    f5 = abs(positive_a2/n - positive_a4/n)
    f6 = abs(positive_a3/n - positive_a4/n)
    f = (f1 + f2 + f3 + f4 + f5 + f6) / 6
    time_end = time.time()
    # 由于取了平均，时间对应除以n
    time_sum = (time_end - time_start) / n
    print("sss")
    # print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male/n, net))#对应性别的>=50K$分类的平均比例
    # print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male/n, net)) #对应性别的<50K$分类的平均比例
    # print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female/n, net))#对应性别的>=50K$分类的平均比例
    # print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female/n, net)) #对应性别的<50K$分类的平均比例
    print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % (net,"age", f, time_sum)) 



if __name__ == "__main__":

    model=torch.load("../data/bank.pt")
    cal_fairness1(model)
    model=BankNet(16)
    recal_acc(model, '../data/bank.pt')

    model=torch.load("../result/bnet24_opt.pt")
    cal_fairness1(model)
    model=BankNet(16)
    recal_acc(model,"../result/b01.pt")

    # model=BankNet(16)
    # cal_fairness(model,"")

    # 用于将字符型数据转换成数字型
    # split_a1()
    # split_a2()
    # split_a3()
    # split_a4()



