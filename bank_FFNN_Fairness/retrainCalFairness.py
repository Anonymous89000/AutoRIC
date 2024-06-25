import sys
import os
import pandas as pd
import torch
from torch import  nn
from  torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from tqdm import tqdm
from pysyrenn.frontend import Network, FullyConnectedLayer
from pysyrenn.frontend import Conv2DLayer, ReluLayer
from pysyrenn.frontend import ConcatLayer, HardTanhLayer
from gurobipy import Model, GRB
from timeit import default_timer as timer
from pysyrenn import Network
from prdnn import ProvableRepair
import util.data_process

import time
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
def rtest(model, test_x, device):
    tensor_test_x = torch.FloatTensor(test_x.copy())  # transform to torch tensor
    test_dataset = TensorDataset(tensor_test_x)  # create dataset
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)  # create dataloader
    # print(type(test_dataloader))
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # print(size)
    # print(num_batches)
    test_loss, correct = 0, 0
    pos = 0
    neg = 0
    gt50 = torch.tensor([[1, 0]])
    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            # x = x.to(device)
            # print(x)
            # print(type(x[0]))
            pred = model(x[0])
            pred = pred.type(torch.FloatTensor)
            # print(pred.shape)
            dim0, dim1 = pred.shape
            for i in range(dim0):
                element = pred[0, :]
                element = element.unsqueeze(0)
                # print(element.shape)
                # print(gt50.shape)
                if (element.argmax(1) == gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size
    negtive = neg / size
    return postive, negtive

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
    #print("网络名称为：%s，特征%s的公平性取值为：%f, 所用时间为：%fs\n" % ("test","age", f, time_sum))
    print("一次公平性计算完成")
    return  f


if __name__=='__main__':
    inputfile='retrain/age_retraintable.xlsx'
    df = pd.read_excel(inputfile)
    current_model_count = df.loc[0, '统计数量'] if not df.empty and '统计数量' in df.columns else 1
    accuracy_raw=0.892
    fairness_raw=0.048
    for i in range(current_model_count):
        modelname=df.loc[i,'模型名称']
        statedict=torch.load('retrain/model/'+modelname+'.pt')
        modelfairness=cal_fairness1(statedict)
        modelaccuracy=df.loc[i,'准确性']
        fairness_change=(modelfairness-fairness_raw)/fairness_raw
        accuracy_change=(modelaccuracy-accuracy_raw)/accuracy_raw
        df.loc[i,'公平性']=modelfairness
        df.loc[i,'公平性变化']=fairness_change
        df.loc[i,'准确性变化']=accuracy_change

    df.to_excel(inputfile, index=False)

