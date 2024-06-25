#直接利用二次规划进行优化 结果保存在文件中

import sys
import os
import time

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
import util
import util.data_process
import util.fit_optimize
from scipy.optimize import linprog

#128
params_index_a=[
    (1, 1),
    (1, 24),
    (1, 33),
    (1, 45)
]
#64
params_index_b=[
    (2, 7),
    (2, 11),
    (2, 18),
    (2, 25)
]
#24
params_index_c=[
    (3, 7),
    (3, 11),
    (3, 12)
]
#222
params_index_o = [
    (1, 1),
    (1, 24),
    (1, 33),
    (1, 45),
    (2, 7),
    (2, 11),
    (2,18),
    (2, 25),
    (3, 7),
    (3, 11),
    (3, 12),
    (4, 3),
    (5, 3)
]

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
        x = torch.flatten(x, 1)
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
        output = x  # cross entropy in pytorch already includes softmax
        return output

def linear_optimize(netname,params_index,flag):

    #flag==0时为无约束优化
    # filename_Q1='quadratic_para/'+netname+'_Q1.txt'
    # filename_b1='quadratic_para/'+netname+'_b1.txt'
    # filename_c1='quadratic_para/'+netname+'_c1.txt'
    #
    # Q1=np.loadtxt(filename_Q1 )
    # b1=np.loadtxt(filename_b1)
    # c1=np.loadtxt(filename_c1)

    filename_c='linear_para/'+netname+'_c.txt'
    c=np.loadtxt(filename_c)

    filename_A='constraint/'+netname+'_A.txt'
    filename_b="constraint/"+netname+'_b.txt'
    constraint_A=np.loadtxt(filename_A)
    constraint_b=np.loadtxt(filename_b)

    print("c:",len(c))
    print("A:",constraint_A.shape)
    print("b:",constraint_b.shape)




    #https://blog.csdn.net/u013421629/article/details/108358409
    constraint_b=[[i] for i in constraint_b]
    # print(constraint_A)
    # print(constraint_b)
    constraint_A=np.array(constraint_A)
    constraint_b=np.array(constraint_b)
    constraint_A=np.float64(constraint_A)
    constraint_b=np.float64(constraint_b)

    start_time=time.time()
    boundsarray=[]
    for i in range(len(constraint_A[0])):
        boundsarray.append((-2,2))

    if(flag==0):
        print("无约束优化！")
        res=util.fit_optimize.linprog(c)
    else:
        print("有约束优化！")
        res = util.fit_optimize.linprog( c,constraint_A,constraint_b,bounds=boundsarray)
    print(res)

    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"优化块运行时间：{elapsed_time} 秒")
    res=res.x
    model = torch.load("data/census.pt")

    count=0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index + 1}.weight'
        tmp_matrix = model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            #tmp_matrix[neuron_index][j] = res[count][0]
            tmp_matrix[neuron_index][j] = res[count]
            count += 1
        model[key] = tmp_matrix.T

    # 这里要改动
    optimized_net='result/'+netname+'_lopt.pt'
    torch.save(model, optimized_net)
    print(count)

    sum = 0
    for j in range(0, 10):
        fairness = util.fit_optimize.cal_fairness1(model.copy())
        print(fairness)
        sum += fairness
    sum = sum / 10
    print("优化后的网络公平性:", sum)



    model=util.data_process.CensusNet(14)
    optimized_acc=util.data_process.recal_acc(model, optimized_net)
    print("优化后的准确性：",optimized_acc)
    exit()

    # print(type(res1))
    # res1=torch.tensor(res1[0])
    # #不加res1[0]会出问题
    # res0=torch.tensor(res0[0])
    # res2=torch.tensor(res2[0])
    #
    # print(res0.shape)
    # print(res1.shape)
    # print(res2.shape)
if __name__=='__main__':
    linear_optimize('Ltestnet24',params_index_c,1)