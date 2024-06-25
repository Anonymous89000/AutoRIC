import random
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
import util
import util.data_process
import util.fit_optimize
import cvx_optimize
import  constraint_by_linearize
import  quadratic_fit

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
disturbs_0=[0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]
nettype_list=[('pnet_c_0_24',24),("pnet_b_0_64",64),('pnet_a_0_128',128)]
params_index_list=[params_index_c,params_index_b,params_index_a]

if __name__=='__main__':
    #获取预测正确以及预测错误的数据的索引 从中挑选适合的数据生成对应约束
    positive_indice=[]
    negative_indice=[]
    positive_indice=np.loadtxt('data/positive_list.txt')
    negative_indice=np.loadtxt('data/negative_list.txt')

    #默认升序排列
    positive_indice=sorted(positive_indice,key=lambda ele:ele[1])
    negative_indice=sorted(negative_indice,key=lambda  ele:ele[1])

    positive_indice=[int(i) for (i,_) in positive_indice]
    negative_indice=[int(i) for (i,_) in negative_indice]

    # 是否反转数组 是否重新随机打乱

    # positive_indice.reverse()
    #negative_indice.reverse()

    #random.shuffle(positive_indice)
    # random.shuffle(negative_indice)

   # print(negative_indice)
    #由于约束求解有时会因无解返回异常 因此使用try except

    #constraint_indice=range(len(positive_indice)+len(negative_indice))

    #1 + 0.0016
    #
    posi_factor=0.0004

    nega_factor=0.004
    posi_num=int(len(positive_indice)*posi_factor)-1
    nega_num=int(len(negative_indice)*nega_factor)-1
    if(posi_num<0):
        posi_num=0
    if(nega_num<0):
        nega_num=0

    constraint_indice=positive_indice[0:posi_num]+negative_indice[0:nega_num]
    print(posi_num)
    print(nega_num)
    print(len(constraint_indice))
    #constraint_indice=negative_indice

    # net1= 'data/census.pt'
    # nettype='pnet_c_0_24'
    # net=nettype
    # for i in range(3):
    #     nettype=nettype_list[i][0]
    #     param_num=nettype_list[i][1]
    #     params_index=params_index_list[i]
    #     quadratic_fit.quadratic_fit(net1,param_num,nettype,params_index,disturbs_0)

    # net1= 'data/census.pt'
    # nettype=nettype_list[2][0]
    # param_num=nettype_list[2][1]
    # params_index=params_index_list[2]
    # quadratic_fit.quadratic_fit(net1,param_num,nettype,params_index,disturbs_0)


    net='bnet24'
    params_index=params_index_c

    # net='pnet_c_0_24'
    # params_index=params_index_c
    constraint_by_linearize.get_constraint(params_index, constraint_indice, net)
    cvx_optimize.cvx_optimize(net,params_index,0)


