#通过线性化得到不等式形式的约束 保存在文件中
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





params_index_c = [
    (3, 7),
    (3, 11),
    (3, 12)
]

def get_constraint(params_index,indice,netname):
    params_layer0=params_index[0][0]
    params_layer=params_layer0*2

    # 初始化模型 并且使用参数加载方式加载训练好的模型
    network = BankNet(16)
    network.load_state_dict(torch.load('data/bank.pt'))
    layers = []

    # 准备输入数据以及标签
    inputs = np.loadtxt('data/testx.txt')
    labels = np.loadtxt('data/testy.txt')
    # 33916*104  33916*2
    # print(inputs.shape)
    # print(labels.shape)

    # 将我们原始形式的网络转换为prdnn要求的Network形式
    parameters_set = list(network.parameters())
    for index, layer in enumerate(list(network.named_modules())[1:]):
        if isinstance(layer[1], nn.Linear):
            # print(parameters_set[index//2])
            # 似乎network.parameters方法里的权重是转置后的因此要转置回来
            # print(parameters_set[2*index].shape)
            # 此处转置操作 警告不能对于大于2维的向量使用
            weights = parameters_set[2 * index].T.clone().detach()
            print(weights.size())
            biases = parameters_set[2 * index + 1].T.clone().detach()
            print(biases.size())
            # if (2 * index == 10):
            #     print("weights", weights)
            #     print("bias", biases)
            layers.append(FullyConnectedLayer(weights, biases))
            layers.append(ReluLayer())
        else:
            print("error")
    # 输出层没有relu
    layers = layers[:-1]

    print(len(layers))
    print(layers)
    print(layers[6].weights)

    indices = indice

    # indices=[i for i in range(1000)]
    network1 = Network(layers)
    network_pre = Network(layers[:-2])
    pre_v = network_pre.compute(inputs[indices])

    provablerepair = ProvableRepair(network1, params_layer, inputs, labels, inputs)
    # 6/10

    # 上面的 index参数实际是network1这个列表里的索引 以长度12的列表为例 0 2 4 6 8 10 为linearlayer 不能填1 3 5 7 9 11因为是relulayer
    # 现在尝试用最后一层 和 最后两层线性层做实验 实际计算
    res1_A, res1_B = provablerepair.network_jacobian(indices)

    constraint_A = []
    constraint_b = []





    #以下为手动输入的神经元索引以及权重 现在要改成根据params_index自动提取
    # neuron_indice = [48, 49, 50, 51, 52, 53, 54, 55, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    # batch_A = res1_A
    # batch_b = res1_B
    # x0 = [
    #     0.2893, -0.2873, 0.0404, 0.0476, -0.2193, 0.2644, -0.0524, 0.2096,
    #     -0.2123, 0.1015, -0.1486, -0.0139, 0.0007, 0.0810, -0.0760, -0.2986,
    #     -0.4043, 0.1651, 0.0472, 0.2808, 0.0555, 0.1248, -0.2412, -0.0129
    # ]
    #

    param_num=[64,32,16,8,4,2]
    neuron_indice=[]

    batch_A=res1_A
    batch_b=res1_B
    for ele in params_index:
        for j in range(param_num[ele[0]]):
            neuron_indice.append((ele[1]-1)*param_num[ele[0]]+j)

    layers_weights=layers[params_layer].weights
    x0=[]
    for ele in params_index:
        x0+=layers_weights[ele[1]-1]
        #x0.append(layers_weights[ele[1]-1])
    print(neuron_indice)
    x0 = list(x0)
    print(x0)







    for i, label in enumerate(np.argmax(labels[indices], axis=1)):
        # res1_A,res1_B
        print(i, label)
        if (label == 0):
            # tmp_res = batch_A[i]*deltax + batch_b[i]
            # tmp_res[0]>=tmp_res[1]:

            # 约束需要满足的形式
            batch_A_sub = [batch_A[i][0][j] - batch_A[i][1][j] for j in range(len(batch_A[i][0]))]
            batch_A_sub = np.array(batch_A_sub)
            batch_A_sub = batch_A_sub[neuron_indice]

            constraint_A.append(list(batch_A_sub))
            temp_b = sum([batch_A_sub[j] * x0[j] for j in range(len(batch_A_sub))]) - (batch_b[i][0] - batch_b[i][1])
            constraint_b.append(temp_b)
        elif (label == 1):
            # tmp_res = batch_A[i]*deltax + batch_b[i]
            # tmp_res[1]>=tmp_res[0]:

            batch_A_sub = [batch_A[i][1][j] - batch_A[i][0][j] for j in range(len(batch_A[i][0]))]
            batch_A_sub = np.array(batch_A_sub)
            batch_A_sub = batch_A_sub[neuron_indice]
            constraint_A.append(list(batch_A_sub))
            temp_b = sum([batch_A_sub[j] * x0[j] for j in range(len(batch_A_sub))]) - (batch_b[i][1] - batch_b[i][0])
            constraint_b.append(temp_b)

        else:
            print("产生错误！", label)


        #得到constraint_A以及constraint_b以后进行保存! 并注意文件名体现出是哪一个网络的约束
    #以上得到了Ax>=b的A以及b   但是调用需要<= 因此有-Ax<=-b  将它们分别取反
    constraint_A_inverse=[]
    constraint_b_inverse=[]

    for i in constraint_A:
        comstraint_tmp=[-l for l in i]
        constraint_A_inverse.append(comstraint_tmp)

    constraint_b_inverse=[-l for l in constraint_b]


    filename_A='constraint/'+netname+'_A.txt'
    filename_b="constraint/"+netname+'_b.txt'
    np.savetxt(filename_A,constraint_A_inverse)
    np.savetxt(filename_b,constraint_b_inverse)




if __name__=='__main__':
    get_constraint(params_index_c,[1,2,3,4,5],'bnet24')


