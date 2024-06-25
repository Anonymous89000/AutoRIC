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
import util.fit_optimize
import util.data_process

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
        output = x # cross entropy in pytorch already includes softmax
        return output

# layers = []
# net_file = open(r'C:\Users\dell\Desktop\project\coding\PRDNN\PRDNN-master\experiments\external\mnist_relu_3_100_model\file\model.eran', "r")
# #print(len(net_file.readline()))
# #"RELU "
# curr_line = net_file.readline()[:-1]
# print(curr_line)

params_index_c=[
    (3, 7),
    (3, 11),
    (3, 12)
]


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





# inputs=torch.ones((100,104))
# batch_size=10
# labels=torch.tensor([[0,1]]*100)

# print("查看网络内部层次结构并输出:")
#network=CensusNet(104)
# network.named_modules 返回每一层的名字和层类型
# for layer in network.named_modules():
#     print(layer)


#通过以下实验 每一层都有两个参数 权重W和偏置b
# network=CensusNet(104)
# print(len(list(network.named_parameters())))
# for para in network.named_parameters():
#     print(len(list(para)))
#     print(para)
#     print(para[0])
#     a=para[1]
#     a=a.detach().numpy()
#     print(a.shape)
#     print(a)

# print(len(list(network.named_modules())))
# for layer in network.named_modules():
#     print(len(list(layer)))
#     print(layer)
#
# exit()


# network=CensusNet_alpha(104)
# for layer in network.named_modules():
#     print(layer)


# 初始化模型 并且使用参数加载方式加载训练好的模型
network=BankNet(16)
network.load_state_dict(torch.load('data/bank.pt'))
layers=[]

#准备输入数据以及标签
inputs = np.loadtxt('data/testx.txt')
labels = np.loadtxt('data/testy.txt')
# 33916*104  33916*2
# print(inputs.shape)
# print(labels.shape)




#将我们原始形式的网络转换为prdnn要求的Network形式
parameters_set=list(network.parameters())
for index,layer in enumerate(list(network.named_modules())[1:]):
    if isinstance(layer[1],nn.Linear):
        #print(parameters_set[index//2])
        #似乎network.parameters方法里的权重是转置后的因此要转置回来
        #print(parameters_set[2*index].shape)
        #此处转置操作 警告不能对于大于2维的向量使用
        weights=parameters_set[2*index].T.clone().detach()
        print(weights.size())
        biases=parameters_set[2*index+1].T.clone().detach()
        print(biases.size())
        if (2*index==10):
            print("weights",weights)
            print("bias",biases)
        layers.append(FullyConnectedLayer(weights, biases))
        layers.append(ReluLayer())
    else:
        print("error")
#输出层没有relu
layers=layers[:-1]

print(len(layers))
print(layers)
#要修复第六层
print(layers[6])
print(layers[6].weights)

#[1,2,3]
indices=[1,2]


#indices=[i for i in range(1000)]
network1=Network(layers)
network_pre=Network(layers[:-2])
pre_v=network_pre.compute(inputs[indices])

provablerepair=ProvableRepair(network1,6,inputs,labels,inputs)
#6/10

#上面的 index参数实际是network1这个列表里的索引 以长度12的列表为例 0 2 4 6 8 10 为linearlayer 不能填1 3 5 7 9 11因为是relulayer
#现在尝试用最后一层 和 最后两层线性层做实验 实际计算


res0=provablerepair.network_jacobian([1])
res1=provablerepair.network_jacobian([1,2,3])
res2=provablerepair.network_jacobian([1,2,3,4,5])

res1_A,res1_B=provablerepair.network_jacobian(indices)
print(type(res1_A),type(res1_B))
print(res1_A.shape,res1_B.shape)
print("res1_A,res1_B:\n",res1_A,res1_B)
print('res1_A[0](2*10):\n',res1_A[0])
print("res1_B[0](2*1):\n",res1_B[0])


print("pre_v :\n",pre_v.shape,'\n',pre_v)
print("pre_v[0]:\n",pre_v[0])

indices=range(len(inputs))
pre_raw=network1.compute(inputs[indices])
print(pre_raw)
#我们的网络的预测结果
pre_list=np.argmax(network1.compute(inputs[indices]),axis=1)
print(pre_list)

#标签标注的结果
label_list=np.argmax(labels[indices],axis=1)
print(label_list)
positive_list=[]
negative_list=[]
for i in indices:
    if  pre_list[i]==label_list[i]:
        #预测正确的 数值越大代表预测结果越可信
        tmp_pair=(i,pre_raw[i][pre_list[i]]-pre_raw[i][1-pre_list[i]])
        positive_list.append(tmp_pair)
    else:
        #预测错误的  数值越大代表预测结果错误越大
        tmp_pair = (i, pre_raw[i][pre_list[i]] - pre_raw[i][1 - pre_list[i]])
        negative_list.append(tmp_pair)
print("acc:",len(positive_list)/len(indices))
print('wrg:',len(negative_list)/len(indices))
positive_list_file='data/positive_list.txt'
negative_list_file='data/negative_list.txt'
np.savetxt(positive_list_file,positive_list)
np.savetxt(negative_list_file,negative_list)

exit()

constraint_A=[]
constraint_b=[]


params_index_c=[
    (3, 7),
    (3, 11),
    (3, 12)
]

params_index=params_index_c

neuron_indice=[48,49,50,51,52,53,54,55,80,81,82,83,84,85,86,87, 88,89,90,91,92,93,94,95]
batch_A=res1_A
batch_b=res1_B

x0=[
0.2893, -0.2873,  0.0404,  0.0476, -0.2193,  0.2644, -0.0524,  0.2096,
-0.2123,  0.1015, -0.1486, -0.0139,  0.0007,  0.0810, -0.0760, -0.2986,
-0.4043,  0.1651,  0.0472,  0.2808,  0.0555,  0.1248, -0.2412, -0.0129
]

for i,label in enumerate(np.argmax(labels[indices],axis=1)):
    #res1_A,res1_B
    print(i,label)
    if(label==0):
        #tmp_res = batch_A[i]*deltax + batch_b[i]
        #tmp_res[0]>=tmp_res[1]:

        #约束需要满足的形式
        batch_A_sub=[batch_A[i][0][j]-batch_A[i][1][j]  for j in range(len(batch_A[i][0]))]
        batch_A_sub=np.array(batch_A_sub)
        batch_A_sub=batch_A_sub[neuron_indice]

        constraint_A.append(list(batch_A_sub))
        temp_b=sum([ batch_A_sub[j]*x0[j] for j in range(len(batch_A_sub))])-(batch_b[i][0]-batch_b[i][1])
        constraint_b.append(temp_b)
    elif (label==1):
        #tmp_res = batch_A[i]*deltax + batch_b[i]
        # tmp_res[1]>=tmp_res[0]:

        batch_A_sub=[batch_A[i][1][j]-batch_A[i][0][j]  for j in range(len(batch_A[i][0]))]
        batch_A_sub=np.array(batch_A_sub)
        batch_A_sub=batch_A_sub[neuron_indice]
        constraint_A.append(list(batch_A_sub))
        temp_b=sum([ batch_A_sub[j]*x0[j] for j in range(len(batch_A_sub))])-(batch_b[i][1]-batch_b[i][0])
        constraint_b.append(temp_b)

    else:
        print("产生错误！",label)



print("constraint_A")
print(len(constraint_A),len(constraint_A[0]))
print(constraint_A)
print("constraint_b")
print(len(constraint_b))
print(constraint_b)

print("下面开始验证约束优化结果正确性")
Q1=np.loadtxt('quadratic_para/Q124_c_2.txt' )
b1=np.loadtxt('quadratic_para/b124_c_2.txt')
c1=np.loadtxt('quadratic_para/c124_c_2.txt')
print("Q1:")
print(Q1)
print("b1:")
print(b1)
print("c1:")
print(c1)

print(len(Q1),len(b1))


if (fit_optimize.check_SPD(Q1)):
    print("拟合出的原始矩阵正定")
else:
    print("原始矩阵非正定")
    Q1 = np.array(Q1)
    eigen = np.linalg.eig(Q1)
    min_eigenvalue = min(eigen[0])
    # 将矩阵转化为正定矩阵
    Q1 = Q1 - 1.001 * min_eigenvalue * np.eye(len(Q1))
    print(fit_optimize.check_SPD(Q1))

#https://blog.csdn.net/u013421629/article/details/108358409
constraint_b=[[i] for i in constraint_b]
# print(constraint_A)
# print(constraint_b)
constraint_A=np.array(constraint_A)
constraint_b=np.array(constraint_b)
constraint_A=np.float64(constraint_A)
constraint_b=np.float64(constraint_b)

res = fit_optimize.quadprog(  Q1, b1,constraint_A,constraint_b)
print(res)

model = torch.load("data/census.pt")

count=0
for i in params_index:
    layer_index, neuron_index = i
    neuron_index -= 1
    key = f'fc{layer_index + 1}.weight'
    tmp_matrix = model[key].T
    for j in range(0, len(tmp_matrix[neuron_index])):
        tmp_matrix[neuron_index][j] = res[count][0]
        count += 1
    model[key] = tmp_matrix.T

# 这里要改动
optimized_net='constrainted1_optimizednet24_c_2.pt'
torch.save(model, optimized_net)
print(count)

sum = 0
for j in range(0, 10):
    fairness = fit_optimize.cal_fairness1(model.copy())
    print(fairness)
    sum += fairness
sum = sum / 10
print("优化后的网络公平性:", sum)



model=data_process.CensusNet(14)
optimized_acc=data_process.recal_acc(model, optimized_net)
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










