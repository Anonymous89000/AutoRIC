#直接利用二次规划进行优化 结果保存在文件中

import sys
import torch
from torch import  nn
import torch.nn.functional as F
import numpy as np
import util
import util.data_process
import util.fit_optimize
from cal_fairness import cal_fairness_with_model, eval_acc

def cvx_optimize(netname, model ,assertion, solver , params_index, flag):
    """ flag==0时为无约束优化
    """
    filename_Q1='quadratic_para/'+netname+'_Q1.txt'
    filename_b1='quadratic_para/'+netname+'_b1.txt'
    filename_c1='quadratic_para/'+netname+'_c1.txt'

    # 二次约束结果
    Q1=np.loadtxt(filename_Q1 )
    b1=np.loadtxt(filename_b1)
    c1=np.loadtxt(filename_c1)

    # 线性约束
    if(flag):
        filename_A = 'constraint/'+netname+'_A.txt'
        filename_b = "constraint/"+netname+'_b.txt'
        constraint_A = np.loadtxt(filename_A)
        constraint_b = np.loadtxt(filename_b)

    if (util.fit_optimize.check_SPD(Q1)):
        print("拟合出的原始矩阵正定")
    else:
        print("原始矩阵非正定")
        Q1 = np.array(Q1)
        eigen = np.linalg.eig(Q1)
        min_eigenvalue = min(eigen[0])
        # 将矩阵转化为正定矩阵
        Q1 = Q1 - 1.001 * min_eigenvalue * np.eye(len(Q1))
        print(util.fit_optimize.check_SPD(Q1))
        
    if(flag):
        #https://blog.csdn.net/u013421629/article/details/108358409
        constraint_b=[[i] for i in constraint_b]
        # print(constraint_A)
        # print(constraint_b)
        constraint_A=np.array(constraint_A)
        constraint_b=np.array(constraint_b)
        constraint_A=np.float64(constraint_A)
        constraint_b=np.float64(constraint_b)

    if(flag==0):
        print("无约束优化！")
        res=util.fit_optimize.quadprog(Q1, b1)
    else:
        print("有约束优化！")
        res = util.fit_optimize.quadprog(Q1, b1, constraint_A, constraint_b)

    count=0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        layer = model.layers[layer_index-1]
        tmp_matrix = layer.get_weights().T
        for j in range(0, len(tmp_matrix[neuron_index])):
            tmp_matrix[neuron_index][j] = res[count][0]
            count += 1
        model.layers[layer_index-1].weights = tmp_matrix.T

    # 这里要改动
    optimized_net='result/'+ netname + '_opt.pt'
    torch.save(model, optimized_net)
    print(count)
    fairness =  cal_fairness_with_model(optimized_net, 'jigsaw', assertion, solver)
    print("优化后的网络公平性:", fairness)


    optimized_acc = eval_acc(optimized_net, 'jigsaw', assertion)
    print("优化后的准确性：",optimized_acc)
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
    cvx_optimize("testnet1",params_index_c,1)