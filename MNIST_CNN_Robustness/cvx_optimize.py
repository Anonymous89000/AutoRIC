# 直接利用二次规划进行优化 结果保存在文件中

import torch
import numpy as np
import util
import util.fit_optimize
from deepZ import Conv
from mnist_net import test
import torchvision
from verify import binarySearch

#n*100 fc1
params_index_1=[
    (6, 1),
    (6, 24),
    (6, 43),
    (6, 65)
]

#n*100 fc2
params_index_2=[
    (8, 7),
    (8, 34),
    (8, 52),
    (8, 97)
]

#n*10 fc3
params_index_2=[
    (10, 1),
    (10, 3),
    (10, 5),
    (10, 7)
]
DEVICE = 'cuda'
DELTA = 10e-4
def cvx_optimize(netname, params_index, flag):
    """ flag==0时为无约束优化
    """

    filename_Q1 = 'quadratic_para/'+netname+'_Q1.txt'
    filename_b1 = 'quadratic_para/'+netname+'_b1.txt'
    filename_c1 = 'quadratic_para/'+netname+'_c1.txt'

    # 二次约束结果
    Q1 = np.loadtxt(filename_Q1)
    b1 = np.loadtxt(filename_b1)
    c1 = np.loadtxt(filename_c1)

    # 线性约束
    if flag:
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

    if flag:
        # https://blog.csdn.net/u013421629/article/details/108358409
        constraint_b = [[i] for i in constraint_b]
        # print(constraint_A)
        # print(constraint_b)
        constraint_A = np.array(constraint_A)
        constraint_b = np.array(constraint_b)
        constraint_A = np.float64(constraint_A)
        constraint_b = np.float64(constraint_b)

    if (flag == 0):
        print("无约束优化！")
        res = util.fit_optimize.quadprog(Q1, b1)
    else:
        print("有约束优化！")
        res = util.fit_optimize.quadprog(Q1, b1, constraint_A, constraint_b)

    model = torch.load("./DeepZ/mnist_nets/conv4.pt")

    count = 0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'layers.{layer_index}.weight'
        tmp_matrix = model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            tmp_matrix[neuron_index][j] = res[count][0]
            count += 1
        model[key] = tmp_matrix.T

    # 这里要改动
    optimized_net = 'result/' + netname + '_opt.pt'
    torch.save(model, optimized_net)


def dichotomy_opt(net1, net2, params_index, result_file_name, inputs, true_label, l, r):
    """ 二分优化
    """
    model1 = torch.load(net1)
    model2 = torch.load(net2)
    model= Conv('cuda', 28, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to('cuda')

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size = 1000, shuffle=True)

    model.load_state_dict(model1)
    acc1 = test(model, test_loader)
    model.load_state_dict(model2)
    acc2 = test(model, test_loader)

    model3=model1.copy()

    mc = 0
    x1 = []
    x2 = []
    acc3 = 0

    tmpfile = result_file_name
    for param in params_index:
        layer, pos = param[0], param[1] - 1
        key = f'layers.{layer}.weight'
        weight1=model1[key].T
        weight2=model2[key].T
        for i in weight1[pos]:
            x1.append(float(i))
        for i in weight2[pos]:
            x2.append(float(i))

    param_count = len(x1)
    print("len x1: ",len(x1),'\n',x1)
    print("len x2: ", len(x1),'\n', x2)
    x3 = x1.copy()
    while(mc < 10):
        #得到参数取值二分之后的模型
        for i in range(param_count):
            x3[i]=(x1[i]+x2[i])/2
        count=0
        for param in params_index:
            layer, pos = param[0], param[1] - 1
            key = f'layers.{layer}.weight'
            weight3 = model3[key].T
            for i in range(len(weight3[pos])):
                weight3[pos][i]=x3[count]
                count+=1
            model3[key]=weight3.T

        torch.save(model3, tmpfile)
        model.load_state_dict(model3)
        acc3 = test(model, test_loader)

        if acc1 > acc2:
            x2 = x3.copy()
            acc2 = acc3
        else:
            x1 = x3.copy()
            acc1 = acc3
        #计算新模型准确性
        mrr = binarySearch(model, inputs, true_label, l, r)
        print(mrr)
        print("mc:", mc,'acc:', acc3)
        mc = mc + 1
    return

def non_constraints_optimize(netname, origin, params_index, target:float, l ,r):
    import time
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])),
    batch_size = 1000, shuffle=True)

    spec = "./DeepZ/test_cases/conv4/img0_0.01240.txt"

    with open(spec, "r") as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]

    inputs = torch.FloatTensor(pixel_values).view(1, 1, 28, 28).to(DEVICE)
    start = time.time()
    # try to optmize first
    cvx_optimize(netname, params_index, 0)
    end1 = time.time()
    optimized_net = 'result/' + netname + '_opt.pt'
    network = Conv(DEVICE, 28, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    network.load_state_dict(torch.load(optimized_net))
    total = sum([p.numel() for p in network.parameters()])
    print(total)
    acc = test(network, test_loader)
    
    # acc engough
    if(acc > target): 
        return

    dichotomy_opt(origin, optimized_net, params_index_1, 'opt.pt', inputs, true_label, l, r)
    end2 = time.time()
    print(f"{end1- start:.2f}")
    print(f"{end2- end1:.2f}")
if __name__ == '__main__':
    origin = "./DeepZ/mnist_nets/conv4.pt"
    # opt = "result/mnist_opt.pt"
    target = 0.9
    l = DELTA
    r = 0.20
    non_constraints_optimize("mnist", origin, params_index_1, target, l, r)