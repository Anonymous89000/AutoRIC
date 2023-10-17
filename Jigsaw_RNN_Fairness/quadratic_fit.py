# fit property for rnns.
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import csv
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F
import util.quadratic_fitting
import util.get_fit_data
import util.fit_optimize
import util.data_process
import time
import autograd.numpy as np
import argparse
import json
import ast
from cal_fairness import add_assertion, add_solver, cal_fairness_with_model, eval_acc
from json_parser import parse

# TODO: add variables to be fit

# n*20
params_index_lstm2_2 = [
    (2, 8), (2, 12), (2, 20), (2, 25), (2, 36),
]

params_index_lstm3_2 = [
    (3, 8), (3, 12), (3, 20), (3, 25), (3, 36),
]

# n*20
params_index_lstm2 = [
    (2, 1), (2, 2), (2, 7), (2, 11), (2, 18), (2, 24), (2, 33),
]
# n*20
params_index_lstm3 = [
    (3, 7), (3, 11), (3,15), (3,19), (3,25), (3,28), (3,35)
]
# n*2
params_index_linear = [
    (4, 1),
    (4, 2)
]


disturbs_0 = [ -0.001,
              0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1, -0.5, 0.5, -1, 1, -5, 5]



def quadratic_fit(model, assertion, solver, n, net_type, params_index, disturbs):
    '''
    net:model
    n:待修复参数的个数
    net_type:为修复后的网络命名（根据）

    '''

    fit_data_filename = 'fit_data/'+net_type+'_paradata.csv'

    quadratic_Q1 = 'quadratic_para/'+net_type+'_Q1.txt'
    quadratic_b1 = 'quadratic_para/'+net_type+'_b1.txt'
    quadratic_c1 = 'quadratic_para/'+net_type+'_c1.txt'

    # 此处应添加清除文件原本内容的代码

    f = open(fit_data_filename, 'a+')
    
    filewriter = csv.writer(f)
    # 写入原始公平性
    p = util.get_fit_data.getpapra2_with_lib_model(model.copy(), 0.06, params_index)

    header = ["p" + str(int(i)) for i in range(1, n + 1)] + ['y']
    filewriter.writerow(header)
    filewriter.writerow(p)

    #(60, 40), (20, 40), (20, 40), (10, 2)
    # 计算得到拟合数据fit_data
    for i, delta in enumerate(disturbs):
        tmp_model = util.get_fit_data.modify_lib_model_param(
            model.copy(), params_index, delta)
        fainess_tmp = cal_fairness_with_model(tmp_model,'jigsaw', assertion, solver)
        eval_acc(model, 'jigsaw', assertion)
        print("扰动为", delta, "的公平性", fainess_tmp)
        p = util.get_fit_data.getpapra2_with_lib_model(tmp_model, fainess_tmp, params_index)
        print(type(model.shape))
        filewriter = csv.writer(f)
        filewriter.writerow(p)
    f.close()

    # 以下为根据生成的数据 来对公平性与参数关系进行拟合的过程
    # fit_data -> quadratic_para
    fit_data_filename = fit_data_filename
    Q1, b1, c1 = util.quadratic_fitting.fitting(n, fit_data_filename)
    print("通过拟合得到的二次型参数如下")
    print('Q1:\n', (0.5) * Q1, '\nb1:\n', b1, '\nc:\n', c1)
    # 这里要注意二次型定义中的二分之一 不要多乘或者漏掉了二分之一

    np.savetxt(quadratic_Q1, (0.5)*Q1)
    np.savetxt(quadratic_b1, b1)
    np.savetxt(quadratic_c1, [c1])


if __name__ == "__main__":
    np.set_printoptions(threshold=20)
    parser = argparse.ArgumentParser(description='nSolver')
    parser.add_argument('--spec', type=str, default='benchmark/rnn/nnet/jigsaw_lstm/spec.json',
                        help='the specification file')
    parser.add_argument('--algorithm', type=str, default='optimize',
                        help='the chosen algorithm')
    parser.add_argument('--threshold', type=float,
                        help='the threshold in sprt')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='the distance value')
    parser.add_argument('--dataset', type=str, default='jigsaw',
                        help='the data set for rnn experiments')

    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    add_assertion(args, spec)
    add_solver(args, spec)

    model, assertion, solver, display = parse(spec)

    net_type = "rnn_lstm3"
    import time
    start = time.time()
    quadratic_fit(model, assertion, solver, 140, net_type, params_index_lstm3, disturbs_0)
    end = time.time()
    print(f"{end - start:.2f}")
