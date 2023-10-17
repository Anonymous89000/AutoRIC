# 拟合
import numpy as np
import torch
import csv
import torch
import util.quadratic_fitting
import util.get_fit_data
import util.fit_optimize
from deepZ import Conv
from verify import binarySearch

disturbs_0 = [0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]
disturbs_1 = [-0.1, -0.2, 0.2, 0.1]
disturbs_2 = [-0.08+0.001*i for i in range(161)]
disturbs_check = [0.0,0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01]

DEVICE = 'cuda'
def quadratic_fit(net:str, n:int, net_type:str, params_index, disturbs, mrr, inputs, true_label, l, r):
    """
    net:保存网络参数的文件名
    n0:待修复参数的个数
    net_type:为修复后的网络命名
    params_index:待修复的参数tuples
    """
    model = torch.load(net)
    # 初始化保存中间数据的文件名

    fit_data_filename='fit_data/'+net_type+'_paradata.csv'
    optimized_net='result/'+net_type+'_optimizednet.pt'
    # record_filename='result/'+net_type+'_optimizednet.csv'

    quadratic_Q1='quadratic_para/'+net_type+'_Q1.txt'
    quadratic_b1='quadratic_para/'+net_type+'_b1.txt'
    quadratic_c1='quadratic_para/'+net_type+'_c1.txt'

    # TODO:此处应添加清除文件原本内容的代码

    print(f"网络优化类型: {optimized_net}")


    p =util.get_fit_data.getpara2(model.copy(), mrr, params_index)

    # write init data to the file.
    f = open(fit_data_filename, 'a+')
    filewriter = csv.writer(f)
    header = ["p" + str(int(i)) for i in range(1, n + 1)] + ['y']
    filewriter.writerow(header)
    filewriter.writerow(p)

    #计算得到拟合数据fit_data
    tmp_model_ = Conv(DEVICE, 28, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    for _, delta in enumerate(disturbs):
        tmp_model = util.get_fit_data.modify(model.copy(), params_index, delta)

        # get robustness
        tmp_model_.load_state_dict(tmp_model)
        mrr = binarySearch(tmp_model_, inputs, true_label, l, r)
        target = 1/mrr
        print(target)
        print(f"扰动为 {delta}, 鲁棒性 {mrr}")
        
        p = util.get_fit_data.getpara2(tmp_model, target, params_index)
        filewriter = csv.writer(f)
        filewriter.writerow(p)
    f.close()

    #以下为根据生成的数据 来对公平性与参数关系进行拟合的过程
    #fit_data -> quadratic_para
    Q1, b1, c1 = util.quadratic_fitting.fitting(n, fit_data_filename)
    #print("通过拟合得到的二次型参数如下")
    #print('Q1:\n', (0.5) * Q1, '\nb1:\n', b1, '\nc:\n', c1)
    #这里要注意二次型定义中的二分之一 不要多乘或者漏掉了二分之一

    np.savetxt(quadratic_Q1, (0.5)*Q1)
    np.savetxt(quadratic_b1, b1)
    np.savetxt(quadratic_c1, [c1])

if __name__=="__main__":
    """
    net = 'model.pth'
    net_type = 'test_mnist'
    quadratic_fit(net, 200, net_type, params_index_fc1, disturbs_2, 0.8)
    """