# 二次拟合部分：准备拟合所需的公平性数据 拟合得到二次型参数 保存在文件中

import numpy as np
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
import time
import util.quadratic_fitting
import util.get_fit_data
import util.fit_optimize
import util.data_process
import time


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

####
#params_index=params_index_c

params_index_all=[params_index_o,params_index_a,params_index_b,params_index_c]

disturbs_0=[0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]
fairness_list_0=[0.131,0.141,0.123,0.092,0.112,0.116,0.125,0.139,0.113,0.164,0.057,0.237,0.052,0.203]
disturbs_1=[-0.0001, 0.0005]
disturbs_2=[-0.08+0.001*i for i in range(161)]
disturbs_check=[0.0,0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]

####
#扰动数据的构造目前只有一个方向上 比如所有都加减相同的数值 更好的方式应该是取一个网格状的扰动
#目前更紧迫的是解决准确性下降的问题
# 1线性化 先对单层进行实验
# 2 二分法 调整结果参数

#disturbs = disturbs_0

# 定义数据表头即参数名
headers = ['age', 'workclass', 'fnlwgt',
           'education', 'education.num',
           'marital.status', 'occupation',
           'relationship', 'race', 'sex',
           'capital.gain', 'capital.loss',
           'hours.per.week', 'native.country',
           'income']

# age,workclass,fnlwgt,education,education.num,marital.status,occupation,
# relationship,race,sex,capital.gain,capital.loss,hours.per.week,native.country,income
# 列举变量取值
wc = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
      'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
edu = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
       'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th',
       '10th', 'Doctorate', '5th-6th', 'Preschool']
ms = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
      'Married-spouse-absent', 'Married-AF-spouse']
# 将neverworked对应的occupation从？变成none
occu = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'none']
rel = ['Wife', 'Own-child', 'Husband',
       'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
nc = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
      'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
      'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
sex = ['Male', 'Female']
ic = ['<=50K', '>50K']



def quadratic_fit(net, n0, net_type0, params_index, disturbs):
    '''
    net:保存网络参数的文件名
    n0:待修复参数的个数
    net_type:为修复后的网络命名（根据）

    '''

    #以下为通过采样计算公平性 生成拟合数据的过程
    model=torch.load(net)
    n = n0
    net_type=net_type0
    fit_data_filename='fit_data/'+net_type+'_paradata.csv'
    optimized_net='result/'+net_type+'_optimizednet.pt'
    record_filename='result/'+net_type+'_optimizednet.csv'

    quadratic_Q1='quadratic_para/'+net_type+'_Q1.txt'
    quadratic_b1='quadratic_para/'+net_type+'_b1.txt'
    quadratic_c1='quadratic_para/'+net_type+'_c1.txt'

    #此处应添加清除文件原本内容的代码

    print("网络优化类型:",optimized_net)
    f = open(fit_data_filename, 'a+')

    filewriter = csv.writer(f)
    p =util.get_fit_data.getpara2(model.copy(), 0.142)
    header = ["p" + str(int(i)) for i in range(1, n + 1)] + ['y']
    filewriter.writerow(header)
    filewriter.writerow(p)

    #计算得到拟合数据fit_data
    for i,delta in enumerate(disturbs):
        #print(i,delta)
        #filename = str(delta)+'.pt'
        tmp_model =util.get_fit_data.modify0(model.copy(), params_index, delta)
        sum=0
        for j in range(10):
            fainess_tmp=util.get_fit_data.cal_fairness1(tmp_model)
            sum+=fainess_tmp
        sum=sum/10
        print("扰动为",delta,"的公平性",sum)
        p=util.get_fit_data.getpara2(tmp_model,sum)
        filewriter = csv.writer(f)
        filewriter.writerow(p)
    f.close()

    #以下为根据生成的数据 来对公平性与参数关系进行拟合的过程
    #fit_data -> quadratic_para
    fit_data_filename = fit_data_filename
    Q1, b1, c1 = util.quadratic_fitting.fitting(n, fit_data_filename)
    print("通过拟合得到的二次型参数如下")
    print('Q1:\n', (0.5) * Q1, '\nb1:\n', b1, '\nc:\n', c1)
    #这里要注意二次型定义中的二分之一 不要多乘或者漏掉了二分之一

    np.savetxt(quadratic_Q1, (0.5)*Q1)
    np.savetxt(quadratic_b1, b1)
    np.savetxt(quadratic_c1, [c1])

if __name__=="__main__":
    net = 'data/census.pt'
    nettype='testnet2'
    import time
    start = time.time()
    quadratic_fit(net,24,nettype,params_index_c,disturbs_1)
    end = time.time()
    print(f"{end - start:.2f}")