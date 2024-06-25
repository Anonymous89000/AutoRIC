import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision
import csv
import numpy as np
import  csv
import cvxopt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import quadratic_fitting
import cvxopt
import torch
from torch import nn
import get_fit_data
import  fit_optimize
import  data_process
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
disturbs_1=[-0.0001,0.0005]
disturbs_2=[-0.08+0.001*i for i in range(161)]
disturbs_check=[0.0,0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]

####
#扰动数据的构造目前只有一个方向上 比如所有都加减相同的数值 更好的方式应该是取一个网格状的扰动
#目前更紧迫的是解决准确性下降的问题
# 1线性化 先对单层进行实验
# 2 二分法 调整结果参数

disturbs = disturbs_0


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

num_type_list=[(222,'222_o_2'),(128,'128_a_2'),(64,'64_b_2'),(24,'24_c_2')]

def total_function(n0,net_type0,params_index):

    time_start=time.time()
    #以下为通过采样计算公平性 生成拟合数据的过程
    model=torch.load('data/census.pt')
    n = n0
    net_type=net_type0
    fit_data_filename='fit_data/paradata'+net_type+'.csv'
    optimized_net='result/optimizednet'+net_type+'.pt'
    record_filename='result/optimizednet'+net_type+'.csv'

    quadratic_Q1='quadratic_para/Q1'+net_type+'.txt'
    quadratic_b1='quadratic_para/b1'+net_type+'.txt'
    quadratic_c1='quadratic_para/c1'+net_type+'.txt'

    print("网络优化类型:",optimized_net)
    f = open(fit_data_filename, 'a+')

    filewriter = csv.writer(f)
    p = get_fit_data.getpara2(model.copy(), 0.142)
    header = ["p" + str(int(i)) for i in range(1, n + 1)] + ['y']
    filewriter.writerow(header)
    filewriter.writerow(p)

    #计算得到拟合数据
    for i,delta in enumerate(disturbs):
        #print(i,delta)
        #filename = str(delta)+'.pt'
        tmp_model =get_fit_data.modify0(model.copy(),params_index,delta)
        sum=0
        for j in range(10):
            fainess_tmp=get_fit_data.cal_fairness1(tmp_model)
            sum+=fainess_tmp
        sum=sum/10
        print("扰动为",delta,"的公平性",sum)
        p=get_fit_data.getpara2(tmp_model,sum)
        filewriter = csv.writer(f)
        filewriter.writerow(p)
    f.close()



    #以下为根据生成的数据 来对公平性与参数关系进行拟合的过程
    fit_data_filename = fit_data_filename
    Q1, b1, c1 = quadratic_fitting.fitting(n, fit_data_filename)
    print("通过拟合得到的二次型参数如下")
    print('Q1:\n', (0.5) * Q1, '\nb1:\n', b1, '\nc:\n', c1)

    np.savetxt(quadratic_Q1,(0.5)*Q1)
    np.savetxt(quadratic_b1,b1)
    np.savetxt(quadratic_c1,[c1])

    # print(type(Q1.T))
    # 以上得到了拟合的二次型  注意二倍关系!
    # print(check_SPD([[1,0],[0,1]]))

    # print(check_SPD(Q1))
    # res=quadprog(np.array([[1,0],[0,-1]]),np.array([2,4]))
    # print(res)
    

    if (fit_optimize.check_SPD(Q1)):
        print("拟合出的原始矩阵正定")
    else:
        print("原始矩阵非正定")
        Q1 = np.array(Q1)
        eigen = np.linalg.eig(Q1)
        min_eigenvalue = min(eigen[0])
        #将矩阵转化为正定矩阵
        Q1 = Q1 - 1.001 * min_eigenvalue * np.eye(len(Q1))
        print(fit_optimize.check_SPD(Q1))


    #此处需要添加约束的if语句

    res = fit_optimize.quadprog((0.5) * Q1, b1)
    print("二次拟合后的结果", res)

    # filename = "solution1.csv"
    # f = open(filename, 'w')
    # header = ["x" + str(int(i)) for i in range(1, n + 1)]
    # filewriter = csv.writer(f)
    # filewriter.writerow(header)
    # res=[i[0] for i in res]
    # filewriter.writerow(res)

    # res  是得到的结果  现在把它装进网络里面
    model = torch.load("data/census.pt")
    sum = 0
    for j in range(0, 10):
        fairness = fit_optimize.cal_fairness1(model.copy())
        print(fairness)
        sum += fairness
    sum = sum / 10
    print("原始网络的公平性：", sum)
    raw_fairness=sum
    count = 0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index + 1}.weight'
        tmp_matrix = model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            tmp_matrix[neuron_index][j] = res[count][0]
            count += 1
        model[key] = tmp_matrix.T

    #这里要改动
    torch.save(model, optimized_net)
    print(count)

    sum = 0
    for j in range(0, 10):
        fairness = fit_optimize.cal_fairness1(model.copy())
        print(fairness)
        sum += fairness
    sum = sum / 10
    print("优化后的网络公平性:", sum)
    optimized_fairness=sum

    #以下为计算最终优化后的神经网络的准确性的过程
    model = data_process.CensusNet(14)
    raw_acc=data_process.recal_acc(model,'data/census.pt')
    model=data_process.CensusNet(14)
    optimized_acc=data_process.recal_acc(model, optimized_net)

    #将结果进行记录
    f = open(record_filename,'w')
    filewriter = csv.writer(f)
    header = ['raw_net','optimized_net','optimized_para_count','raw_fairness','optimized_fairness','raw_acc','optimized_acc']
    filewriter.writerow(header)
    res_list=['census.pt']
    res_list.append(optimized_net)
    res_list.append(n)
    res_list.append(raw_fairness)
    res_list.append(optimized_fairness)
    res_list.append(raw_acc)
    res_list.append(optimized_acc)
    filewriter.writerow(res_list)
    f.close()

    #输出结果
    print(f"原网络公平性:{raw_fairness},优化后公平性:{optimized_fairness}\n原网络准确性:{raw_acc},优化后准确性:{optimized_acc}\n")

    time_end=time.time()
    time_cost=time_end-time_start
    print("耗时：",time_cost,"s")

def  check():


    model = torch.load("data/census.pt")
    tmp_model = get_fit_data.modify0(model.copy(), params_index_c, 0)
    torch.save(tmp_model,'temp/tmp_model.pt')
    r_model = data_process.CensusNet(14)
    print(data_process.recal_acc(r_model,'data/census.pt'))
    print(data_process.recal_acc(r_model,'temp/tmp_model.pt'))

    sum =0
    for j in range(10):
        fainess_tmp = get_fit_data.cal_fairness1(model)
        sum+=fainess_tmp
        print(fainess_tmp)
    sum=sum/10
    print("原网络公平性:",sum)


    sum = 0
    for j in range(10):
        fainess_tmp = get_fit_data.cal_fairness1(tmp_model)
        sum += fainess_tmp
        print(fainess_tmp)
    sum = sum / 10
    print("扰动为", 0, "的公平性", sum)



    return 0
if __name__=="__main__":


    total_function(222,'222_o_0',params_index_o)




    # for i in range(len(num_type_list)):
    #     n0,net_type0=num_type_list[i]
    #     total_function(n0,net_type0,params_index_all[i])















