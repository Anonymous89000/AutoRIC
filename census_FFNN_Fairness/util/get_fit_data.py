#给定待修改参数位置以及扰动 给出公平性取值
#输出为csv文件
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

params_index_a=[
    (1, 1),
    (1, 24),
    (1, 33),
    (1, 45)
]
params_index_b=[
    (2, 7),
    (2, 11),
    (2, 18),
    (2, 25)
]
params_index_c=[
    (3, 7),
    (3, 11),
    (3, 12)
]


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

params_index=params_index_o

disturbs_0=[0.0001, -0.0001, -0.0005, 0.0005, -0.001, 0.001, -0.005, 0.005, -0.01, 0.01, -0.05, 0.05, -0.1, 0.1]
fairness_list_0=[0.131,0.141,0.123,0.092,0.112,0.116,0.125,0.139,0.113,0.164,0.057,0.237,0.052,0.203]
disturbs_1=[-0.0001,0.0005]
disturbs_2=[]

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







def getpara(filename,fairness):
    file_name = filename
    fairness=fairness
    p=[]
    model = torch.load(filename)
    count = 0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index}.weight'
        for j in range(0, len(model[key][neuron_index])):
            print(key, layer_index, neuron_index, j)
            p.append(float(model[key][neuron_index][j]))
            count += 1
            print(count)
    p.append(fairness)
    return p.copy()

def getpara2(model0,fairness):
    fairness=fairness
    p=[]
    model=model0
    count = 0
    for i in params_index:
        layer_index, neuron_index = i
        neuron_index -= 1
        key = f'fc{layer_index+1}.weight'
        tmp_matrix=model[key].T
        for j in range(0, len(tmp_matrix[neuron_index])):
            #print(tmp_matrix.shape)
            p.append(float(tmp_matrix[neuron_index][j]))
            count+=1
        # for j in range(0, len(model[key][neuron_index])):
        #     print(key, layer_index, neuron_index, j)
        #     p.append(float(model[key][neuron_index][j]))
        #     count += 1
        #     print(count)
    #print(count)
    p.append(fairness)
    return p.copy()


def modify0(model, params, disturb):
    #count=0
    for  param in params:
        layer, pos = param[0], param[1] - 1
        key = f'fc{layer+1}.weight'

        weight = model[key].T
        weight[pos] += disturb
        #count+=weight.shape[1]
        model[key] = weight.T
    #print(count)
    return  model.copy()


#加载已有网络并计算二分类数量
def rtest(model, test_x, device):
    tensor_test_x = torch.FloatTensor(test_x.copy())  # transform to torch tensor
    test_dataset = TensorDataset(tensor_test_x)  # create dataset
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)  # create dataloader
    # print(type(test_dataloader))
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # print(size)
    # print(num_batches)
    test_loss, correct = 0, 0
    pos = 0
    neg = 0
    gt50 = torch.tensor([[1, 0]])
    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            # x = x.to(device)
            # print(x)
            # print(type(x[0]))
            pred = model(x[0])
            pred = pred.type(torch.FloatTensor)
            # print(pred.shape)
            dim0, dim1 = pred.shape
            for i in range(dim0):
                element = pred[0, :]
                element = element.unsqueeze(0)
                # print(element.shape)
                # print(gt50.shape)
                if (element.argmax(1) == gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size
    negtive = neg / size
    return postive, negtive

def cal_fairness1(model,device='cpu'):
    time_start = time.time()
    # 修改网络


    # 修改输入文件
    test_x_female = np.loadtxt('data/female_feature.txt')
    test_x_male = np.loadtxt('data/male_feature.txt')

    model1=CensusNet(14)
    model1.load_state_dict(model)
    positive_male = 0
    positive_female = 0
    negtive_male = 0
    negtive_female = 0
    n = 10
    # 测试n次得到二分类的平均数
    for i in range(n):
        # 返回二分类的比例，pos表示>=50K$分类的比例，neg表示<50K$分类的比例，pos+neg=1
        pos_male, neg_male = rtest(model1, test_x_male, device)
        positive_male = positive_male + pos_male
        negtive_male = negtive_male + neg_male

        pos_female, neg_female = rtest(model1, test_x_female, device)
        positive_female = positive_female + pos_female
        negtive_female = negtive_female + neg_female
    fairness = abs(positive_female / n - positive_male / n)
    time_end = time.time()
    # 由于取了平均，时间对应除以n
    time_sum = (time_end - time_start) / n

    # print("男性每年收入大于等于50K$的概率：%f , 网络参数扰动为：%f" % (positive_male / n, 1))  # 对应性别的>=50K$分类的平均比例
    # print("男性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_male / n, 1))  # 对应性别的<50K$分类的平均比例
    # print("女性每年收入大于等于50K$的概率：%f, 网络参数扰动为：%f" % (positive_female / n, 1))  # 对应性别的>=50K$分类的平均比例
    # print("女性每年收入小于50K$的概率：%f, 网络参数扰动为：%f" % (negtive_female / n, 1))  # 对应性别的<50K$分类的平均比例
    # print("网络参数扰动为%f下，公平性取值为：%f, 所用时间为：%fs\n" % (1, fairness, time_sum))
    #print("一次公平性计算完成")
    return  fairness

if __name__=='__main__':


    model=torch.load('data/census.pt')
    f = open('fit_data/paradata222_1.csv', 'a+')
    filewriter = csv.writer(f)

    p = getpara2(model.copy(), 0.142)
    header = ["p" + str(int(i)) for i in range(1, 222 + 1)] + ['y']
    filewriter.writerow(header)
    filewriter.writerow(p)

    for i,delta in enumerate(disturbs):

        #print(i,delta)
        #filename = str(delta)+'.pt'
        tmp_model =modify0(model.copy(),params_index,delta)
        sum=0
        for j in range(10):
            fainess_tmp=cal_fairness1(tmp_model)
            sum+=fainess_tmp
        sum=sum/10

        print("扰动为",delta,"的公平性",sum)
        p=getpara2(tmp_model,sum)

        filewriter = csv.writer(f)
        filewriter.writerow(p)

    f.close()

















