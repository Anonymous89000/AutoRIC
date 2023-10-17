#将均匀采样后的数据读取并零一编码，然后放到训练好的网络得到输出，计算公平性

# 可视化
import matplotlib.pyplot as plt  # 绘图
import matplotlib
import time
import datetime
import pandas as pd
import numpy as np
# import missingno  # 缺失值可视化
# import seaborn as sns  # 绘图库
from pandas.plotting import scatter_matrix #绘制散布矩阵图
from sklearn.model_selection import StratifiedShuffleSplit #分层交叉验证
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
# import gurobipy as gp
# from gurobipy import GRB
import math
import ast
# from FFNN import CensusNet

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
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'None']
rel = ['Wife', 'Own-child', 'Husband',
       'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
nc = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
      'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
      'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
sex = ['Male','Female']
ic = ['<=50K','>50K']

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

# 加载已有网络并计算二分类数量
def test(model,test_x,device):
    tensor_test_x = torch.FloatTensor(test_x.copy()) # transform to torch tensor 
    test_dataset = TensorDataset(tensor_test_x) # create dataset                                                     
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle = True) # create dataloader
    # print(type(test_dataloader))
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # print(size)
    # print(num_batches)
    test_loss, correct = 0, 0
    pos = 0
    neg = 0
    gt50 = torch.tensor([[1,0]])
    model.eval()
    with torch.no_grad():
        for x in test_dataloader:
            # x = x.to(device)
            # print(x)
            # print(type(x[0]))
            pred = model(x[0])
            pred = pred.type(torch.FloatTensor)
            # print(pred.shape)
            dim0,dim1 = pred.shape
            for i in range(dim0):
                element = pred[0,:]
                element = element.unsqueeze(0)
                # print(element.shape)
                # print(gt50.shape)
                if(element.argmax(1)==gt50.argmax(1)):
                    pos = pos + 1
                else:
                    neg = neg + 1
            # pred_1 = (pred.argmax(1)).type(torch.float).sum().item()
            # pred_0 = (pred.argmax(0)).type(torch.float).sum().item()
    postive = pos / size
    negtive = neg /size
    return postive, negtive
    

def calculate_gen_r1_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r1.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r1_feature.txt", feature)

def calculate_gen_r2_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r2.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r2_feature.txt", feature)

def calculate_gen_r3_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r3.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r3_feature.txt", feature)

def calculate_gen_r4_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r4.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r4_feature.txt", feature)

def calculate_gen_r5_acc():
    feature = []
    # income = []
    with open("gender_fairness/sample_race/gen_r5.csv", "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(';')
            features = features[0].split(',')
            # print(features[1])
            features[0] = np.clip(int(features[1]) / 10, 1, 9)#age
            features[1] = wc.index(features[2])#workclass
            features[2] = np.clip(int(features[3]) / 10000, 1, 148)#fnlwgt
            features[3] = edu.index(features[4])#education
            features[4] = np.clip(int(features[5]), 1, 16)#education.num
            features[5] = ms.index(features[6])#marital.status
            features[6] = occu.index(features[7])#occupation
            features[7] = rel.index(features[8])#relationship
            features[8] = race.index(features[9])#race
            features[9] = sex.index(features[10])#sex
            features[10] = np.clip(int(features[11]) / 100, 0, 41)#capital.gain
            features[11] = np.clip(int(features[12]) / 100, 0, 43)#capital.loss
            features[12] = np.clip(int(features[13]) / 10, 1, 9)#hours.per.week
            features[13] = nc.index(features[14])#native.country
            feature.append(features[:14])
         
    feature = np.asarray(feature)
    # income = np.asarray(income)
    np.savetxt("gender_fairness/sample_race/r5_feature.txt", feature)

def split_test():
    # np.savetxt('data/col.txt', adult.columns)
    # adult["new_income_1，new_income_0"].to_csv('data/income.csv')
    # print(adult.columns)
    adult = pd.read_csv(r'data/adult_13fes.csv')

    object_columns = []
    int_columns = []
    for i in adult.columns:
        if adult.loc[:,i].dtype == "object":
            object_columns.append(i)
        if adult.loc[:,i].dtype == "int64":
            int_columns.append(i)

    for col_name in object_columns:
        values = np.array(adult[col_name])
        # print(values)
        onehot_encoder = OneHotEncoder(sparse=False)
        values = values.reshape(len(values), 1)
        onehot_matrix = onehot_encoder.fit_transform(values)
        # print(onehot_matrix)
        adult.drop([col_name],axis=1,inplace=True)
        # 在 Dataframe 首列插入one-hot后新列
        for i in range(onehot_matrix.shape[1]):
            adult.insert(0, 'new_'+col_name+"_"+str(i), value=onehot_matrix[:,i])
    # 当然，这也可以批量操作（前提是对dataframe数据） 尝试一下：onehot_encoder.fit_transform(adult[object_columns])

    for col_name in int_columns:
        Scaler = MinMaxScaler(feature_range=(-1, 1))
        col_value = np.array(adult[col_name]).reshape(-1,1)
        new_col = Scaler.fit_transform(col_value)
        adult[col_name] = new_col

    # 将数据按照gender值分成两部分
    # new_gender_1,new_gender_0
    grouped = adult.groupby(['new_gender_1','new_gender_0'])

    # 按组返回两部分数据
    male = grouped.get_group((1,0))
    female = grouped.get_group((0,1))

    # 将输入特征和目标分开
    income_columns = ['new_income_1','new_income_0']
    # adult[income_columns].to_csv('data/income.txt', sep = '\t', index = False)
    income_y_m =male[income_columns]
    income_y_fm =female[income_columns]
    male = male.drop(columns=income_columns)
    female = female.drop(columns=income_columns)

    # 转成numpy格式
    male = male.values
    print(male.shape)
    np.savetxt("data3/male.txt",male)
    female = female.values
    print(female.shape)
    np.savetxt("data3/female.txt",female)
    income_y_m = income_y_m.values
    print(income_y_m.shape)
    np.savetxt("data3/income_m.txt",income_y_m)
    income_y_fm = income_y_fm.values
    print(income_y_fm.shape)
    np.savetxt("data3/income_fm.txt",income_y_fm)

    # 划分数据集
    split = StratifiedShuffleSplit(n_splits=2,train_size=0.75)
    for train_index,test_index in split.split(male,income_y_m):
        trainx_m,testx_m = male[train_index],male[test_index]
        trainy_m,testy_m = income_y_m[train_index],income_y_m[test_index]



    # print(trainy_m.shape)
    print(testx_m.shape)
    np.savetxt("data3/trainx_m.txt" , trainx_m)
    np.savetxt("data3/trainy_m.txt" , trainy_m)
    np.savetxt("data3/testx_m.txt" , testx_m)
    np.savetxt("data3/testy_m.txt" , testy_m)

    split = StratifiedShuffleSplit(n_splits=2,train_size=0.75)
    for train_index,test_index in split.split(female,income_y_fm):
        trainx_fm,testx_fm = female[train_index],female[test_index]
        trainy_fm,testy_fm = income_y_fm[train_index],income_y_fm[test_index]



    # print(trainy_fm.shape)
    print(testx_fm.shape)
    np.savetxt("data3/trainx_fm.txt" , trainx_fm)
    np.savetxt("data3/trainy_fm.txt" , trainy_fm)
    np.savetxt("data3/testx_fm.txt" , testx_fm)
    np.savetxt("data3/testy_fm.txt" , testy_fm)

if __name__ == "__main__":
    # male = pd.read_csv(r'gender_fairness/sample/male_gen.csv',)
    # female = pd.read_csv(r'gender_fairness/sample/female_gen.csv',)
    # male = male.drop(['Unnamed: 0'],axis = 1)
    # female = female.drop(['Unnamed: 0'],axis = 1)
    # male.to_csv('gender_fairness/sample/male_gen1.csv')
    # female.to_csv('gender_fairness/sample/female_gen1.csv')
    
    # calculate_male_acc()
    # calculate_gen_r1_acc()
    # calculate_gen_r2_acc()
    # calculate_gen_r3_acc()
    # calculate_gen_r4_acc()
    # split_test()
    # calculate_all_acc()

    model = CensusNet(14)
    device = 'cpu'
    PATH = "data4/census.pt"
    test_x = np.loadtxt('gender_fairness/sample_race/r4_feature.txt')# 删减三个特性后数据
    # test_x = np.loadtxt('data2/testx.txt')# 所有的不分男女的测试数据
    # print(test_x.shape)
    # test_x = np.loadtxt('data3/testx_fm.txt')#原测试集中male数据
    # test_x = np.loadtxt('gender_fairness/sample/all_feature_s3.txt')# 删减三个特性后数据
    model.load_state_dict(torch.load(PATH))
    # model = torch.load(PATH)
    # model.eval()
    positive,negtive = test(model, test_x, device)
    print(positive)
    print(negtive)



# np.savetxt('data/col.txt', adult.columns)
# adult["new_income_1，new_income_0"].to_csv('data/income.csv')
# print(adult.columns)

# # 将数据按照gender值分成两部分
# # new_gender_1,new_gender_0
# grouped = adult.groupby(['new_gender_1','new_gender_0'])

# # 按组返回两部分数据
# male = grouped.get_group((1,0))
# female = grouped.get_group((0,1))

# # 将输入特征和目标分开
# income_columns = ['new_income_1','new_income_0']
# # adult[income_columns].to_csv('data/income.txt', sep = '\t', index = False)
# income_y_m =male[income_columns]
# income_y_fm =female[income_columns]
# male = male.drop(columns=income_columns)
# female = female.drop(columns=income_columns)

# # 转成numpy格式
# male = male.values
# print(male.shape)
# np.savetxt("gender_fairness/male.txt",male)
# female = female.values
# print(female.shape)
# np.savetxt("gender_fairness/female.txt",female)
# income_y_m = income_y_m.values
# print(income_y_m.shape)
# np.savetxt("gender_fairness/income_m.txt",income_y_m)
# income_y_fm = income_y_fm.values
# print(income_y_fm.shape)
# np.savetxt("gender_fairness/income_fm.txt",income_y_fm)

# # 划分数据集
# split = StratifiedShuffleSplit(n_splits=2,train_size=0.75)
# for train_index,test_index in split.split(male,income_y_m):
#     trainx_m,testx_m = male[train_index],male[test_index]
#     trainy_m,testy_m = income_y_m[train_index],income_y_m[test_index]



# print(trainy_m.shape)
# print(trainx_m.shape)
# np.savetxt("gender_fairness/trainx_m.txt" , trainx_m)
# np.savetxt("gender_fairness/trainy_m.txt" , trainy_m)
# np.savetxt("gender_fairness/testx_m.txt" , testx_m)
# np.savetxt("gender_fairness/testy_m.txt" , testy_m)

# split = StratifiedShuffleSplit(n_splits=2,train_size=0.75)
# for train_index,test_index in split.split(female,income_y_fm):
#     trainx_fm,testx_fm = female[train_index],female[test_index]
#     trainy_fm,testy_fm = income_y_fm[train_index],income_y_fm[test_index]



# print(trainy_fm.shape)
# print(trainx_fm.shape)
# np.savetxt("trainx_fm.txt" , trainx_fm)
# np.savetxt("trainy_fm.txt" , trainy_fm)
# np.savetxt("testx_fm.txt" , testx_fm)
# np.savetxt("testy_fm.txt" , testy_fm)

