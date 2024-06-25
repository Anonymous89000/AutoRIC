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

####
disturbs = disturbs_2


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



if  __name__=='__main__':


    a=0

