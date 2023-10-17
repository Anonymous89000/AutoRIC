import torch
import pandas as pd
from census_FFNN_training import CensusNet
import numpy as np
import random
import pickle
import sklearn

# try1
def try1():
    x = torch.randn(2, 3)
    y = x
    equal_tensor = torch.eq(x, y)
    correct = 0
    print(x)
    print(y)
    print(equal_tensor.shape)
    correct += (x.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    print(correct)
    # model = CensusNet()
# try2
# 删减了一个特性educational-num
def try2():
    df = pd.read_csv('data/adult.csv')
    df = df.drop(columns=['educational-num'])
    df.to_csv('data/adult_sim.csv', index = False)

# try3
def try3():
    model = CensusNet(14)
    # PATH = 'optimize/optimizednet1.pt'
    PATH = 'census_training/data/census.pt'
    model.load_state_dict(torch.load(PATH))
    params = model.state_dict()
    for name, param in params.items():
        print(f'{name}:{param.size()}')
        print(np.array(param))
# 依据两列数值将数据分成两部分

def try4():
    # 创建一个DataFrame对象
    df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                       'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                       'C': [1, 2, 3, 4, 5, 6, 7, 8],
                       'D': [10, 20, 30, 40, 50, 60, 70, 80]})

    # 根据 A 列和 B 列的值将数据分成两部分
    grouped = df.groupby(['A', 'B'])

    # 按组返回两部分数据
    part1 = grouped.get_group(('foo', 'one'))
    part2 = grouped.get_group(('bar', 'two'))
    part1.to_csv("part1.csv")

def try5():
    l1 = ['1','2','3']
    l2 = ['a','b','c']
    d1 = dict(zip(l1,l2))
    print(d1)

# 创建Dataframe
def try6():
    # 创建一个DataFrame  
    df = pd.DataFrame(columns=['Name', 'Age', 'City'])  
    df = df.append({'age': 83, 'workclass': 'Private', 'fnlwgt': 556763, 'education': '12th', 'marital-status': ' Separated', 'occupation': 'Craft-repair', 'relationship': 'Own-child', 'race': 'Other-relative', 'sex': 'Female', 'capital-gain': (0, 99999), 'capital-loss': 3549, 'hours-per-week': 15, 'native-country': 'South'}, ignore_index=True)
    print(df)

# try7
# 删减了3个特性，剩11个特性
def try7():
    df = pd.read_csv('data/adult.csv')
    df = df.drop(columns=['educational-num','capital-gain', 'capital-loss'])
    df.to_csv('data2/adult_sim_3.csv', index = False)

# 查看均匀采样生成男性和生成所有性别的数据的维度
def try10():
    df1 = pd.read_csv('gender_fairness/sample/all_feature_col.csv')
    df2 = pd.read_csv('gender_fairness/sample/all_feature_col.csv')

def try8():
    i = 1
    if(1<=i<=3):
        print("yes")

    fnlwgt_i = random.randint(1, 20)
    if(fnlwgt_i == 1):
      fnlwgt = random.randint(400000,1490400)
    if(1 < fnlwgt_i <= 3):
      fnlwgt = random.randint(300000,399999)
    if(3 < fnlwgt_i <= 9):
      fnlwgt = random.randint(200000,299999)
    if(9 < fnlwgt_i <= 17):
      fnlwgt = random.randint(100000,199999)
    if(17 < fnlwgt_i <= 20):
      fnlwgt = random.randint(13492,99999)
    print(fnlwgt)

def p_random(arr1,arr2):
    assert len(arr1) == len(arr2), "Length does not match."
    assert int(sum(arr2)) == 1 , "Total rate is not 1."

    sup_list = [len(str(i).split(".")[-1]) for i in arr2]
    top = 10 ** max(sup_list)
    new_rate = [int(i*top) for i in arr2]
    rate_arr = []
    for i in range(1,len(new_rate)+1):
        rate_arr.append(sum(new_rate[:i]))
    rand = random.randint(1,top)
    data = None
    for i in range(len(rate_arr)):
        if rand <= rate_arr[i]:
            data = arr1[i]
            break
    return data

def try9():
    wc = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
      'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
    # 离散型采样
    work_class_list = list(np.arange(0, 8))
    work_class_p = [0.7214,0.103,0.046,0.038,0.060,0.031,0.0004,0.0002]
    print(sum(work_class_p))
    work_class = p_random(work_class_list, work_class_p)
    work_class = wc[work_class]
    print(work_class)
    fnlwgt_i = [1,2,3,4,5]
    fnlwgt_p = [0.17,0.372,0.29,0.1,0.068]
    fnlwgt_i = p_random(fnlwgt_i, fnlwgt_p)
    if(fnlwgt_i == 1):
      fnlwgt = random.randint(400000,1490400)
    if(fnlwgt_i == 2):
      fnlwgt = random.randint(300000,399999)
    if(fnlwgt_i == 3):
      fnlwgt = random.randint(200000,299999)
    if(fnlwgt_i == 4):
      fnlwgt = random.randint(100000,199999)
    if(fnlwgt_i == 5):
      fnlwgt = random.randint(13492,99999)
    print(fnlwgt_i)
    print(fnlwgt)
def try11():
    net = -0.01
    PATH = "gender_fairness/census_modify/"+ str(net) +".pt"
    print(PATH)
    print(abs(0-1))

def try12():
    with open('census.pkl','rb') as f:
      data = pickle.load(f)
    print(data)

if __name__ == "__main__":
    try12()