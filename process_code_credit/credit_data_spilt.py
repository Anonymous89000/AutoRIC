#将删除了education-num的census income数据集进行编码 并划分测试集数据集，处理成可以放入网络的数据文件。

## 导入相关数据包  # 数据处理
import time, datetime, math, random
from io import StringIO
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 可视化
import matplotlib.pyplot as plt  # 绘图
import matplotlib
import time
import datetime

from sklearn.model_selection import StratifiedShuffleSplit #分层交叉验证
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


 # 定义数据表头即参数名
headers = ['status', 'duration', 'credit_history',
           'purpose', 'amount',
           'savings', 'employment_duration',
           'installment_rate', 'personal_status_sex', 'other_debtors',
           'present_residence', 'property',
           'age', 'other_installment_plans',
           'housing', 'number_credits',
           'job', 'people_liable',
           'telephone', 'foreign_worker',
           'credit_risk'] 

#[(0,3),(1,80),(0,4),(0,10),(1,200),(0,4),(0,4),(1,4),(0,1),(0,2),
# (1,4),(0,3),(1,8),(0,2),(0,2),(1,4),(0,3),(1,2),(0,1),(0,1)]
# 列举变量取值
st = ['A11', 'A12', 'A13', 'A14']#(0,3)
#dura[4,72]->(0,80)
cre_his = ['A30', 'A31', 'A32', 'A33', 'A34']#(0,4) 
pur = ['A40', 'A41', 'A42', 'A43', 'A44','A45', 'A46', 'A47', 'A48',
      'A49', 'A410'] #(0,10)                   
#amo[250,18424]-（1,200）
sav = [ 'A61', 'A62', 'A63', 'A64', 'A65']#(0,4)
employ_dura = [ 'A71', 'A72', 'A73', 'A74', 'A75']#(0,4)
#ins_rate = ['1', '2', '3', '4'](1,4)
per_sta_sex = ['A91', 'A92', 'A93', 'A94']#(0,1)？文件中有A95
other_deb = ['A101', 'A102', 'A103']#(0,2)
#pre_res = ['1', '2', '3', '4'](1,4)
pro = ['A121', 'A122', 'A123', 'A124']#(0,3)
#age [19,75](1,8)
other_ins_pls = ['A141', 'A142', 'A143']#(0,2)
hou = ['A151', 'A152', 'A153']#(0,2)
#num_cre = ['1', '2', '3', '4'](1,4)SS
job = ['A171', 'A172', 'A173', 'A174']#(0,3)
#peo_lia = ['1', '2'] (1,2)
tele = ['A191', 'A192']#(0,1)
for_work = ['A201', 'A202']#(0,1)

cre_risk = ['1', '2']

feature = []
out = []

# 打开文件
with open('data/initial/german.data', 'r', encoding='utf-8') as file:
    # 逐行读取文件内容
    for line in file:
        # 对每一行进行处理
        # 例如，将每一行拆分成单词并打印出来
        features = line.split()
        features[0] = st.index(features[0])
        features[1] = np.clip(int(features[1]), 1, 80)
        features[2] = cre_his.index(features[2])
        features[3] = pur.index(features[3])
        features[4] = np.clip(int(features[4]) /  100, 1, 200)
        features[5] = sav.index(features[5])
        features[6] = employ_dura.index(features[6])
        features[7] = np.clip(int(features[7]), 1, 4)
        features[8] = per_sta_sex.index(features[8])
        features[9] = other_deb.index(features[9])
        features[10] = np.clip(int(features[10]), 1, 4)
        features[11] = pro.index(features[11])
        features[12] = np.clip(int(features[12]) / 10, 1, 8)#age
        features[13] = other_ins_pls.index(features[13])
        features[14] = hou.index(features[14])
        features[15] = np.clip(int(features[15]), 1, 4)
        features[16] = job.index(features[16])
        features[17] = np.clip(int(features[17]), 1, 2)
        features[18] = tele.index(features[18])
        features[19] = for_work.index(features[19])

        feature.append(features[:20])
        
        features[20] = cre_risk.index(features[20])#
        s = features[20]
        if(s==0):
            out.append([0,1])
        else:
            out.append([1,0])


feature = np.asarray(feature)
out = np.asarray(out)

np.savetxt("data/data_numeric/feature.txt", feature, fmt="%d",delimiter=",")
np.savetxt("data/data_numeric/out.txt", out, fmt="%d",delimiter=",")

# 划分数据集
split = StratifiedShuffleSplit(n_splits=2,train_size=0.75)
for train_index,test_index in split.split(feature,out):
    trainx,testx = feature[train_index],feature[test_index]
    trainy,testy = out[train_index],out[test_index]

print(trainy.shape)
print(trainx.shape)
print(testy.shape)
print(testx.shape)
np.savetxt("data/data_numeric/trainx.txt" , trainx, fmt="%d")
np.savetxt("data/data_numeric/trainy.txt" , trainy, fmt="%d")
np.savetxt("data/data_numeric/testx.txt" , testx, fmt="%d")
np.savetxt("data/data_numeric/testy.txt" , testy, fmt="%d")





