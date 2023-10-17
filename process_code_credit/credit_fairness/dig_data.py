# 寻找原始数据集中的规律
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 可视化
import matplotlib.pyplot as plt  # 绘图
import matplotlib
import time
import datetime
import seaborn as sns
import math
sns.set_style('whitegrid')

from sklearn.model_selection import StratifiedShuffleSplit #分层交叉验证
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['MicroSoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def split():    
    # 定义数据表头即参数名
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

    adult = pd.read_csv(r'data/initial/german.csv')
    # 将数据按照gender值分成两部分
    grouped = adult.groupby(['age'])
    # 按组返回两部分数据
    male = grouped.get_group(('male'))
    female = grouped.get_group(('female')) 

    # 转成numpy格式
    male = male.values
    print(male.shape)
    np.savetxt("gender_fairness/male.txt",male)
    female = female.values
    print(female.shape)
    np.savetxt("gender_fairness/female.txt",female)

# #定义函数来显示柱子上的数值
# def autolabel(g,data):
#     for index,row in data.iterrows():
#     #采用iterrows方法对这个dataframe进行遍历即可
#         g.text(row.name,row.nums,row.nums,ha="center")
    
# 绘制分布图
def plot_distribution(dataset, cols, width, height, hspace, wspace):
    fig = plt.figure(figsize = (width, height))
    fig.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=wspace,hspace=hspace)
    rows = math.ceil(dataset.shape[1] / cols)
    for i,column in enumerate(dataset.columns):
        # print(i)
        # print(column)
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:            
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            # print(g.get_yticklabels())
            
            g.set(yticklabels = substrings)
            # print(g.containers[0])
            g.bar_label(g.containers[0])
            # print(dataset[column])            
            plt.xticks(rotation = 25)
        else:
            g = sns.distplot(dataset[column])            
            plt.xticks(rotation = 25)
            # g.bar_label(g.containers[0])
    plt.tight_layout()
    plt.show()
    # plt.savefig("gender_fairness/female.txt")

if __name__ == "__main__":
    adult = pd.read_csv(r'data/initial/german.csv')
     # 将数据按照gender值分成两部分
    grouped = adult.groupby(['age'])#19-75

    a1 = (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)#19-30
    a2 = (32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45)#31-45
    a3 = (47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60)#46-60
    a4 = (62, 63, 64, 65, 66, 67, 68,  70, 74, 75) #去掉69的61-75

    # a1 = ('17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30')
    # a2 = ('31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45')
    # a3 = ('46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60')
    # a4 = ('61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
    # '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90') 
    
    age1 = grouped.get_group(19)
    # 按组返回两部分数据
    for a in a1:
        temp = grouped.get_group(a)
        age1 = pd.concat([age1, temp])
    
    age2 = grouped.get_group(31)
    # 按组返回两部分数据
    for a in a2:
        temp = grouped.get_group(a)
        age2 = pd.concat([age2, temp])


    age3 = grouped.get_group(46)
    # 按组返回两部分数据
    for a in a3:
        temp = grouped.get_group(a)
        age3 = pd.concat([age3, temp])

    age4 = grouped.get_group(61)
    # 按组返回两部分数据
    for a in a4:
        temp = grouped.get_group(a)
        age4 = pd.concat([age4, temp])
   
    # age1 = grouped.get_group((17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30))
    # age2 = grouped.get_group(a2)
    # age3 = grouped.get_group(a3)
    # age4 = grouped.get_group(a4)

    # # 按组返回两部分数据
    # r1 = grouped.get_group(a1)
    # r2 = grouped.get_group(('Asian-Pac-Islander'))
    # r3 = grouped.get_group(('Amer-Indian-Eskimo'))
    # r4 = grouped.get_group(('Black'))
    # r5 = grouped.get_group(('Other'))
  
    plot_distribution(age1, cols=3, width=24, height=20, hspace=0.2, wspace=0.5)
    # for a,b,i in zip(x,y,range(len(x))): # zip 函数
    #     plt.text(a,b+0.01,"%.2f"%y[i],ha='center',fontsize=20) # plt.text 函数