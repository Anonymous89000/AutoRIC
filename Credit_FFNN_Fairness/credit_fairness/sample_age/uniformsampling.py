# Census-income数据集进行均匀采样，固定race，均匀生成其他特性的数据
import random
import pandas as pd
import numpy as np

# adult = pd.read_csv(r'data2/adult_sim_3.csv')
# print(type(adult))
# 定义数据表头即参数名(删减education-num和income版本)
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

# 对特性取值进行独热编码
# 列举变量取值
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

def gen_a1():
      # 创建新的DATAFRAME
      df_a1 = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(17, 30)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.2,0.45,0.2,0.1,0.05]
          fnlwgt_i = p_random(fnlwgt_i, fnlwgt_p)
          if(fnlwgt_i == 5):
            fnlwgt = random.randint(400000,1490400)
          if(fnlwgt_i == 4):
            fnlwgt = random.randint(300000,399999)
          if(fnlwgt_i == 3):
            fnlwgt = random.randint(200000,299999)
          if(fnlwgt_i == 2):
            fnlwgt = random.randint(100000,199999)
          if(fnlwgt_i == 1):
            fnlwgt = random.randint(13492,99999)
          
          education_num_i =  [1,2,3]
          education_num_p = [0.13,0.6,0.27]
          education_num_i = p_random(education_num_i, education_num_p)
          if(education_num_i == 3):
            education_num = random.randint(12,19)
          if(education_num_i == 2):
            education_num = random.randint(8,11)
          if(education_num_i == 1):
            education_num = random.randint(1,7)
          
          capital_gain_i =  [1,2,3]
          capital_gain_p = [0.96,0.04,0.0]
          capital_gain_i = p_random(capital_gain_i, capital_gain_p)
          if(capital_gain_i == 3):
            capital_gain = random.randint(20001,41310)
          if(capital_gain_i == 2):
            capital_gain = random.randint(10001,20000)
          if(capital_gain_i == 1):
            capital_gain = random.randint(0,10000)

          capital_loss_i =  [1,2,3,4]
          capital_loss_p = [0.97,0,0.03,0.0]
          capital_loss_i = p_random(capital_loss_i, capital_loss_p)
          if(capital_loss_i == 4):
            capital_loss = random.randint(2001,4536)
          if(capital_loss_i == 3):
            capital_loss = random.randint(1801,2000)
          if(capital_loss_i == 2):
            capital_loss = random.randint(301,1800)
          if(capital_loss_i == 1):
            capital_loss = random.randint(0,300)

          
          hr_per_week_i = [1,2,3,4]
          hr_per_week_p = [0.12,0.68,0.18,0.02]
          hr_per_week_i = p_random(hr_per_week_i, hr_per_week_p)
          if(hr_per_week_i == 4):
            hr_per_week = random.randint(62,99)
          if(hr_per_week_i == 3):
            hr_per_week = random.randint(48,61)
          if(hr_per_week_i == 2):
            hr_per_week = random.randint(32,47)
          if(hr_per_week_i == 1):
            hr_per_week = random.randint(1,31)
      
          # 离散型采样
          work_class_list = list(np.arange(0, 8))
          work_class_p = [0.853706789,0.036443149,0.012390671,0.016139109,0.044356518,
          0.035714286,0.000520616,0.000728863]
          # print(work_class_list)
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.153581841,0.27696793,0.058725531,0.319970845,0.006143274,
          0.031028738,0.038421491,0.014785506,0.010620575,0.021449396,0.019471054,
          0.002915452,0.034048313,0.002082466,0.008538109,0.001249579]
      #     print(sum(education_p))
          education = p_random(education_list, education_p)
          education = edu[education]
          
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.222823823,0.047376093,0.697105373,0.02030404,0.001353603,0.009683465,0.001354603]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.032798834,0.109537693,0.153685964,0.141607663,0.072990421,0.097980008,0.077675968,
          0.06726364,0.149208663,0.028217409,0.04154519,0.005622657,0.020512287,0.00062474,0.000729863]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0.032902957,0.37786339,0.179196168,0.290191587,0.052582257,0.06736364]
          relationship = p_random(relationship_list, relationship_p)
          relationship = rel[relationship]

          #均匀采样
          race_list = list(np.arange(0, 5))
          rac = random.choice(race_list)
          race = rc[rac]
          #均匀采样
          gender_list = list(np.arange(0, 2))
          ge = random.choice(gender_list)
          gender = gen[ge]
          #均匀采样
          native_country_list = list(np.arange(0, 41))
          native_country = random.choice(native_country_list)
          native_country = nc[native_country]
      
          gen_a1 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
                      race, gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          df_a1.loc[i] = gen_a1

      df_a1.to_csv('census_fairness/sample_age/gen_a1.csv')

def gen_a2():
      # 创建新的DATAFRAME
      df_a2 = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(31, 45)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.2,0.4,0.2,0.15,0.05]
          fnlwgt_i = p_random(fnlwgt_i, fnlwgt_p)
          if(fnlwgt_i == 5):
            fnlwgt = random.randint(400000,1490400)
          if(fnlwgt_i == 4):
            fnlwgt = random.randint(300000,399999)
          if(fnlwgt_i == 3):
            fnlwgt = random.randint(200000,299999)
          if(fnlwgt_i == 2):
            fnlwgt = random.randint(100000,199999)
          if(fnlwgt_i == 1):
            fnlwgt = random.randint(13492,99999)
          
          education_num_i =  [1,2,3]
          education_num_p = [0.1,0.5,0.4]
          education_num_i = p_random(education_num_i, education_num_p)
          if(education_num_i == 3):
            education_num = random.randint(12,19)
          if(education_num_i == 2):
            education_num = random.randint(8,11)
          if(education_num_i == 1):
            education_num = random.randint(1,7)
          
          capital_gain_i =  [1,2,3]
          capital_gain_p = [0.96,0.04,0.0]
          capital_gain_i = p_random(capital_gain_i, capital_gain_p)
          if(capital_gain_i == 3):
            capital_gain = random.randint(20001,41310)
          if(capital_gain_i == 2):
            capital_gain = random.randint(10001,20000)
          if(capital_gain_i == 1):
            capital_gain = random.randint(0,10000)

          capital_loss_i =  [1,2,3,4]
          capital_loss_p = [0.97,0,0.03,0.0]
          capital_loss_i = p_random(capital_loss_i, capital_loss_p)
          if(capital_loss_i == 4):
            capital_loss = random.randint(2001,4536)
          if(capital_loss_i == 3):
            capital_loss = random.randint(1801,2000)
          if(capital_loss_i == 2):
            capital_loss = random.randint(301,1800)
          if(capital_loss_i == 1):
            capital_loss = random.randint(0,300)

          
          hr_per_week_i = [1,2,3,4]
          hr_per_week_p = [0.12,0.68,0.18,0.02]
          hr_per_week_i = p_random(hr_per_week_i, hr_per_week_p)
          if(hr_per_week_i == 4):
            hr_per_week = random.randint(62,99)
          if(hr_per_week_i == 3):
            hr_per_week = random.randint(48,61)
          if(hr_per_week_i == 2):
            hr_per_week = random.randint(32,47)
          if(hr_per_week_i == 1):
            hr_per_week = random.randint(1,31)
      
          # 离散型采样
          work_class_list = list(np.arange(0, 8))
          work_class_p = [0.723744292,0.086335194,0.035261289,0.034415694,0.074243193,0.046000338,0,0]
          # print(work_class_list)
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.191527144,0.205310333,0.021478099,0.325469305,0.022239134,0.04109589,0.052342297,
          0.012092001,0.010992728,0.008371385,0.065026213,0.004312532,0.018011162,0.013022155,0.007441231,0.001268392]
      #     print(sum(education_p))
          education = p_random(education_list, education_p)
          education = edu[education]
          
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.549720954,0.183578556,0.202012515,0.041434128,0.009893455,0.012768476,0.000591916]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.032386268,0.154405547,0.082107221,0.10062574,0.153475393,0.153813631,0.0335701,0.067224759,
          0.111533908,0.031117876,0.055217318,0.002198546,0.022154575,0.000169119,0]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0.057415863,0.059022493,0.487231524,0.244461356,0.017419246,0.134449518]
          relationship = p_random(relationship_list, relationship_p)
          relationship = rel[relationship]

          #均匀采样
          race_list = list(np.arange(0, 5))
          rac = random.choice(race_list)
          race = rc[rac]
          #均匀采样
          gender_list = list(np.arange(0, 2))
          ge = random.choice(gender_list)
          gender = gen[ge]
          #均匀采样
          native_country_list = list(np.arange(0, 41))
          native_country = random.choice(native_country_list)
          native_country = nc[native_country]
      
          gen_a2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
                      race, gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          df_a2.loc[i] = gen_a2

      df_a2.to_csv('census_fairness/sample_age/gen_a2.csv')

def gen_a3():
      # 创建新的DATAFRAME
      df_a3 = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(46, 60)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.25,0.35,0.2,0.15,0.05]
          fnlwgt_i = p_random(fnlwgt_i, fnlwgt_p)
          if(fnlwgt_i == 5):
            fnlwgt = random.randint(400000,1490400)
          if(fnlwgt_i == 4):
            fnlwgt = random.randint(300000,399999)
          if(fnlwgt_i == 3):
            fnlwgt = random.randint(200000,299999)
          if(fnlwgt_i == 2):
            fnlwgt = random.randint(100000,199999)
          if(fnlwgt_i == 1):
            fnlwgt = random.randint(13492,99999)
          
          education_num_i =  [1,2,3]
          education_num_p = [0.1,0.55,0.35]
          education_num_i = p_random(education_num_i, education_num_p)
          if(education_num_i == 3):
            education_num = random.randint(12,19)
          if(education_num_i == 2):
            education_num = random.randint(8,11)
          if(education_num_i == 1):
            education_num = random.randint(1,7)
          
          capital_gain_i =  [1,2,3]
          capital_gain_p = [0.96,0.04,0.0]
          capital_gain_i = p_random(capital_gain_i, capital_gain_p)
          if(capital_gain_i == 3):
            capital_gain = random.randint(20001,41310)
          if(capital_gain_i == 2):
            capital_gain = random.randint(10001,20000)
          if(capital_gain_i == 1):
            capital_gain = random.randint(0,10000)

          capital_loss_i =  [1,2,3,4]
          capital_loss_p = [0.97,0,0.03,0.0]
          capital_loss_i = p_random(capital_loss_i, capital_loss_p)
          if(capital_loss_i == 4):
            capital_loss = random.randint(2001,4536)
          if(capital_loss_i == 3):
            capital_loss = random.randint(1801,2000)
          if(capital_loss_i == 2):
            capital_loss = random.randint(301,1800)
          if(capital_loss_i == 1):
            capital_loss = random.randint(0,300)

          
          hr_per_week_i = [1,2,3,4]
          hr_per_week_p = [0.1,0.7,0.18,0.02]
          hr_per_week_i = p_random(hr_per_week_i, hr_per_week_p)
          if(hr_per_week_i == 4):
            hr_per_week = random.randint(62,99)
          if(hr_per_week_i == 3):
            hr_per_week = random.randint(48,61)
          if(hr_per_week_i == 2):
            hr_per_week = random.randint(32,47)
          if(hr_per_week_i == 1):
            hr_per_week = random.randint(1,31)
      
          # 离散型采样
          work_class_list = list(np.arange(0, 8))
          work_class_p = [0.645607962,0.114380499,0.057262368,0.046733016,
          0.088850426,0.046877254,0.000288475,0]
          # print(work_class_list)
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.154190105,0.185633925,0.025674311,0.333910284,
          0.02452041,0.028991778,0.038367229,0.017164287,0.02827059,0.007644598,
          0.082648204,0.007500361,0.029568729,0.022068369,0.012115967,0.001740852]
      #     print(sum(education_p))
          education = p_random(education_list, education_p)
          education = edu[education]
          
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.629597577,0.198903793,0.075724794,0.032309246,0.048031155,0.015289197,0.000145238]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.026683975,0.144093466,0.07976345,0.112938122,0.171787105,0.149718736,0.023366508,0.06543968,
          0.108466753,0.031876533,0.060724073,0.004615607,0.020481754,0.000144238,0]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0.051348623,0.019183615,0.574066061,0.218231646,0.018173951,0.118996106]
          relationship = p_random(relationship_list, relationship_p)
          relationship = rel[relationship]

          #均匀采样
          race_list = list(np.arange(0, 5))
          rac = random.choice(race_list)
          race = rc[rac]
          #均匀采样
          gender_list = list(np.arange(0, 2))
          ge = random.choice(gender_list)
          gender = gen[ge]
          #均匀采样
          native_country_list = list(np.arange(0, 41))
          native_country = random.choice(native_country_list)
          native_country = nc[native_country]
      
          gen_a3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
                      race, gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          df_a3.loc[i] = gen_a3

      df_a3.to_csv('census_fairness/sample_age/gen_a3.csv')

def gen_a4():
      # 创建新的DATAFRAME
      df_a4 = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(61, 90)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.3,0.35,0.20,0.11,0.05]
          # print(sum(fnlwgt_p))
          fnlwgt_i = p_random(fnlwgt_i, fnlwgt_p)
          if(fnlwgt_i == 5):
            fnlwgt = random.randint(400000,1490400)
          if(fnlwgt_i == 4):
            fnlwgt = random.randint(300000,399999)
          if(fnlwgt_i == 3):
            fnlwgt = random.randint(200000,299999)
          if(fnlwgt_i == 2):
            fnlwgt = random.randint(100000,199999)
          if(fnlwgt_i == 1):
            fnlwgt = random.randint(13492,99999)
          
          education_num_i =  [1,2,3]
          education_num_p = [0.2,0.5,0.3]
          education_num_i = p_random(education_num_i, education_num_p)
          if(education_num_i == 3):
            education_num = random.randint(12,19)
          if(education_num_i == 2):
            education_num = random.randint(8,11)
          if(education_num_i == 1):
            education_num = random.randint(1,7)
          
          capital_gain_i =  [1,2,3]
          capital_gain_p = [0.96,0.04,0.0]
          capital_gain_i = p_random(capital_gain_i, capital_gain_p)
          if(capital_gain_i == 3):
            capital_gain = random.randint(20001,41310)
          if(capital_gain_i == 2):
            capital_gain = random.randint(10001,20000)
          if(capital_gain_i == 1):
            capital_gain = random.randint(0,10000)

          capital_loss_i =  [1,2,3,4]
          capital_loss_p = [0.97,0,0.03,0.0]
          capital_loss_i = p_random(capital_loss_i, capital_loss_p)
          if(capital_loss_i == 4):
            capital_loss = random.randint(2001,4536)
          if(capital_loss_i == 3):
            capital_loss = random.randint(1801,2000)
          if(capital_loss_i == 2):
            capital_loss = random.randint(301,1800)
          if(capital_loss_i == 1):
            capital_loss = random.randint(0,300)

          
          hr_per_week_i = [1,2,3,4]
          hr_per_week_p = [0.2,0.6,0.15,0.05]
          hr_per_week_i = p_random(hr_per_week_i, hr_per_week_p)
          if(hr_per_week_i == 4):
            hr_per_week = random.randint(62,99)
          if(hr_per_week_i == 3):
            hr_per_week = random.randint(48,61)
          if(hr_per_week_i == 2):
            hr_per_week = random.randint(32,47)
          if(hr_per_week_i == 1):
            hr_per_week = random.randint(1,31)
      
          # 离散型采样
          work_class_list = list(np.arange(0, 8))
          work_class_p = [0.582502769,0.185492802,0.07807309,0.031561462,
          0.081395349,0.03709856,0.003875969,0]
          # print(work_class_list)
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.130121816,0.168881506,0.029346622,0.334440753,
          0.027685493,0.012735327,0.029346622,0.028239203,0.071982281,
          0.010520487,0.054263566,0.011074197,0.042635659,0.026578073,0.018826135,0.003332259]
      #     print(sum(education_p))
          education = p_random(education_list, education_p)
          education = edu[education]
          
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.586932447,0.11627907,0.067552602,0.016611296,0.201550388,0.011074197,0]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.016057586,0.084717608,0.117386489,0.138981174,0.157807309,0.132890365,
          0.024916944,0.03986711,0.120155039,0.071428571,0.054817276,0.017165006,0.023809524,0,0]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0.031007752,0.006090808,0.553709856,0.296788483,0.028792913,0.083610188]
          relationship = p_random(relationship_list, relationship_p)
          relationship = rel[relationship]

          #均匀采样
          race_list = list(np.arange(0, 5))
          rac = random.choice(race_list)
          race = rc[rac]
          #均匀采样
          gender_list = list(np.arange(0, 2))
          ge = random.choice(gender_list)
          gender = gen[ge]
          #均匀采样
          native_country_list = list(np.arange(0, 41))
          native_country = random.choice(native_country_list)
          native_country = nc[native_country]
      
          gen_a4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
                      race, gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          df_a4.loc[i] = gen_a4

      df_a4.to_csv('census_fairness/sample_age/gen_a4.csv')


if __name__ == "__main__":
      # gen_a1()
      # gen_a2()
      # gen_a3()
      gen_a4()

#     df_genr1.loc[i] = gen_r1
#     df_genr2.loc[i] = gen_r2
#     df_genr3.loc[i] = gen_r3
#     df_genr4.loc[i] = gen_r4

# df_genr1.to_csv('gender_fairness/sample_race/gen_r1.csv')
# df_genr2.to_csv('gender_fairness/sample_race/gen_r2.csv')
# df_genr3.to_csv('gender_fairness/sample_race/gen_r3.csv')
# df_genr4.to_csv('gender_fairness/sample_race/gen_r4.csv')


