# Census-income数据集进行均匀采样，固定gender(sex)，均匀生成其他特性的数据,去掉education-num
import random
import pandas as pd
import numpy as np

# adult = pd.read_csv(r'data2/adult_sim_3.csv')
# print(type(adult))
# 定义数据表头即参数名(删减education-num和income版本)
headers = ['age', 'workclass', 'fnlwgt',
           'education',  'marital-status', 'occupation',
           'relationship', 'race', 'sex',
           'capital-gain', 'capital-loss',
           'hours-per-week', 'native-country']

# 对特性取值进行独热编码
# 列举变量取值
wc = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
      'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
edu = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
       'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th',
       '10th', 'Doctorate', '5th-6th', 'Preschool']
ms = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
      'Married-spouse-absent', 'Married-AF-spouse']
occu = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
        'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
        'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces','none']
rel = ['Wife', 'Own-child', 'Husband',
       'Not-in-family', 'Other-relative', 'Unmarried']
gen = ['Female','Male']
rc = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
nc = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
      'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
      'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
      'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
      'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
      'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

# headers = ['age', 'workclass', 'fnlwgt',
#            'education',  'marital-status', 'occupation',
#            'relationship', 'race', 'gender',
#            'hours-per-week', 'native-country',]

# # 对特性取值进行独热编码
# # 列举变量取值
# wc = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
#       'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']
# edu = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
#        'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th',
#        '10th', 'Doctorate', '5th-6th', 'Preschool']
# ms = ['Married-civ-spouse', 'Divorced', 'Never-married', ' Separated', 'Widowed',
#       'Married-spouse-absent', 'Married-AF-spouse']
# occu = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
#         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
#         'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
# rel = ['Wife', 'Own-child', 'Husband',
#        'Not-in-family', 'Other-relative', 'Unmarried']
# rc = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
# gen = ['Female','Male']
# nc = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
#       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba',
#       'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
#       'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
#       'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand',
#       'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
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

def gen_male():
      # 创建新的DATAFRAME
      df_male = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(17, 90)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.17,0.372,0.29,0.1,0.068]
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

          capital_gain_i = random.randint(1, 20)
          if(capital_gain_i == 1):
            capital_gain = 0
          if(1 < capital_gain_i <= 20):
            capital_gain = random.randint(1,41310)

          capital_loss_i = random.randint(1, 100)
          if(1 <= capital_loss_i <= 3):
            capital_loss = 0
          if(3 < capital_loss_i <= 100):
            capital_loss = random.randint(1,4536)
          
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
          work_class_p = [0.7214,0.103,0.046,0.031,0.060,0.038,0.0004,0.0002]
          # print(work_class_list)
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.172,0.204,0.0332,0.337,0.0223,0.03,0.041,0.016,0.020,0.012,0.0548,0.005,0.028,0.014,0.010,0.001]
          education = p_random(education_list, education_p)
          education = edu[education]
          # print(education_list)
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.6186,0.082,0.265,0.018,0.007,0.009,0.0004]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.028,0.19293,0.071,0.114,0.139,0.124,0.058,0.069,0.059,0.045,0.072,0.00039,0.027,0.00044,0.00024]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0,0.123,0.612,0.204,0.024,0.037]
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
      
      #     gen_r1 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'White', gender, capital_gain, capital_loss, hr_per_week, native_country]
      
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          gen_male = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
                      race, 'Male', capital_gain, capital_loss,hr_per_week, native_country]
          df_male.loc[i]= gen_male
      #     print(gen_male)
      
      #     gen_female = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
                        # race, 'Female', hr_per_week, native_country]
      
      
          # gen_all = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
          #             race, gender, hr_per_week, native_country]
      
          # adult_gen_male = dict(zip(headers,gen_male))
          # adult_gen_female = dict(zip(headers,gen_female))
          # print(adult_gen_female)
          # print(adult_gen_male)
      
      #   df_female.append(adult_gen_female, ignore_index = True)
      #     df_male.append(gen_male, ignore_index = True)
      
      # df_female.to_csv('gender_fairness/sample_sex/gen_female.csv')
      df_male.to_csv('census_fairness/sample_sex/gen_male.csv')

def gen_female():
      # 创建新的DATAFRAME
      df_female = pd.DataFrame(columns=headers)
      # 进行均匀采样生成新数据
      n = 10000
      for i in range(n):
          # 连续型采样
      
          #均匀采样
          age = random.randint(17, 90)

          fnlwgt_i = [1,2,3,4,5]
          fnlwgt_p = [0.25,0.45,0.25,0.05,0]
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

          capital_gain_i = random.randint(1, 20)
          if(capital_gain_i == 1):
            capital_gain = 0
          if(1 < capital_gain_i <= 20):
            capital_gain = random.randint(1,41310)

          capital_loss_i = random.randint(1, 100)
          if(1 <= capital_loss_i <= 3):
            capital_loss = 0
          if(3 < capital_loss_i <= 100):
            capital_loss = random.randint(1,4536)
          
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
          work_class_p = [0.781071137,0.040065413,0.012878168,0.031582175,0.084219133,0.04946852,0.000511038,0.000304415]
      #     print(sum(work_class_p))
          work_class = p_random(work_class_list, work_class_p)
          work_class = wc[work_class]
          # print(work_class)
      
          education_list = list(np.arange(0, 16))
          education_p = [0.155560098,0.256234669,0.038021259,0.31755928,0.008892069,0.040372036,0.046504497,0.012162715
          ,0.01349141,0.012469338,0.052023712,0.00439493,0.025551922,0.008278823,0.00705233,0.002430908]
      #     print(sum(education_p))
          education = p_random(education_list, education_p)
          education = edu[education]
          # print(education_list)
      
          marital_status_list = list(np.arange(0, 7))
          marital_status_p = [0.151369583,0.258483238,0.44082175,0.058667212,0.070114473,0.019317253,0.001226492]
          marital_status = p_random(marital_status_list, marital_status_p)
          marital_status = ms[marital_status]
      
          occupation_list = list(np.arange(0, 15))
          occupation_p = [0.034852821,0.02207686,0.179681112,0.127555192,0.116823385,0.15239166,0.016762061,0.055498774
           ,0.256745707,0.0066435,0.009198692,0.013798038,0.007767784,0.000,0.000204415]
          occupation = p_random(occupation_list, occupation_p)
          occupation = occu[occupation]
      
          relationship_list = list(np.arange(0, 6))
          relationship_p = [0.143704007,0.20053148,0.000102208,0.364472608,0.039452167,0.251737531]
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
      
      #     gen_r1 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'White', gender, capital_gain, capital_loss, hr_per_week, native_country]
      
      #     gen_r2 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                   'Asian-Pac-Islander', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r3 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Amer-Indian-Eskimo', gender, capital_gain, capital_loss, hr_per_week, native_country]
      #     gen_r4 = [age, work_class, fnlwgt, education,education_num, marital_status, occupation, relationship,
      #                 'Black', gender, capital_gain, capital_loss, hr_per_week, native_country]
          gen_female = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
                      race, 'Female', capital_gain, capital_loss,hr_per_week, native_country]
          df_female.loc[i]= gen_female
      #     print(gen_male)
      
      #     gen_female = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
                        # race, 'Female', hr_per_week, native_country]
      
      
          # gen_all = [age, work_class, fnlwgt, education, marital_status, occupation, relationship,
          #             race, gender, hr_per_week, native_country]
      
          # adult_gen_male = dict(zip(headers,gen_male))
          # adult_gen_female = dict(zip(headers,gen_female))
          # print(adult_gen_female)
          # print(adult_gen_male)
      
      #   df_female.append(adult_gen_female, ignore_index = True)
      #     df_male.append(gen_male, ignore_index = True)
      
      # df_female.to_csv('gender_fairness/sample_sex/gen_female.csv')
      df_female.to_csv('census_fairness/sample_sex/gen_female.csv')
if __name__ == "__main__":
      gen_male()