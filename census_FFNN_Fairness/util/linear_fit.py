import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import time


def linear_fit(n,filename):

    netname='Ltestnet222'
    df= pd.read_csv(filename)
    x = np.array(df.iloc[:,0:n].values)

    y = np.array(df.iloc[:,n].values)

    cft = linear_model.LinearRegression()
    print(x.shape)
    cft.fit(x, y) #

    print("model coefficients", cft.coef_)
    print("model intercept", cft.intercept_)

    predict_y =  cft.predict(x)
    score = cft.score(x, y) ##sklearn中自带的模型评估，与R2_1逻辑相同
    filename_c='../linear_para/'+netname+'_c.txt'
    filename_b='../linear_para/'+netname+'_b.txt'
    np.savetxt(filename_c,cft.coef_)
    np.savetxt(filename_b,[cft.intercept_])

    return  (cft.coef_,cft.intercept_)

if __name__=='__main__':
    start_time=time.time()
    #linear_fit(128,'../fit_data/pnet_a_0_128_paradata.csv')
    linear_fit(222,'../fit_data/paradata222_1.csv')
    end_time = time.time()

    # 计算运行时间
    elapsed_time = end_time - start_time
    print(f"拟合块运行时间：{elapsed_time} 秒")