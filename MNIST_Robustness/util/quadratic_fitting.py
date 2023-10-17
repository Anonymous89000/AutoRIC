import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model



def stdError_func(y_test, y):
  """ helper function: 计算标准差
  """
  return np.sqrt(np.mean((y_test - y) ** 2))


def R2_1_func(y_test, y):
  return 1 - ((y_test - y) ** 2).sum() / ((y.mean() - y) ** 2).sum()


def R2_2_func(y_test, y):
  y_mean = np.array(y)
  y_mean[:] = y.mean()
  return 1 - stdError_func(y_test, y) / stdError_func(y_mean, y)

def fitting(n, filename):
    df= pd.read_csv(filename)

    x = np.array(df.iloc[:,0:n].values)

    y = np.array(df.iloc[:,n].values)

    degree=2
    poly_reg =PolynomialFeatures(degree=degree) #2次多项式
    # 该拟合方式的原理应该是和我们的论文中一样的
    # 此处的poly_reg的含义:一个包含多项式特征信息的对象 告诉函数fit_transform要拟合出什么样的特征

    X_poly =poly_reg.fit_transform(x)
    # X_poly就是利用现有特征计算出的多项式特征
    #print(X_poly)
    lin_reg_2=linear_model.LinearRegression()
    lin_reg_2.fit(X_poly,y)
    predict_y = lin_reg_2.predict(X_poly)

    strError = stdError_func(predict_y, y)
    R2_1 = R2_1_func(predict_y, y)
    R2_2 = R2_2_func(predict_y, y)
    score = lin_reg_2.score(X_poly, y)  ##sklearn中自带的模型评估，与R2_1逻辑相同


    # print("coefficients", lin_reg_2.coef_)
    # print("intercept", lin_reg_2.intercept_)
    # print('degree={}: strError={:.2f}, R2_1={:.2f},  R2_2={:.2f}, clf.score={:.2f}'.format(
    #     degree, strError, R2_1, R2_2, score))

    #下面将coefficients转化为Q1,b1,c1

    coefficients=lin_reg_2.coef_
    if len(coefficients)!=1+n+n*(n+1)/2:
        print("error!")
        exit()
    intercept=lin_reg_2.intercept_
    c1=coefficients[0]+intercept
    b1=coefficients[1:n+1]
    q=coefficients[n+1:]
    #现在变量q中存储的就是 xixj的系数 它们的排列顺序是 x1x1,x1x2,x1x3,...
    
    Q1=np.zeros(n*n).reshape(n,n)
    for i in range(0,n):
        for j in range(i,n):
            #Q1[i][j]中应该存储x_{i+1}*x_{j+1}的对应系数
            #而x_m*x_n对应的系数在q中的索引为
            if(i==j):
                Q1[i][j]=q[int(( (2*n-i+1)*i )/2)+j-i]
            else:
                Q1[i][j] = q[int(((2 * n - i + 1) * i) / 2) + j - i]/2
    Q1=Q1 + Q1.T - np.diag(Q1.diagonal())

    return (Q1,b1,c1)

