#用于处理sun课题组的数据，处理成可以输入的格式
import torch
import numpy as np 
import ast 

# 处理socrates里的测试数据中的feature部分
def cat_feature():
    input1 = open('census_fairness/sample_sex/data_from_sun/data/data0.txt', 'r')
    line1 = input1.readlines()

    f1 = np.array(ast.literal_eval(line1[0]))
    f1 = torch.from_numpy(f1)

    input2 = open('census_fairness/sample_sex/data_from_sun/data/data1.txt', 'r')
    line2 = input2.readlines()

    f2 = np.array(ast.literal_eval(line2[0]))
    f2 = torch.from_numpy(f2)

    f1 = torch.unsqueeze(f1,dim=0)
    f2 = torch.unsqueeze(f2,dim=0)
    # print(w1.shape)
    # print(w2.shape)
    f = torch.cat((f1,f2),0)

    for i in range(2,100):
        input1 = open(f"census_fairness/sample_sex/data_from_sun/data/data{i}.txt", 'r')
        line1 = input1.readlines()

        f1 = np.array(ast.literal_eval(line1[0]))
        f1 = torch.from_numpy(f1)
        f1 = torch.unsqueeze(f1,dim=0)
        f = torch.cat((f,f1),0)

    torch.save(f, "census_fairness/sample_sex/data_from_sun/data_13_100.pth")
    print(f.shape)

# 处理socrates里的测试数据中的label部分
def cat_output():
    input = open("census_fairness/sample_sex/data_from_sun/data_from_sun.txt", 'r')

    line1 = input.read()
    o1 = np.array(ast.literal_eval(line1))
    line_new = []
    for i in o1:
        if i==0:
            t = [0.0,1.0]
            line_new.append(t)
        if i==1:
            t = [1.0,0.0]
            line_new.append(t)
        else:
            continue
    # print(line_new)
    line_new = np.array(line_new)
    print(line_new[0])
    o1 = torch.from_numpy(line_new)
    print(o1.shape)

    torch.save(o1,"census_fairness/sample_sex/data_from_sun/data_13_1.pth")
    print(o1)


    # PATH1 = f"census_fairness/sample_sex/data_from_sun/data/data0.txt"
    # PATH2 = f"census_fairness/sample_sex/data_from_sun/data/data1.txt"
    # # 修改输入文件
    # test_x1 = np.loadtxt(PATH1)
    # test_X2 = np.loadtxt(PATH2) 
    # tensor_test_x1 = torch.FloatTensor(test_x1.copy())                                                            
    # tensor_test_X2 = torch.FloatTensor(test_X2.copy()) 
    # tensor_x = torch.cat([test_x1,test_X2], dim = 0)

    # for i in range(2,100):
    #     PATH = f"census_fairness/sample_sex/data_from_sun/data/data{i}.txt" 
    #     # 修改输入文件
    #     test_x1 = np.loadtxt(PATH1)
    #     tensor_test_x1 = torch.FloatTensor(test_x1.copy())                                                            
    #     tensor_x = torch.cat([test_x,test_x1], dim = 0)
    # print(tensor_x.shape)

# 处理sun项目的全部census数据集
def split_data_sun():
    input = open("census_fairness/sample_sex/data_from_sun/data_from_sun.txt", 'r')

    lines = input.readlines()
    feature = []
    label = []
    data=[]
    l = []
    for line in lines:
        l.append(line)
    l.pop(0)
     
    for i in l:
        t = np.array(ast.literal_eval(i))
        la = t[13]
        if la==0:
            te = [0.0,1.0]
            label.append(te)
        if la==1:
            te = [1.0,0.0]
            label.append(te)
        fe = np.delete(t,13)
        feature.append(fe)
    # print(label)
    label = np.array(label)
    feature = np.array(feature)
    label = torch.from_numpy(label).float()
    feature = torch.from_numpy(feature).float()

    torch.save(label,"census_fairness/sample_sex/data_from_sun/all_label_13.pth")
    torch.save(feature,"census_fairness/sample_sex/data_from_sun/all_featrue_13.pth")

# 处理sun项目的部分census数据集
def split_data_sun_sample():
    input = open("census_fairness/sample_sex/data_from_sun/data_sample_from_sun.txt", 'r')

    lines = input.readlines()
    feature = []
    label = []
    data=[]
    l = []
    for line in lines:
        l.append(line)
    l.pop(0)
     
    for i in l:
        t = np.array(ast.literal_eval(i))
        la = t[13]
        if la==0:
            te = [0.0,1.0]
            label.append(te)
        if la==1:
            te = [1.0,0.0]
            label.append(te)
        fe = np.delete(t,13)
        feature.append(fe)
    # print(label)
    label = np.array(label)
    feature = np.array(feature)
    label = torch.from_numpy(label).float()
    feature = torch.from_numpy(feature).float()

    torch.save(label,"census_fairness/sample_sex/data_from_sun/sample_label_13.pth")
    torch.save(feature,"census_fairness/sample_sex/data_from_sun/sample_featrue_13.pth")


def tryy():
    # input1 = open('benchmark/rnn/data/jigsaw/labels.txt', 'r')
    # line1 = input1.readlines()

    # f1 = np.array(ast.literal_eval(line1[0]))
    # print(f1.shape)
    for i in range(5):
        print(i)
    print("sss")
    for i in range (5,10):
        print(i)

if __name__ == "__main__":
    print("sss")
    # split_data_sun_sample()
    tryy()